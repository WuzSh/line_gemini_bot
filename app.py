# app.py
# 2025.11.18

import os
import json
import time
import requests
from threading import Thread
from collections import defaultdict

from flask import Flask, request, abort
from dotenv import load_dotenv

# LINE SDK
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage
from linebot.exceptions import InvalidSignatureError

# Google GenAI (Gemini) client
from google import genai

load_dotenv()
app = Flask(__name__)

# ====== 環境変数 ======
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Optional: Google Custom Search (for web info enrichment)
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")  # optional
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX")          # optional

# Safety / behavior tuning
MAX_HISTORY = int(os.getenv("MAX_HISTORY", 10))
USE_WEB_SEARCH = bool(GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX)
GENI_MODEL = os.getenv("GENI_MODEL", "gemini-2.0-flash")  # fast model by default

# basic checks
if not (LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET and GEMINI_API_KEY):
    raise RuntimeError("Set LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET, GEMINI_API_KEY in environment")

line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ====== メモリと重複防止 ======
# conversation_memory: target_id -> {"history":[...], "phase":"empathy"}
# target_id is userId or groupId or roomId depending on source
conversation_memory = defaultdict(lambda: {"history": [], "phase": "empathy"})
# processed event ids to avoid duplicate processing
processed_event_ids = set()

# ====== 緊急キーワード（簡易） ======
EMERGENCY_KEYWORDS = ["自殺", "死にたい", "殺す", "放火", "爆弾", "薬を飲む"]
ESCALATE_MESSAGE = (
    "緊急の可能性があります。今すぐお住まいの地域の緊急連絡先に連絡するか、"
    "専門の相談窓口へご連絡ください。必要であれば、日本の相談窓口をご案内します。"
)

# ====== プロンプトテンプレート（共通） ======
BASE_RULES = """
あなたは、共感的で落ち着いた心理カウンセリングを行います。以下のルールを必ず守ってください。

1) 自己紹介するときは、「鳥神明」、「みなさんのお友達」とだけ答える。
2) ユーザーが質問をした場合は、**まず明確な回答を行うこと**。質問だけで返すことは禁止する。
3) 回答の後で補助的な質問を行う場合でも、**1ターンにつき質問は最大1つ**までとする。
4) 医療的・法的・診断的な助言は行わず、必要に応じて専門機関に案内する。
5) ユーザーの言葉を否定しないこと。必ず肯定的・受容的なトーンで答える。
6) 必要に応じて外部の信頼できる情報を参照して要約し、ユーザーに分かりやすく伝える（ただし、専門家の断定は避ける）。
"""

PHASE_INSTRUCTIONS = {
    "empathy": "現在は【共感フェーズ】です。まず感情を受け止め、励ます表現と短い共感文で始めてください。",
    "awareness": "現在は【気づきフェーズ】です。やさしく背景や理由を探る質問やリフレームを促す表現を使ってください。",
    "reconstruction": "現在は【再構築フェーズ】です。小さな実行可能な行動案や自己選択を促す表現を使ってください。"
}

# ====== ユーティリティ: Google Custom Search（任意） ======
def perform_web_search(query, max_results=3):
    if not USE_WEB_SEARCH:
        return []
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_CSE_API_KEY,
            "cx": GOOGLE_CSE_CX,
            "q": query,
            "num": max_results,
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])[:max_results]
        results = []
        for it in items:
            results.append({
                "title": it.get("title"),
                "snippet": it.get("snippet"),
                "link": it.get("link")
            })
        return results
    except Exception:
        return []


# ====== フェーズ遷移ロジック（単純ルール） ======
def detect_phase_from_text(current_phase, user_text, history_len):
    text = user_text.lower()
    if current_phase == "empathy":
        if any(k in text for k in ["なぜ", "どうして", "理由", "意味"]) or history_len >= 3:
            return "awareness"
    if current_phase == "awareness":
        if any(k in text for k in ["自分", "自由", "選ぶ", "したい", "できる"]):
            return "reconstruction"
    return current_phase


# ====== プロンプト組み立て ======
def build_prompt_with_context(target_id, user_text):
    memory = conversation_memory[target_id]
    history = memory["history"][-MAX_HISTORY:]
    phase = memory["phase"]

    web_items = []
    if USE_WEB_SEARCH:
        web_items = perform_web_search(user_text, max_results=3)

    history_text = ""
    for turn in history[-6:]:
        role = "ユーザー" if turn["role"] == "user" else "AI"
        history_text += f"{role}: {turn['content']}\n"

    web_text = ""
    if web_items:
        web_text = "\n\n参考にした外部情報（要約）:\n"
        for i, it in enumerate(web_items, start=1):
            web_text += f"{i}. {it['title']} — {it['snippet']}\n"

    prompt = (
        BASE_RULES
        + "\n\n"
        + PHASE_INSTRUCTIONS.get(phase, "")
        + "\n\n"
        + "これまでの会話:\n"
        + history_text
        + web_text
        + f"\nユーザー: {user_text}\n\n"
        + "AIはまず【質問があれば回答】を行い、その後必要なら補助質問は最大1つにとどめること。\n"
        + "出力は日本語で、簡潔に、優しい口調で答えてください。\n\nAI:"
    )
    return prompt


# ====== 生成後チェック: 質問だけで終わっていないかを補正 ======
def ensure_answer_not_only_question(reply_text, user_text):
    trimmed = reply_text.strip()
    if ("?" in trimmed or "？" in trimmed) and len(trimmed) < 60:
        return f"まず結論を言うと、{_simple_answer_hint(user_text)}\n\n{trimmed}"
    return trimmed

def _simple_answer_hint(user_text):
    if "なぜ" in user_text or "どうして" in user_text:
        return "多くの場合、その背景には複数の要因が考えられます。"
    if "どうやって" in user_text or "どうすれば" in user_text:
        return "まずは小さな一歩から始めるのが現実的です。"
    return "落ち着いて一つずつ整理することで、次の一歩が見えてくるかもしれません。"


# ====== 非同期処理ワークフロー（target_id対応） ======
def process_and_push(target_id, original_text, reply_token=None):
    """
    target_id: userId or groupId or roomId
    """
    try:
        memory = conversation_memory[target_id]
        current_phase = memory["phase"]
        history = memory["history"]

        new_phase = detect_phase_from_text(current_phase, original_text, len(history))
        memory["phase"] = new_phase

        prompt = build_prompt_with_context(target_id, original_text)

        # Call Gemini
        try:
            resp = genai_client.models.generate_content(
                model=GENI_MODEL,
                contents=prompt,
            )
            reply_text = getattr(resp, "text", str(resp)).strip()
        except Exception as e:
            print("GenAI call failed:", e)
            reply_text = "すみません、ただいま外部情報の取得に失敗しました。もう一度お願いできますか？"

        reply_text = ensure_answer_not_only_question(reply_text, original_text)

        # update memory using target_id key so group chats keep shared context
        memory["history"].append({"role": "user", "content": original_text})
        memory["history"].append({"role": "assistant", "content": reply_text})
        if len(memory["history"]) > MAX_HISTORY * 2:
            memory["history"] = memory["history"][-MAX_HISTORY*2 :]

        # push result to the correct target (user/group/room)
        try:
            line_api.push_message(target_id, TextSendMessage(text=reply_text))
            print(f"Pushed reply to {target_id}")
        except Exception as e:
            print("Push failed:", e)
            # log and continue
    except Exception as e:
        print("Background processing error:", e)


# ====== Flask webhook endpoint ======
@app.route("/callback", methods=["GET", "POST"])
def callback():
    # allow GET for verification
    if request.method == "GET":
        return "OK", 200

    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    # Basic LINE signature handling
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception:
        # continue; we'll parse body manually
        pass

    try:
        data = json.loads(body)
    except Exception:
        return "bad request", 400

    events = data.get("events", [])
    for event in events:
        # dedupe key: use event['replyToken'] if present else timestamp
        event_id = event.get("replyToken") or event.get("timestamp") or None
        if event_id and event_id in processed_event_ids:
            continue
        if event_id:
            processed_event_ids.add(event_id)

        # only handle message/text events
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("type") != "text":
            # reply fallback quickly
            try:
                line_api.reply_message(event["replyToken"], TextSendMessage(text="テキストで話しかけてください。"))
            except Exception:
                pass
            continue

        # Determine source and target_id (supports user/group/room)
        source = event.get("source", {}) or {}
        user_id = source.get("userId")
        group_id = source.get("groupId")
        room_id = source.get("roomId")

        # choose target_id for push/message storage
        # 改修 group_idのチェックを先行した
        if group_id:
            target_id = group_id
        elif user_id:
            target_id = user_id
        elif room_id:
            target_id = room_id
        else:
            # unknown source type; skip
            print("Unknown source in event:", source)
            continue

        user_text = msg.get("text", "").strip()
        if not user_text:
            continue

        # Emergency handling (check in original user_text)
        lowered = user_text.replace(" ", "")
        if any(k in lowered for k in EMERGENCY_KEYWORDS):
            try:
                # Use reply_token for immediate reply
                line_api.reply_message(event["replyToken"], TextSendMessage(text=ESCALATE_MESSAGE))
            except Exception as e:
                print("Failed to send escalate message:", e)
            continue

        # immediate reply to satisfy replyToken constraints (works for group as well)
        # 考え中です。少しお待ちください。の表示を省略した
        interim = "考え中です。少しお待ちください。"
        try:
            # line_api.reply_message(event["replyToken"], TextSendMessage(text=interim))
            pass
        except Exception as e:
            # log but continue to background processing
            print("Immediate reply failed:", e)

        # spawn background thread passing target_id (so group responses are pushed to the group)
        Thread(target=process_and_push, args=(target_id, user_text, event.get("replyToken")), daemon=True).start()

    return "OK", 200


# ====== health check ======
@app.route("/", methods=["GET"])
def index():
    return "LINE Gemini Bot running — async workflow with memory & web enrichment.", 200


if __name__ == "__main__":
    # debug False in production
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
