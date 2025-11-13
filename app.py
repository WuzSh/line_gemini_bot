import os
import json
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage
from linebot.exceptions import InvalidSignatureError
from google import genai
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
app = Flask(__name__)

# ==== 環境変数 ====
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
client = genai.Client(api_key=GEMINI_API_KEY)

# ==== メモリ（簡易実装：サーバー内一時保存）====
# ※ Renderの無料環境では再起動でリセットされるため、永続化する場合はFireStoreやRedisを推奨
user_memory = defaultdict(lambda: {"history": [], "phase": "empathy"})

# ==== プロンプト構築 ====
def build_prompt(user_text, history, phase):
    base_prompt = """
あなたは共感的で優しい心理カウンセラーです。
相手の言葉を受け止め、安心感と自立的な気づきを促します。
"""

    # --- フェーズ別トーン設定 ---
    if phase == "empathy":
        role = (
            "現在のフェーズ: 共感フェーズ。\n"
            "目的: 安心と受容を重視。相手を否定せず、共感的な返答をしてください。\n"
            "質問は浅く、感情を受け止める内容中心にしてください。"
        )
    elif phase == "awareness":
        role = (
            "現在のフェーズ: 気づきフェーズ。\n"
            "目的: ユーザーの感情・価値観の理解を深める。\n"
            "『なぜ』『どんな気持ち』などの質問を使い、内省を促してください。"
        )
    else:
        role = (
            "現在のフェーズ: 再構築フェーズ。\n"
            "目的: ユーザーが自分の考えや希望を再発見し、小さな前向きな行動を見つける。\n"
            "励ましと自立を促す対話を行ってください。"
        )

    # --- 会話履歴を文脈に追加 ---
    conversation = ""
    for msg in history[-5:]:  # 直近5ターンのみ
        role_label = "ユーザー" if msg["role"] == "user" else "AI"
        conversation += f"{role_label}: {msg['content']}\n"

    return f"{base_prompt}\n{role}\n\nこれまでの会話:\n{conversation}\nユーザー: {user_text}\nAI:"


# ==== フェーズ遷移ロジック ====
def next_phase(current_phase, user_text):
    """
    単純ロジック：
    - empathy → awareness：3ターン経過または「なぜ」「どうして」などを含む
    - awareness → reconstruction：「自分」「自由」「前向き」などを含む
    """
    text = user_text.lower()
    if current_phase == "empathy" and any(k in text for k in ["なぜ", "どうして", "意味", "理由"]):
        return "awareness"
    elif current_phase == "awareness" and any(k in text for k in ["自分", "自由", "選ぶ", "前向き"]):
        return "reconstruction"
    else:
        return current_phase


# ==== LINEコールバック ====
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    data = json.loads(body)
    for event in data.get("events", []):
        if event["type"] == "message" and event["message"]["type"] == "text":
            user_id = event["source"]["userId"]
            user_text = event["message"]["text"]

            # --- 履歴取得 ---
            memory = user_memory[user_id]
            history = memory["history"]
            phase = memory["phase"]

            # --- フェーズ遷移 ---
            new_phase = next_phase(phase, user_text)
            user_memory[user_id]["phase"] = new_phase

            # --- Gemini呼び出し ---
            prompt = build_prompt(user_text, history, new_phase)
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                reply_text = response.text.strip()
            except Exception:
                reply_text = "少し考え中です。もう一度話しかけてくださいね。"

            # --- 履歴更新 ---
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply_text})

            # --- LINE返信 ---
            line_api.reply_message(
                event["replyToken"],
                TextSendMessage(text=reply_text)
            )

    return "OK", 200


# ==== 動作確認用 ====
@app.route("/", methods=["GET"])
def index():
    return "LINE Gemini Bot running with memory & 3-phase counseling.", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
