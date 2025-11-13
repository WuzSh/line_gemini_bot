import os
import json
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage
from linebot.exceptions import InvalidSignatureError
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

line_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
client = genai.Client(api_key=GEMINI_API_KEY)


def build_prompt(user_text):
    return f"""
あなたは共感的で優しい心理カウンセラーです。
相談者の話を聞き、気づきを促す質問を1つ返してください。
医療・宗教・法的判断は行わず、安全で前向きな表現にしてください。

ユーザー: {user_text}
AI:
"""


# 👇 Webhook検証(GET)にも対応
@app.route("/callback", methods=["GET", "POST"])
def callback():
    # ✅ LINEのWebhook検証時（GET）の対応
    if request.method == "GET":
        return "OK", 200

    # ✅ 実際のLINEからのPOSTリクエスト処理
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    data = json.loads(body)
    for event in data.get("events", []):
        if event["type"] == "message" and event["message"]["type"] == "text":
            text = event["message"]["text"]
            prompt = build_prompt(text)

            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                reply_text = response.text.strip()
            except Exception as e:
                reply_text = "少し考え中です。もう一度試してみてください。"

            line_api.reply_message(
                event["replyToken"],
                TextSendMessage(text=reply_text)
            )

    return "OK", 200


@app.route("/", methods=["GET"])
def index():
    return "LINE Gemini Bot running.", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

