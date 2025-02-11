import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

TOKEN = os.getenv('TOKEN')
CHAT_ID = os.getenv('CHAT_ID')


def sendPlate(message):
    """
    Sends a message to the Telegram bot.
    """
    if not message.strip():
        print("⚠️ Warning: Attempted to send an empty message.")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data.get("ok"):
            print(f"✅ Message sent successfully: {message}")
        else:
            print(f"❌ Telegram API error: {data}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send message: {e}")
