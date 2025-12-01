import os
import requests
from dotenv import load_dotenv

load_dotenv()  # if you use .env

api_key = os.getenv("OPENAI_API_KEY")

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "How are you today?"},
    ],
}

response = requests.post(url, headers=headers, json=payload)
print(response.status_code)
print(response.text)
