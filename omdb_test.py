from dotenv import load_dotenv
import os
import requests

load_dotenv()  # loads .env from the project root

api_url = os.getenv("OMDB_API_URL")
api_key = os.getenv("OMDB_API_KEY")

def get_movie(title: str):
    response = requests.get(
        api_url,
        params={"t": title, "apikey": api_key}
    )
    return response.json()

print(get_movie("Spider-Man: No Way Home"))