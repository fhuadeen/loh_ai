import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = str(os.getenv("OPENAI_API_KEY"))
