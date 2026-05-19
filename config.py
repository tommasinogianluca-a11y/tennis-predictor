import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///tennis.db")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "changeme")
PORT = int(os.getenv("PORT", 8000))
