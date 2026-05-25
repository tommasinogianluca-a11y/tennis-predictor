import os
import sys
from unittest.mock import MagicMock

# Set required env vars before any imports
os.environ.setdefault("API_SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:///test_tennis.db")
os.environ.setdefault("DASHBOARD_PASSWORD", "test-dashboard-pass")

# Mock xgboost before it's imported (libomp missing on this dev machine)
sys.modules.setdefault("xgboost", MagicMock())
