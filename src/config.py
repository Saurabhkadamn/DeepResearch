"""
Deep Research - Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LLM CONFIGURATION (OpenRouter)
# ============================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Dual model strategy:
# FAST: cheap, high volume — analysis, planning, synthesis, sub-agents
# QUALITY: better, 1 call — final report writing
LLM_MODEL_FAST = os.getenv("LLM_MODEL_FAST", "deepseek/deepseek-chat-v3-0324")
LLM_MODEL_QUALITY = os.getenv("LLM_MODEL_QUALITY", "deepseek/deepseek-chat-v3-0324")

# ============================================================
# TAVILY SEARCH
# ============================================================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ============================================================
# RESEARCH SETTINGS
# ============================================================
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
MAX_RESEARCH_LOOPS = int(os.getenv("MAX_RESEARCH_LOOPS", "3"))
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "10"))

# ============================================================
# STORAGE
# ============================================================
DATA_DIR = os.getenv("DATA_DIR", "data")
CHAT_HISTORY_FILE = os.path.join(DATA_DIR, "chat_history.json")

# ============================================================
# SERVER
# ============================================================
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
