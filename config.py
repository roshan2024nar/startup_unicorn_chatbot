"""
config.py
─────────
Single source of truth for all constants, paths, and environment variables.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── Redis (conversation memory) ───────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Langfuse (observability) ──────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR          = Path("data")
CLEAN_CSV         = DATA_DIR / "unicorns_clean.csv"
CHROMA_DB_PATH    = DATA_DIR / "processed" / "funding_db"
CHROMA_COLLECTION = "unicorn_startups"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"

# ── ChromaDB thresholds ───────────────────────────────────────────────────────
QUALITY_THRESHOLD  = 0.30   # minimum score for a good result
FALLBACK_THRESHOLD = 0.20   # very weak match — last resort before fallback msg
CHROMA_TOP_K       = 5      # default number of results to retrieve

# ── Session ───────────────────────────────────────────────────────────────────
SESSION_ID_FILE = Path(".session_id")

# ── LLM context window ────────────────────────────────────────────────────────
HISTORY_WINDOW = 6          # how many prior messages to pass to the LLM

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = Path("chatbot.log")

# ── Input constraints ─────────────────────────────────────────────────────────
MAX_INPUT_LENGTH = 500

# ── Response templates ────────────────────────────────────────────────────────
FALLBACK_RESPONSE = (
    "I couldn't find reliable information on that in my dataset of "
    "Indian unicorn startups. Try asking about a specific company, "
    "sector, city, or funding stage."
)
OFF_TOPIC_MSG = (
    "I'm designed to answer questions about Indian unicorn startups. "
    "Feel free to ask about companies, funding, investors, sectors, or cities."
)
BLOCKED_MSG = (
    "I'm designed to answer questions about Indian unicorn startups. "
    "Feel free to ask about companies, funding, investors, sectors, or cities."
)
EMPTY_MSG = (
    "I didn't catch that. You can ask me about Indian unicorn startups — "
    "for example: 'What does Juspay do?' or 'Show me fintech startups in Bangalore'."
)
GREETING_MSG = (
    "Hi! I can help you explore Indian unicorn startups.\n"
)


# ── File validation ───────────────────────────────────────────────────────────
def validate_files() -> None:
    """
    Called at startup by chroma_store and llama_store.
    Raises FileNotFoundError early if required data files are missing.
    """
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"Required CSV not found: {CLEAN_CSV}\n"
            "Run data preprocessing first to generate unicorns_clean.csv."
        )
    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at: {CHROMA_DB_PATH}\n"
            "Run: python -m db.build_chroma"
        )