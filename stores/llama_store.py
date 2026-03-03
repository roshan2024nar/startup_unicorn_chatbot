"""
stores/llama_store.py

Structured query layer using LlamaIndex PandasQueryEngine over unicorns_clean.csv.
"""

import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.groq import Groq as LlamaGroq

from config import GROQ_API_KEY, GROQ_MODEL, validate_files
from utils.data_loader import load_csv

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

validate_files()

df  = load_csv()
llm = LlamaGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

engine = PandasQueryEngine(
    df=df,
    llm=llm,
    verbose=False,
    synthesize_response=True,
)


def query_llama(question: str) -> str:
    """
    Execute a structured query against the unicorn CSV using LlamaIndex.
    Returns natural language output or '__error__:<message>' on failure.
    """
    try:
        response = engine.query(question)
        return str(response).strip()
    except Exception as e:
        return f"__error__:{e}"


def is_error_response(ans: str) -> bool:
    """Detect whether a response should be treated as a failure."""
    if not ans or len(ans) < 5:
        return True
    if ans.startswith("__error__"):
        return True
    if any(w in ans.lower() for w in ["i don't know", "cannot", "no information"]):
        return True
    return False