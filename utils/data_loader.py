"""
utils/data_loader.py

Centralized data loading for the unicorn dataset.
"""

import pandas as pd
from pathlib import Path

from config import CLEAN_CSV


# Required columns
REQUIRED_CSV_COLUMNS = {
    "company_name", "sector", "city", "country",
    "founded_year", "unicorn_joined_year", "valuation_usd_bn",
    "total_funding_usd_mn", "top_investors", "embedding_text",
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_csv(path: Path = None, validate: bool = True) -> pd.DataFrame:
    """Load unicorn CSV with optional schema validation."""
    p = Path(path) if path else CLEAN_CSV

    if not p.exists():
        raise FileNotFoundError(
            f"CSV not found: {p}\n"
            "Ensure unicorns_clean.csv exists in data/"
        )

    df = pd.read_csv(p)

    if validate:
        missing = REQUIRED_CSV_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}\n"
                f"Found: {df.columns.tolist()}"
            )

    return df

