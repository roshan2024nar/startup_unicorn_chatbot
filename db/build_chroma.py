"""
db/build_chroma.py
Run: python -m db.build_chroma
"""
import os
import sys
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEAN_CSV, CHROMA_DB_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL


def safe_str(val, default="unknown"):
    if pd.isna(val) or str(val).strip() in ("", "nan", "NaN"):
        return default
    return str(val).strip().lower()


def safe_float(val):
    try:
        v = float(val)
        return v if not np.isnan(v) else 0.0
    except Exception:
        return 0.0


def safe_int(val):
    try:
        v = float(val)
        return int(v) if not np.isnan(v) else 0
    except Exception:
        return 0


def build():
    print(f"Loading {CLEAN_CSV}...")
    df = pd.read_csv(CLEAN_CSV)
    print(f"  {len(df)} companies loaded")

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("  Dropped existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = [], [], []

    for _, row in df.iterrows():
        company_id = str(row["company_name"]).strip().lower()

        doc = str(row.get("embedding_text", "")).strip()
        if not doc:
            doc = company_id

        metadata_filters = {
            "company_name"        : safe_str(row.get("company_name"), ""),
            "sector"              : safe_str(row.get("sector"), "unknown"),
            "city"                : safe_str(row.get("city"), "unknown"),
            "stage"               : safe_str(row.get("stage"), "unknown"),
            "unicorn_joined_year" : safe_int(row.get("unicorn_joined_year")),
            "valuation_usd_bn"    : safe_float(row.get("valuation_usd_bn")),
            "top_investors"       : safe_str(row.get("top_investors"), "undisclosed"),
            "founded_year"        : safe_int(row.get("founded_year")),
        }

        ids.append(company_id)
        documents.append(doc)
        metadatas.append(metadata_filters)

    BATCH = 50 
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids       = ids[i : i + BATCH],
            documents = documents[i : i + BATCH],
            metadatas = metadatas[i : i + BATCH],
        )
        print(f"  Inserted {min(i + BATCH, len(ids))}/{len(ids)}")

    print(f"ChromaDB built: {collection.count()} documents at '{CHROMA_DB_PATH}'")

if __name__ == "__main__":
    build()