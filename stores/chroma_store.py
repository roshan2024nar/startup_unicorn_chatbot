"""
stores/chroma_store.py
"""

import logging
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL, validate_files

validate_files()

model      = SentenceTransformer(EMBEDDING_MODEL)
client     = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = client.get_collection(name=CHROMA_COLLECTION)

logger = logging.getLogger("chatbot")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_collection_count() -> int:
    return collection.count()


def encode(query: str) -> list:
    return model.encode([query], show_progress_bar=False).tolist()

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def semantic_search(query: str, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
    """Pure semantic search across all unicorn companies."""
    q_embedding = encode(query)

    results = collection.query(
        query_embeddings=q_embedding,
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    output = []
    for i, id_ in enumerate(results["ids"][0]):
        sim = round(1 - results["distances"][0][i] / 2, 4)
        if sim >= threshold:
            meta = results["metadatas"][0][i]
            output.append({
                "company_name"  : meta.get("company_name", id_),
                "score"         : sim,
                "metadata"      : meta,
                "embedding_text": results["documents"][0][i],
            })

    return output


def filtered_search(
    query: str,
    filters: dict,
    top_k: int = 5,
    threshold: float = 0.3,
) -> list[dict]:
    """
    Semantic search with metadata filters.
    Supported keys: sector, city, stage, unicorn_joined_year.
    Falls back to semantic_search if filtering fails.
    """
    if not filters:
        return semantic_search(query, top_k, threshold)

    conditions = []
    for key, value in filters.items():
        if not value:
            continue
        conditions.append({key: {"$eq": value}})

    if not conditions:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    q_embedding = encode(query)

    try:
        results = collection.query(
            query_embeddings=q_embedding,
            n_results=top_k,
            where=where,
            include=["metadatas", "distances", "documents"],
        )
    except Exception:
        logger.warning("Filtered search failed, falling back to semantic_search.")
        return semantic_search(query, top_k, threshold)

    output = []
    for i, id_ in enumerate(results["ids"][0]):
        sim = round(1 - results["distances"][0][i] / 2, 4)
        if sim >= threshold:
            meta = results["metadatas"][0][i]
            output.append({
                "company_name"  : meta.get("company_name", id_),
                "score"         : sim,
                "metadata"      : meta,
                "embedding_text": results["documents"][0][i],
            })

    return output

# ---------------------------------------------------------------------------
# Exact Retrieval
# ---------------------------------------------------------------------------

def get_companies_by_name(company_names: list[str]) -> list[dict]:
    """Retrieve unicorn companies by exact name match."""
    results = []

    for name in company_names:
        try:
            res = collection.get(
                where={"company_name": {"$eq": name.lower()}},
                limit=1,
                include=["metadatas", "documents"],
            )
            if res["documents"] and res["metadatas"]:
                results.append({
                    "company_name"  : name,
                    "score"         : 1.0,
                    "metadata"      : res["metadatas"][0],
                    "embedding_text": res["documents"][0],
                })
        except Exception as e:
            logger.error(f"Error fetching company '{name}': {e}")

    return results


def get_all_company_names() -> list[str]:
    """Return all unique company names in the collection."""
    try:
        results = collection.get(include=["metadatas"])
        return list({
            meta["company_name"]
            for meta in results["metadatas"]
            if meta and "company_name" in meta
        })
    except Exception as e:
        logger.error(f"Error fetching company names: {e}")
        return []