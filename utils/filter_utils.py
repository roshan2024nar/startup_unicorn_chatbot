"""
utils/filter_utils.py

Query sanitization, routing rules, clarification mapping,
and filter extraction utilities.
"""

import re
from utils.text_utils import SECTOR_KW, CITY_KW


# ---------------------------------------------------------------------------
# Injection & Off-topic patterns
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: list[str] = [
    r"ignore.{0,20}instructions",
    r"disregard.{0,20}instructions",
    r"you are now",
    r"act as .*(evil|unrestricted|dan)",
    r"forget (everything|your|all)",
    r"system prompt",
    r"jailbreak",
    r"new persona",
    r"<\|.*?\|>",
]

IRRELEVANT_PATTERNS: list[str] = [
    r"\b(weather|recipe|movie|sport|cricket|football|song|music|joke|poem)\b",
    r"\b(war|military|politics|election)\b",
    r"\b(password|hack|exploit|bypass)\b",
]


# ---------------------------------------------------------------------------
# Routing keywords
# ---------------------------------------------------------------------------

STRUCTURED_KW: list[str] = [
    # Counts / aggregations
    "how many", "count", "total", "average", "list all",
    # Rankings
    "top ", "rank", "most funded", "highest valuation", "most valued",
    "most startups", "compare",
    # Funding specifics
    "how much", "how much did", "raised", "total funding",
    "series a", "series b", "series c", "series d",
    "seed round", "what rounds", "rounds did",
    # Investors
    "who invested", "investors", "funded by", "backed by",
    "led by", "round led",
    # Unicorn specifics
    "unicorn", "billion dollar", "valuation",
    "when did", "which city", "which sector", "which year",
    "founded in", "joined unicorn",
]

VAGUE_KW: list[str] = [
    "best", "good fit", "recommend", "suggest", "collaborate",
    "similar to", "interesting", "promising", "should i", "worth",
]

FOLLOWUP_WORDS: list[str] = [
    # Pronouns
    "these", "they", "them", "those",
    # Possessives
    "its", "their",
    # Reference phrases
    "the ones", "which of", "of them", "of these",
    # Demonstratives
    "this company", "that company", "this startup", "that startup",
    "the company", "the startup",
]


# ---------------------------------------------------------------------------
# Clarification mapping
# ---------------------------------------------------------------------------

CLARIFICATION_MAP: list[tuple[list[str], str]] = [
    (
        ["collaborate", "partnership"],
        "What kind of collaboration are you looking for — technology, payments, logistics, marketing, or something else?"
    ),
    (
        ["invest", "investing"],
        "Are you looking for companies to invest in, or asking about investors in a specific company?"
    ),
    (
        ["best", "good fit", "recommend", "suggest"],
        "Could you narrow it down? For example, by sector, city, funding stage, or valuation range."
    ),
    (
        ["similar", "like"],
        "Which company should I compare against? And should the match be based on sector, city, or funding stage?"
    ),
    (
        ["promising", "interesting", "worth"],
        "What criteria matter most to you — sector, city, investor quality, or valuation?"
    ),
]

DEFAULT_CLARIFICATION = (
    "Could you provide a bit more detail? Mention a sector (e.g., fintech, edtech, saas) "
    "or a city (e.g., Bangalore, Mumbai, Delhi) to refine the results."
)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def is_injection(text: str) -> tuple[bool, str]:
    """Return (True, pattern) if prompt injection detected."""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True, pattern
    return False, ""


def is_off_topic(text: str) -> bool:
    """Return True if query is clearly unrelated to startups."""
    for pattern in IRRELEVANT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def sanitize_text(raw: str, max_len: int = 500) -> str:
    """Normalize whitespace, remove unsafe characters, and truncate."""
    clean = re.sub(r"\s+", " ", str(raw).strip())
    clean = re.sub(r"[^\w\s\?\.,!\-&/()'']", "", clean)
    return clean[:max_len]


def extract_filters(query: str) -> dict:
    """
    Extract sector, city, and stage filters from query.
    Returns dict with keys matching ChromaDB metadata: sector, city, stage.
    """
    q = query.lower()
    filters = {}

    for kw, sector in SECTOR_KW.items():
        if kw in q:
            filters["sector"] = sector
            break

    for kw, city in CITY_KW.items():
        if kw in q:
            filters["city"] = city
            break

    stage_map = {
        "seed"    : "seed",
        "series a": "series a",
        "series b": "series b",
        "series c": "series c",
        "series d": "series d",
    }
    for kw, stage in stage_map.items():
        if kw in q:
            filters["stage"] = stage
            break

    return filters


def get_route(query: str, filters: dict) -> str:
    """
    Determine route for a cleaned query.
    Order: structured → vague → exploratory → factual
    """
    q = query.lower()

    if any(kw in q for kw in STRUCTURED_KW):
        return "structured"

    if any(kw in q for kw in VAGUE_KW):
        return "vague"

    if filters:
        return "exploratory"

    return "factual"


def get_clarification_question(query: str) -> str:
    """Return best clarification question for a vague query."""
    q = query.lower()
    for keywords, question in CLARIFICATION_MAP:
        if any(kw in q for kw in keywords):
            return question
    return DEFAULT_CLARIFICATION


def merge_filters(prior: dict, current: dict, is_followup: bool) -> dict:
    """
    Merge filters across turns.
    - Follow-up → merge prior + current (current wins on conflict)
    - New query with filters → use current only
    - New query without filters → reset
    """
    if is_followup:
        return {**prior, **current}
    if current:
        return current
    return {}