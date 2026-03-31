from fastapi import APIRouter, Query
from typing import Optional

from backend import database
from backend.news_events import parse_event_types

router = APIRouter()


RETURN_FIELDS = ("ret_t0", "ret_t1", "ret_t3", "ret_t5", "ret_t10")


def _percent_or_none(value):
    if value is None:
        return None
    return round(float(value) * 100, 2)


def _normalize_return_fields(row: dict) -> dict:
    normalized = dict(row)
    for field in RETURN_FIELDS:
        if field in normalized:
            normalized[field] = _percent_or_none(normalized[field])
    normalized["event_types"] = parse_event_types(
        normalized.get("event_type_tags_json"),
        normalized.get("title"),
        normalized.get("description"),
        normalized.get("key_discussion"),
        normalized.get("reason_growth"),
        normalized.get("reason_decrease"),
    )
    normalized["event_type"] = normalized.get("event_type") or normalized["event_types"][0]
    return normalized


def _ensure_news_schema_ready() -> None:
    ensure = getattr(database, "ensure_news_aligned_attribution_columns", None)
    if callable(ensure):
        ensure()
    ensure_layer1 = getattr(database, "ensure_layer1_event_columns", None)
    if callable(ensure_layer1):
        ensure_layer1()


@router.get("/{symbol}")
def get_news_for_date(
    symbol: str,
    date: Optional[str] = None,
):
    """Get news for a symbol, optionally filtered to a specific trading day."""
    _ensure_news_schema_ready()
    conn = database.get_conn()
    symbol = symbol.upper()

    try:
        with conn.cursor() as cur:
            if date:
                cur.execute(
                    """SELECT na.news_id, na.trade_date, na.published_utc,
                              na.session_bucket, na.label_anchor,
                              na.ret_t0, na.ret_t1, na.ret_t3, na.ret_t5, na.ret_t10,
                              nr.title, nr.description, nr.publisher, nr.article_url,
                              l1.relevance, l1.key_discussion, l1.chinese_summary,
                              l1.sentiment, l1.event_type, l1.event_type_tags_json,
                              l1.reason_growth, l1.reason_decrease
                       FROM news_aligned na
                       JOIN news_raw nr ON na.news_id = nr.id
                       LEFT JOIN layer1_results l1 ON na.news_id = l1.news_id AND l1.symbol = %s
                       WHERE na.symbol = %s AND na.trade_date = %s
                       ORDER BY na.published_utc DESC""",
                    (symbol, symbol, date),
                )
            else:
                cur.execute(
                    """SELECT na.news_id, na.trade_date, na.published_utc,
                              na.session_bucket, na.label_anchor,
                              na.ret_t0, na.ret_t1, na.ret_t3, na.ret_t5, na.ret_t10,
                              nr.title, nr.description, nr.publisher, nr.article_url,
                              l1.relevance, l1.key_discussion, l1.chinese_summary,
                              l1.sentiment, l1.event_type, l1.event_type_tags_json,
                              l1.reason_growth, l1.reason_decrease
                       FROM news_aligned na
                       JOIN news_raw nr ON na.news_id = nr.id
                       LEFT JOIN layer1_results l1 ON na.news_id = l1.news_id AND l1.symbol = %s
                       WHERE na.symbol = %s
                       ORDER BY na.published_utc DESC
                       LIMIT 100""",
                    (symbol, symbol),
                )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_normalize_return_fields(row) for row in rows]


@router.get("/{symbol}/range")
def get_news_for_range(
    symbol: str,
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
):
    """Get news within a date range, with top bullish/bearish articles."""
    _ensure_news_schema_ready()
    conn = database.get_conn()
    symbol = symbol.upper()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT na.news_id, na.trade_date, na.published_utc,
                          na.session_bucket, na.label_anchor,
                          na.ret_t0, na.ret_t1, na.ret_t3, na.ret_t5, na.ret_t10,
                          nr.title, nr.description, nr.publisher, nr.article_url,
                          l1.relevance, l1.key_discussion, l1.chinese_summary,
                          l1.sentiment, l1.event_type, l1.event_type_tags_json,
                          l1.reason_growth, l1.reason_decrease
                   FROM news_aligned na
                   JOIN news_raw nr ON na.news_id = nr.id
                   LEFT JOIN layer1_results l1 ON na.news_id = l1.news_id AND l1.symbol = %s
                   WHERE na.symbol = %s AND na.trade_date BETWEEN %s AND %s
                   ORDER BY na.published_utc DESC""",
                (symbol, symbol, start, end),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    articles = [_normalize_return_fields(row) for row in rows]

    top_bullish = sorted(
        [a for a in articles if a.get("sentiment") == "positive" and a.get("ret_t0") is not None],
        key=lambda a: a["ret_t0"],
        reverse=True,
    )[:5]

    top_bearish = sorted(
        [a for a in articles if a.get("sentiment") == "negative" and a.get("ret_t0") is not None],
        key=lambda a: a["ret_t0"],
    )[:5]

    return {
        "total": len(articles),
        "date_range": [start, end],
        "articles": articles,
        "top_bullish": top_bullish,
        "top_bearish": top_bearish,
    }


@router.get("/{symbol}/particles")
def get_news_particles(symbol: str):
    """Return lightweight per-article data for chart particle visualization."""
    _ensure_news_schema_ready()
    conn = database.get_conn()
    symbol = symbol.upper()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT na.news_id, na.trade_date, na.ret_t1,
                          na.session_bucket, na.label_anchor,
                          nr.title,
                          l1.sentiment, l1.relevance, l1.event_type, l1.event_type_tags_json
                   FROM news_aligned na
                   JOIN news_raw nr ON na.news_id = nr.id
                   LEFT JOIN layer1_results l1 ON na.news_id = l1.news_id AND l1.symbol = %s
                   WHERE na.symbol = %s
                   ORDER BY na.trade_date ASC, l1.relevance DESC""",
                (symbol, symbol),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "id": r["news_id"],
            "d": r["trade_date"],
            "s": r["sentiment"],
            "r": r["relevance"],
            "t": (r["title"] or "")[:80],
            "rt1": _percent_or_none(r["ret_t1"]),
            "session_bucket": r.get("session_bucket"),
            "label_anchor": r.get("label_anchor"),
            "event_type": (r.get("event_type") or parse_event_types(r.get("event_type_tags_json"), r.get("title"))[0]),
            "event_types": parse_event_types(r.get("event_type_tags_json"), r.get("title")),
        }
        for r in rows
    ]


@router.get("/{symbol}/categories")
def get_news_categories(symbol: str):
    """Categorize ALL news for a symbol by topic using keyword matching."""
    _ensure_news_schema_ready()
    conn = database.get_conn()
    symbol = symbol.upper()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT na.news_id,
                          nr.title,
                          nr.description,
                          l1.key_discussion,
                          l1.reason_growth,
                          l1.reason_decrease,
                          l1.sentiment,
                          l1.event_type_tags_json
                   FROM news_aligned na
                   JOIN news_raw nr ON na.news_id = nr.id
                   LEFT JOIN layer1_results l1 ON na.news_id = l1.news_id AND l1.symbol = %s
                   WHERE na.symbol = %s
                   ORDER BY na.trade_date DESC""",
                (symbol, symbol),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    EVENT_LABELS = {
        "earnings": "earnings",
        "policy": "policy",
        "order_contract": "order_contract",
        "product_tech": "product_tech",
        "buyback_increase": "buyback_increase",
        "reduction_unlock": "reduction_unlock",
        "mna_restructuring": "mna_restructuring",
        "litigation_penalty": "litigation_penalty",
        "management": "management",
        "other": "other",
    }

    categories = {}
    for cat in EVENT_LABELS.values():
        categories[cat] = {
            "label": cat,
            "count": 0,
            "article_ids": [],
            "positive_ids": [],
            "negative_ids": [],
            "neutral_ids": [],
        }

    total = len(rows)
    for r in rows:
        event_types = parse_event_types(
            r.get("event_type_tags_json"),
            r.get("title"),
            r.get("description"),
            r.get("key_discussion"),
            r.get("reason_growth"),
            r.get("reason_decrease"),
        )
        sentiment = r["sentiment"]
        for cat in event_types:
            if cat not in categories:
                categories[cat] = {
                    "label": cat,
                    "count": 0,
                    "article_ids": [],
                    "positive_ids": [],
                    "negative_ids": [],
                    "neutral_ids": [],
                }
            categories[cat]["count"] += 1
            categories[cat]["article_ids"].append(r["news_id"])
            if sentiment == "positive":
                categories[cat]["positive_ids"].append(r["news_id"])
            elif sentiment == "negative":
                categories[cat]["negative_ids"].append(r["news_id"])
            else:
                categories[cat]["neutral_ids"].append(r["news_id"])

    return {"categories": categories, "total": total}


@router.get("/{symbol}/timeline")
def get_news_timeline(symbol: str):
    """Get dates that have news for a symbol (used for chart markers)."""
    _ensure_news_schema_ready()
    conn = database.get_conn()
    symbol = symbol.upper()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT trade_date, COUNT(*) as news_count,
                          SUM(CASE WHEN l1.relevance = 'relevant' THEN 1 ELSE 0 END) as relevant_count
                   FROM news_aligned na
                   LEFT JOIN layer1_results l1 ON na.news_id = l1.news_id AND l1.symbol = na.symbol
                   WHERE na.symbol = %s
                   GROUP BY trade_date
                   ORDER BY trade_date ASC""",
                (symbol,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return list(rows)
