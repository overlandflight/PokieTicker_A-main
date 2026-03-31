"""TF-IDF similarity engine for finding historically similar news articles.

Builds a TF-IDF matrix from all analyzed articles (title + key_discussion),
uses cosine similarity to find similar articles across all tickers.
Lazy-loaded with pickle caching.
"""

import os
import time
import pickle
from typing import Dict, Any, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import PROJECT_ROOT
from backend.database import get_conn

PICKLE_PATH = str(PROJECT_ROOT / "data" / "tfidf_matrix.pkl")
CACHE_TTL = 24 * 3600  # 24 hours

# Module-level cache
_vectorizer: Optional[TfidfVectorizer] = None
_matrix = None
_news_ids: Optional[List[str]] = None
_symbols: Optional[List[str]] = None


RETURN_FIELDS = ("ret_t0", "ret_t1", "ret_t3", "ret_t5", "ret_t10")


def _ratio_to_percent(value):
    if value is None:
        return None
    return round(float(value) * 100, 2)


def _needs_rebuild() -> bool:
    if _matrix is None:
        return True
    if not os.path.exists(PICKLE_PATH):
        return True
    age = time.time() - os.path.getmtime(PICKLE_PATH)
    return age > CACHE_TTL


def _build_index():
    """Build TF-IDF matrix from all analyzed articles."""
    global _vectorizer, _matrix, _news_ids, _symbols

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT l1.news_id, l1.symbol, nr.title, l1.key_discussion
                   FROM layer1_results l1
                   JOIN news_raw nr ON l1.news_id = nr.id
                   WHERE l1.relevance = 'relevant'
                   ORDER BY l1.news_id"""
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        _vectorizer = TfidfVectorizer()
        _matrix = None
        _news_ids = []
        _symbols = []
        return

    corpus = []
    _news_ids = []
    _symbols = []

    for row in rows:
        text = (row["title"] or "") + " " + (row["key_discussion"] or "")
        corpus.append(text.strip())
        _news_ids.append(row["news_id"])
        _symbols.append(row["symbol"])

    _vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    _matrix = _vectorizer.fit_transform(corpus)

    # Save to pickle
    os.makedirs(os.path.dirname(PICKLE_PATH), exist_ok=True)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(
            {
                "vectorizer": _vectorizer,
                "matrix": _matrix,
                "news_ids": _news_ids,
                "symbols": _symbols,
            },
            f,
        )


def _load_index():
    """Load index from pickle or rebuild."""
    global _vectorizer, _matrix, _news_ids, _symbols

    if not _needs_rebuild():
        return

    if os.path.exists(PICKLE_PATH):
        age = time.time() - os.path.getmtime(PICKLE_PATH)
        if age <= CACHE_TTL:
            with open(PICKLE_PATH, "rb") as f:
                data = pickle.load(f)
                _vectorizer = data["vectorizer"]
                _matrix = data["matrix"]
                _news_ids = data["news_ids"]
                _symbols = data["symbols"]
                return

    _build_index()


def find_similar(news_id: str, symbol: str, top_k: int = 20) -> Dict[str, Any]:
    """Find articles similar to the given news_id."""
    _load_index()

    if _matrix is None or not _news_ids:
        return {"query": None, "stats": {}, "similar_articles": []}

    # Find query article in index
    query_idx = None
    for i, (nid, sym) in enumerate(zip(_news_ids, _symbols)):
        if nid == news_id and sym == symbol:
            query_idx = i
            break

    if query_idx is None:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT nr.title, l1.key_discussion
                       FROM layer1_results l1
                       JOIN news_raw nr ON l1.news_id = nr.id
                       WHERE l1.news_id = %s AND l1.symbol = %s""",
                    (news_id, symbol),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            return {"query": None, "stats": {}, "similar_articles": []}

        text = (row["title"] or "") + " " + (row["key_discussion"] or "")
        query_vec = _vectorizer.transform([text.strip()])
    else:
        query_vec = _matrix[query_idx]

    # Compute similarities
    sims = cosine_similarity(query_vec, _matrix).flatten()

    # Get top K (excluding self)
    top_indices = np.argsort(sims)[::-1]
    results = []
    for idx in top_indices:
        idx = int(idx)
        if _news_ids[idx] == news_id and _symbols[idx] == symbol:
            continue
        if sims[idx] < 0.05:
            break
        results.append((idx, float(sims[idx])))
        if len(results) >= top_k:
            break

    if not results:
        return {"query": {"news_id": news_id, "symbol": symbol}, "stats": {}, "similar_articles": []}

    # Fetch article details and returns
    result_ids = [(int(idx), _news_ids[idx], _symbols[idx]) for idx, _ in results]
    sim_scores = {(_news_ids[idx], _symbols[idx]): score for idx, score in results}

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Get query article info
            cur.execute(
                """SELECT nr.title, na.trade_date, na.ret_t1, na.ret_t5
                   FROM news_raw nr
                   LEFT JOIN news_aligned na ON nr.id = na.news_id AND na.symbol = %s
                   WHERE nr.id = %s""",
                (symbol, news_id),
            )
            query_row = cur.fetchone()

            # Get similar articles' details - build IN clause
            pairs = [(nid, sym) for _, nid, sym in result_ids]
            if pairs:
                conditions = " OR ".join(
                    f"(l1.news_id = %s AND l1.symbol = %s)" for _ in pairs
                )
                params = []
                for nid, sym in pairs:
                    params.extend([nid, sym])

                cur.execute(
                    f"""SELECT nr.id as news_id, l1.symbol, nr.title,
                               na.trade_date, na.ret_t0, na.ret_t1, na.ret_t3, na.ret_t5, na.ret_t10,
                               l1.sentiment
                        FROM news_raw nr
                        JOIN layer1_results l1 ON nr.id = l1.news_id
                        LEFT JOIN news_aligned na ON nr.id = na.news_id AND na.symbol = l1.symbol
                        WHERE {conditions}""",
                    params,
                )
                similar_rows = cur.fetchall()
            else:
                similar_rows = []
    finally:
        conn.close()

    # Build similar articles list
    similar_articles = []
    ret_t1_vals = []
    ret_t5_vals = []
    ticker_set = set()

    for row in similar_rows:
        key = (row["news_id"], row["symbol"])
        score = sim_scores.get(key, 0)
        article = {
            "news_id": row["news_id"],
            "symbol": row["symbol"],
            "title": row["title"],
            "trade_date": row["trade_date"],
            "similarity": round(score, 3),
            "sentiment": row["sentiment"],
            "ret_t0": _ratio_to_percent(row["ret_t0"]),
            "ret_t1": _ratio_to_percent(row["ret_t1"]),
            "ret_t3": _ratio_to_percent(row["ret_t3"]),
            "ret_t5": _ratio_to_percent(row["ret_t5"]),
            "ret_t10": _ratio_to_percent(row["ret_t10"]),
        }
        similar_articles.append(article)
        ticker_set.add(row["symbol"])
        if row["ret_t1"] is not None:
            ret_t1_vals.append(row["ret_t1"])
        if row["ret_t5"] is not None:
            ret_t5_vals.append(row["ret_t5"])

    similar_articles.sort(key=lambda a: a["similarity"], reverse=True)

    stats = {
        "total": len(similar_articles),
        "cross_ticker_count": len(ticker_set),
        "positive_t1_pct": round(sum(1 for v in ret_t1_vals if v > 0) / len(ret_t1_vals) * 100, 1) if ret_t1_vals else None,
        "positive_t5_pct": round(sum(1 for v in ret_t5_vals if v > 0) / len(ret_t5_vals) * 100, 1) if ret_t5_vals else None,
        "avg_ret_t1": _ratio_to_percent(sum(ret_t1_vals) / len(ret_t1_vals)) if ret_t1_vals else None,
        "avg_ret_t5": _ratio_to_percent(sum(ret_t5_vals) / len(ret_t5_vals)) if ret_t5_vals else None,
        "median_ret_t1": _ratio_to_percent(float(np.median(ret_t1_vals))) if ret_t1_vals else None,
        "median_ret_t5": _ratio_to_percent(float(np.median(ret_t5_vals))) if ret_t5_vals else None,
    }

    query_info = {
        "news_id": news_id,
        "symbol": symbol,
        "title": query_row["title"] if query_row else None,
        "trade_date": query_row["trade_date"] if query_row else None,
    }

    return {
        "query": query_info,
        "stats": stats,
        "similar_articles": similar_articles,
    }
