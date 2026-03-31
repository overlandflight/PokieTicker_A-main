import logging
import re

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, timezone

from backend.database import ensure_ticker_alias_table, get_conn
from backend.tushare.client import search_tickers

logger = logging.getLogger(__name__)

router = APIRouter()


class AddTickerRequest(BaseModel):
    symbol: str
    name: Optional[str] = None


class TickerAliasRequest(BaseModel):
    alias: str
    alias_type: Optional[str] = None


def _normalize_alias(alias: str) -> str:
    return re.sub(r"\s+", " ", str(alias or "").strip())


def _invalidate_layer1_keyword_cache(symbol: str) -> None:
    try:
        from backend.pipeline.layer1 import invalidate_keyword_cache

        invalidate_keyword_cache(symbol)
    except Exception:
        logger.debug("Layer1 keyword cache invalidation skipped for %s", symbol, exc_info=True)


def _ensure_ticker_exists(cur, symbol: str) -> dict:
    cur.execute("SELECT symbol, name, sector FROM tickers WHERE symbol = %s", (symbol,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Ticker not found: {symbol}")
    return row


@router.get("")
def list_tickers():
    """List all tracked tickers."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM tickers ORDER BY symbol")
            rows = cur.fetchall()
    finally:
        conn.close()
    return list(rows)


@router.get("/search")
def search(q: str = Query(..., min_length=1)):
    """Fuzzy search tickers via Tushare."""
    ensure_ticker_alias_table()
    query_text = q.strip()
    like = f"%{query_text}%"
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT t.symbol, t.name, t.sector,
                          GROUP_CONCAT(
                              DISTINCT CASE WHEN ta.alias LIKE %s THEN ta.alias END
                              ORDER BY ta.alias SEPARATOR ' / '
                          ) AS alias_hits
                   FROM tickers t
                   LEFT JOIN ticker_aliases ta ON ta.symbol = t.symbol
                   WHERE t.symbol LIKE %s OR t.name LIKE %s OR ta.alias LIKE %s
                   GROUP BY t.symbol, t.name, t.sector
                   ORDER BY
                     CASE
                       WHEN t.symbol = %s THEN 0
                       WHEN t.name = %s THEN 1
                       WHEN MAX(CASE WHEN ta.alias = %s THEN 1 ELSE 0 END) = 1 THEN 2
                       ELSE 3
                     END,
                     t.symbol
                   LIMIT 10""",
                (like, like, like, like, query_text.upper(), query_text, query_text),
            )
            local = cur.fetchall()
    finally:
        conn.close()

    results = list(local)

    if len(results) < 5:
        try:
            remote = search_tickers(q, limit=10)
            seen = {r["symbol"] for r in results}
            for r in remote:
                if r["symbol"] not in seen:
                    results.append(r)
        except Exception:
            logger.debug("Tushare search failed for query=%s", q)

    return results


@router.get("/{symbol}/aliases")
def list_ticker_aliases(symbol: str):
    """List manually maintained aliases for a ticker."""
    ensure_ticker_alias_table()
    symbol = symbol.upper()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            ticker = _ensure_ticker_exists(cur, symbol)
            cur.execute(
                """SELECT symbol, alias, alias_type
                   FROM ticker_aliases
                   WHERE symbol = %s
                   ORDER BY alias_type IS NULL, alias_type, alias""",
                (symbol,),
            )
            aliases = cur.fetchall()
    finally:
        conn.close()

    return {
        "symbol": ticker["symbol"],
        "name": ticker.get("name"),
        "sector": ticker.get("sector"),
        "aliases": list(aliases),
        "count": len(aliases),
    }


@router.get("/{symbol}/keywords")
def get_ticker_keywords(symbol: str):
    """Preview the merged keyword set used by Layer1 entity extraction."""
    ensure_ticker_alias_table()
    symbol = symbol.upper()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            ticker = _ensure_ticker_exists(cur, symbol)
            cur.execute(
                """SELECT alias, alias_type
                   FROM ticker_aliases
                   WHERE symbol = %s
                   ORDER BY alias_type IS NULL, alias_type, alias""",
                (symbol,),
            )
            aliases = cur.fetchall()
    finally:
        conn.close()

    from backend.pipeline.layer1 import TICKER_KEYWORDS, get_keywords

    return {
        "symbol": ticker["symbol"],
        "name": ticker.get("name"),
        "sector": ticker.get("sector"),
        "builtin_keywords": TICKER_KEYWORDS.get(symbol, []),
        "aliases": list(aliases),
        "keywords": get_keywords(symbol),
    }


@router.post("/{symbol}/aliases")
def add_ticker_alias(symbol: str, req: TickerAliasRequest):
    """Add or update a manually maintained alias for a ticker."""
    ensure_ticker_alias_table()
    symbol = symbol.upper()
    alias = _normalize_alias(req.alias)
    alias_type = _normalize_alias(req.alias_type) if req.alias_type else None

    if not alias:
        raise HTTPException(status_code=400, detail="Alias cannot be empty")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            ticker = _ensure_ticker_exists(cur, symbol)
            cur.execute(
                """INSERT INTO ticker_aliases (symbol, alias, alias_type)
                   VALUES (%s, %s, %s)
                   ON DUPLICATE KEY UPDATE alias_type = VALUES(alias_type)""",
                (symbol, alias, alias_type),
            )
        conn.commit()
    finally:
        conn.close()

    _invalidate_layer1_keyword_cache(symbol)
    return {
        "symbol": ticker["symbol"],
        "name": ticker.get("name"),
        "alias": alias,
        "alias_type": alias_type,
        "status": "saved",
    }


@router.delete("/{symbol}/aliases")
def delete_ticker_alias(symbol: str, alias: str = Query(..., min_length=1)):
    """Delete a manually maintained alias for a ticker."""
    ensure_ticker_alias_table()
    symbol = symbol.upper()
    normalized_alias = _normalize_alias(alias)
    if not normalized_alias:
        raise HTTPException(status_code=400, detail="Alias cannot be empty")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            _ensure_ticker_exists(cur, symbol)
            cur.execute(
                "DELETE FROM ticker_aliases WHERE symbol = %s AND alias = %s",
                (symbol, normalized_alias),
            )
            deleted = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Alias not found for {symbol}: {normalized_alias}")

    _invalidate_layer1_keyword_cache(symbol)
    return {
        "symbol": symbol,
        "alias": normalized_alias,
        "status": "deleted",
    }


@router.get("/{symbol}/ohlc")
def get_ohlc(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Get OHLC data for a symbol."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            query = "SELECT * FROM ohlc WHERE symbol = %s"
            params: list = [symbol.upper()]

            if start:
                query += " AND `date` >= %s"
                params.append(start)
            if end:
                query += " AND `date` <= %s"
                params.append(end)

            query += " ORDER BY `date` ASC"
            cur.execute(query, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    return list(rows) if rows else []


@router.post("")
def add_ticker(req: AddTickerRequest, background_tasks: BackgroundTasks):
    """Add a new ticker and trigger the full fetch+process pipeline."""
    symbol = req.symbol.upper()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT IGNORE INTO tickers (symbol, name) VALUES (%s, %s)",
                (symbol, req.name or symbol),
            )
        conn.commit()
    finally:
        conn.close()

    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=2 * 366)).isoformat()
    end = today.isoformat()

    from backend.api.routers.pipeline import _do_fetch

    background_tasks.add_task(_do_fetch, symbol, start, end, True)
    return {
        "symbol": symbol,
        "status": "added",
        "message": "Fetch + process started in background",
        "start": start,
        "end": end,
    }
