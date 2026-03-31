"""Bulk fetch OHLC + news for all tickers missing data.

使用 Tushare Pro API 获取 A 股数据。
"""

import json
import time
from datetime import datetime, timedelta, timezone

from backend.config import settings
from backend.database import ensure_ohlc_a_share_columns, get_conn
from backend.tushare.client import fetch_ohlc, fetch_news, get_ticker_name
from backend.pipeline.alignment import align_news_for_symbol
from backend.pipeline.layer0 import run_layer0
from backend.market_index import ensure_symbol_benchmark_history

# 2 years of data
TODAY = datetime.now(timezone.utc).date()
START = (TODAY - timedelta(days=2 * 366)).isoformat()
END = TODAY.isoformat()


def fetch_and_store_ohlc(symbol: str) -> int:
    """Fetch OHLC data and store in database. Returns row count."""
    ensure_ohlc_a_share_columns()
    ensure_symbol_benchmark_history(symbol, START, END)
    try:
        rows = fetch_ohlc(symbol, START, END)
    except Exception as e:
        print(f"  OHLC error for {symbol}: {e}")
        return 0

    if not rows:
        return 0

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """INSERT INTO ohlc
                       (symbol, `date`, `open`, high, low, `close`, volume, vwap,
                        turnover_rate, circ_mv, total_mv, transactions)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                         `open` = VALUES(`open`),
                         high = VALUES(high),
                         low = VALUES(low),
                         `close` = VALUES(`close`),
                         volume = VALUES(volume),
                         vwap = VALUES(vwap),
                         turnover_rate = COALESCE(VALUES(turnover_rate), turnover_rate),
                         circ_mv = COALESCE(VALUES(circ_mv), circ_mv),
                         total_mv = COALESCE(VALUES(total_mv), total_mv),
                         transactions = VALUES(transactions)""",
                    (symbol, row["date"], row["open"], row["high"], row["low"],
                     row["close"], row["volume"], row["vwap"], row.get("turnover_rate"),
                     row.get("circ_mv"), row.get("total_mv"), row["transactions"]),
                )
            cur.execute(
                "UPDATE tickers SET last_ohlc_fetch = %s WHERE symbol = %s",
                (END, symbol),
            )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def fetch_and_store_news(symbol: str) -> int:
    """Fetch news and store in database. Returns article count."""
    try:
        all_articles = fetch_news(symbol, start=START, end=END, max_items=500)
    except Exception as e:
        print(f"  News error for {symbol}: {e}")
        return 0

    if not all_articles:
        return 0

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for art in all_articles:
                news_id = art.get("id")
                if not news_id:
                    continue
                tickers = art.get("tickers") or []
                cur.execute(
                    """INSERT IGNORE INTO news_raw
                       (id, title, description, publisher, author,
                        published_utc, article_url, amp_url, tickers_json, insights_json)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (news_id, art.get("title"), art.get("description"),
                     art.get("publisher"), art.get("author"), art.get("published_utc"),
                     art.get("article_url"), art.get("amp_url"),
                     json.dumps(tickers),
                     json.dumps(art.get("insights")) if art.get("insights") else None),
                )
                # 关联到当前股票
                cur.execute(
                    "INSERT IGNORE INTO news_ticker (news_id, symbol) VALUES (%s, %s)",
                    (news_id, symbol),
                )

            cur.execute(
                "UPDATE tickers SET last_news_fetch = %s WHERE symbol = %s",
                (END, symbol),
            )
        conn.commit()
    finally:
        conn.close()
    return len(all_articles)


def main():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol FROM tickers WHERE last_ohlc_fetch IS NULL ORDER BY symbol"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    pending = [r["symbol"] for r in rows]
    print(f"=== Bulk Fetch: {len(pending)} tickers pending ===")
    print(f"Date range: {START} to {END}\n")

    total_ohlc = 0
    total_news = 0
    errors = []

    for idx, symbol in enumerate(pending, 1):
        print(f"[{idx}/{len(pending)}] {symbol}")

        # Fetch company name if missing
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT name FROM tickers WHERE symbol = %s", (symbol,)
                )
                name = cur.fetchone()
        finally:
            conn.close()

        if not name or not name["name"]:
            company_name = get_ticker_name(symbol)
            if company_name:
                conn = get_conn()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE tickers SET name = %s WHERE symbol = %s",
                            (company_name, symbol),
                        )
                    conn.commit()
                finally:
                    conn.close()
                print(f"  Name: {company_name}")

        # Fetch OHLC
        ohlc_count = fetch_and_store_ohlc(symbol)
        print(f"  OHLC: {ohlc_count} rows")
        total_ohlc += ohlc_count

        if ohlc_count == 0:
            print(f"  WARNING: No OHLC data, possibly delisted or invalid ticker")
            errors.append(symbol)
            continue

        # Fetch news
        news_count = fetch_and_store_news(symbol)
        print(f"  News: {news_count} articles")
        total_news += news_count

        # Run alignment + layer 0
        try:
            align_news_for_symbol(symbol)
            l0 = run_layer0(symbol)
            passed = l0.get("passed", 0)
            total = l0.get("total", 0)
            print(f"  Layer0: {passed}/{total} passed")
        except Exception as e:
            print(f"  Alignment/Layer0 error: {e}")

        print()

    print(f"\n=== DONE ===")
    print(f"Total OHLC rows: {total_ohlc}")
    print(f"Total news articles: {total_news}")
    if errors:
        print(f"Errors ({len(errors)}): {', '.join(errors)}")


if __name__ == "__main__":
    main()
