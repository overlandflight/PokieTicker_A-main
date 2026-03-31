"""Weekly incremental update: fetch new OHLC + news for all active tickers.

Only fetches data newer than the last fetch date for each ticker.
Run manually or via cron: python -m backend.weekly_update
"""

import json
from datetime import datetime, timedelta, timezone

from backend.config import settings
from backend.database import ensure_ohlc_a_share_columns, get_conn
from backend.tushare.client import fetch_ohlc, fetch_news
from backend.pipeline.alignment import align_news_for_symbol
from backend.pipeline.layer0 import run_layer0
from backend.market_index import ensure_symbol_benchmark_history

TODAY = datetime.now(timezone.utc).date().isoformat()


def update_ohlc(symbol: str, last_fetch: str) -> int:
    """Fetch OHLC data from day after last fetch to today."""
    ensure_ohlc_a_share_columns()
    start = (datetime.fromisoformat(last_fetch) + timedelta(days=1)).date().isoformat()
    if start > TODAY:
        return 0
    ensure_symbol_benchmark_history(symbol, start, TODAY)

    try:
        rows = fetch_ohlc(symbol, start, TODAY)
    except Exception as e:
        print(f"  OHLC error: {e}")
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
                (TODAY, symbol),
            )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def update_news(symbol: str, last_fetch: str) -> int:
    """Fetch news from day after last fetch to today."""
    start = (datetime.fromisoformat(last_fetch) + timedelta(days=1)).date().isoformat()
    if start > TODAY:
        return 0

    try:
        all_articles = fetch_news(symbol, start=start, end=TODAY, max_items=500)
    except Exception as e:
        print(f"  News error: {e}")
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
                cur.execute(
                    "INSERT IGNORE INTO news_ticker (news_id, symbol) VALUES (%s, %s)",
                    (news_id, symbol),
                )

            cur.execute(
                "UPDATE tickers SET last_news_fetch = %s WHERE symbol = %s",
                (TODAY, symbol),
            )
        conn.commit()
    finally:
        conn.close()
    return len(all_articles)


def main():
    print(f"=== Weekly Update: {TODAY} ===\n")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, last_ohlc_fetch, last_news_fetch FROM tickers WHERE last_ohlc_fetch IS NOT NULL ORDER BY symbol"
            )
            tickers = cur.fetchall()
    finally:
        conn.close()

    total_ohlc = 0
    total_news = 0

    for i, t in enumerate(tickers, 1):
        symbol = t["symbol"]
        ohlc_fetch = t["last_ohlc_fetch"] or "2024-01-01"
        news_fetch = t["last_news_fetch"] or ohlc_fetch

        if ohlc_fetch >= TODAY and news_fetch >= TODAY:
            continue

        print(f"[{i}/{len(tickers)}] {symbol}")

        ohlc_count = update_ohlc(symbol, ohlc_fetch)
        if ohlc_count > 0:
            print(f"  OHLC: +{ohlc_count} rows")
        total_ohlc += ohlc_count

        news_count = update_news(symbol, news_fetch)
        if news_count > 0:
            print(f"  News: +{news_count} articles")
        total_news += news_count

        if news_count > 0:
            try:
                align_news_for_symbol(symbol)
                l0 = run_layer0(symbol)
                print(f"  Layer0: {l0.get('passed', 0)} new passed")
            except Exception as e:
                print(f"  Pipeline error: {e}")

    print(f"\n=== Done ===")
    print(f"Updated OHLC: +{total_ohlc} rows")
    print(f"Updated News: +{total_news} articles")


if __name__ == "__main__":
    main()
