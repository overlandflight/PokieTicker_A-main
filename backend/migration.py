"""One-time migration: import existing CSV/JSONL/JSON data into MySQL."""

import csv
import json
import re
from pathlib import Path

from backend.database import ensure_layer1_event_columns, get_conn, init_db
from backend.config import PROJECT_ROOT
from backend.news_events import classify_event_types, event_types_to_json

DATA_DIR = PROJECT_ROOT / "data"

# Map filename patterns to tickers
OHLC_FILES = {
    "BABA": "ohlc_BABA_20201031_20251104.csv",
    "TSLA": "ohlc_TSLA_20231103_20251104.csv",
    "AAPL": "ohlc_AAPL_20231103_20251104.csv",
    "NVDA": "ohlc_NVDA_20231103_20251104.csv",
    "GLD": "ohlc_GLD_20231104_20251105.csv",
}

NEWS_FILES = {
    "BABA": "news_BABA_20201031_202511042backup.jsonl",
    "TSLA": "news_TSLA_20231103_20251104.jsonl",
    "AAPL": "news_AAPL_20231103_20251104.jsonl",
    "NVDA": "news_NVDA_20231103_20251104.jsonl",
    "GLD": "news_GLD_20231104_20251105.jsonl",
}

TICKER_NAMES = {
    "BABA": "Alibaba Group",
    "TSLA": "Tesla Inc",
    "AAPL": "Apple Inc",
    "NVDA": "NVIDIA Corp",
    "GLD": "SPDR Gold Shares",
}

OUTPUT_DIR = DATA_DIR / "output"


def _infer_symbol_from_parsed(parsed: dict, json_file: Path) -> str | None:
    """Infer ticker symbol from parsed payload or filename."""
    for key in ("symbol", "ts_code", "ticker"):
        raw = parsed.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().upper()

    raw_tickers = parsed.get("tickers")
    if isinstance(raw_tickers, list) and raw_tickers:
        first = raw_tickers[0]
        if isinstance(first, str) and first.strip():
            return first.strip().upper()

    stem_upper = json_file.stem.upper()
    for token in re.split(r"[_\-\.]", stem_upper):
        if token in TICKER_NAMES:
            return token

    return None


def migrate_tickers(conn):
    print("Migrating tickers...")
    with conn.cursor() as cur:
        for symbol, name in TICKER_NAMES.items():
            cur.execute(
                "INSERT IGNORE INTO tickers (symbol, name) VALUES (%s, %s)",
                (symbol, name),
            )
    conn.commit()
    print(f"  {len(TICKER_NAMES)} tickers registered")


def migrate_ohlc(conn):
    print("Migrating OHLC data...")
    total = 0
    for symbol, filename in OHLC_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  SKIP {filename} (not found)")
            continue
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            with conn.cursor() as cur:
                for row in reader:
                    date = row.get("date", "").strip()
                    if not date:
                        continue
                    cur.execute(
                        """INSERT IGNORE INTO ohlc
                           (symbol, `date`, `open`, high, low, `close`, volume, vwap, transactions)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            symbol,
                            date,
                            _float(row.get("open")),
                            _float(row.get("high")),
                            _float(row.get("low")),
                            _float(row.get("close")),
                            _float(row.get("volume")),
                            _float(row.get("vwap")),
                            _int(row.get("transactions")),
                        ),
                    )
                    count += 1
        conn.commit()
        total += count
        print(f"  {symbol}: {count} rows")
    print(f"  Total OHLC rows: {total}")


def migrate_news(conn):
    print("Migrating news data...")
    total = 0
    for symbol, filename in NEWS_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  SKIP {filename} (not found)")
            continue
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            with conn.cursor() as cur:
                for line in f:
                    try:
                        art = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    news_id = art.get("id")
                    if not news_id:
                        continue
                    tickers = art.get("tickers") or []
                    cur.execute(
                        """INSERT IGNORE INTO news_raw
                           (id, title, description, publisher, author,
                            published_utc, article_url, amp_url, tickers_json, insights_json)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            news_id,
                            art.get("title"),
                            art.get("description"),
                            art.get("publisher"),
                            art.get("author"),
                            art.get("published_utc"),
                            art.get("article_url"),
                            art.get("amp_url"),
                            json.dumps(tickers),
                            json.dumps(art.get("insights")) if art.get("insights") else None,
                        ),
                    )
                    for tk in tickers:
                        cur.execute(
                            "INSERT IGNORE INTO news_ticker (news_id, symbol) VALUES (%s, %s)",
                            (news_id, tk),
                        )
                    count += 1
        conn.commit()
        total += count
        print(f"  {symbol}: {count} articles")
    print(f"  Total news articles: {total}")


def migrate_parsed_output(conn):
    """Import already-parsed JSON files from data/output/ into layer1_results."""
    print("Migrating parsed output (layer1_results)...")
    if not OUTPUT_DIR.exists():
        print("  output/ directory not found, skipping")
        return
    ensure_layer1_event_columns()
    count = 0
    with conn.cursor() as cur:
        for json_file in OUTPUT_DIR.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
            except (json.JSONDecodeError, ValueError):
                continue
            news_id = parsed.get("id")
            if not news_id:
                continue
            symbol = _infer_symbol_from_parsed(parsed, json_file)
            if not symbol:
                print(f"  SKIP {json_file.name}: symbol not found")
                continue
            event_types = classify_event_types(
                parsed.get("title", ""),
                parsed.get("description", ""),
                parsed.get("key_discussion", ""),
                parsed.get("reason_growth", ""),
                parsed.get("reason_decrease", ""),
            )
            cur.execute(
                """INSERT IGNORE INTO layer1_results
                   (news_id, symbol, relevance, key_discussion, chinese_summary,
                    discussion, event_type, event_type_tags_json, reason_growth, reason_decrease)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    news_id,
                    symbol,
                    parsed.get("relevance", ""),
                    parsed.get("key_discussion", ""),
                    parsed.get("chinese_key_discussion", ""),
                    parsed.get("discussion", ""),
                    event_types[0],
                    event_types_to_json(event_types),
                    parsed.get("reason_growth", ""),
                    parsed.get("reason_decrease", ""),
                ),
            )
            count += 1
    conn.commit()
    print(f"  {count} parsed articles imported")


def _float(val):
    if val is None or val == "":
        return None
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def _int(val):
    if val is None or val == "":
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


def run_migration():
    print("=== Stock News Migration ===")
    init_db()
    conn = get_conn()
    try:
        migrate_tickers(conn)
        migrate_ohlc(conn)
        migrate_news(conn)
        migrate_parsed_output(conn)
    finally:
        conn.close()
    print("=== Migration complete ===")


if __name__ == "__main__":
    run_migration()
