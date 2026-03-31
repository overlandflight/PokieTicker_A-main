"""Submit Layer 1 analysis using DeepSeek API for top N tickers.

Usage: python -m backend.batch_submit [--top 50]

注意: DeepSeek 不支持 Anthropic 的 Batch API，
此处改为顺序调用 DeepSeek API 逐批处理。
"""

import json
import sys
from typing import List, Dict, Any

from backend.config import settings
from backend.database import get_conn
from backend.pipeline.layer1 import (
    get_pending_articles, run_layer1,
)


def get_top_tickers(n: int = 50) -> List[Dict[str, Any]]:
    """Get top N tickers by Layer 0 passed count, with pending articles."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT l0.symbol, t.name,
                       SUM(CASE WHEN l0.passed=1 THEN 1 ELSE 0 END) as passed
                FROM layer0_results l0
                JOIN tickers t ON l0.symbol = t.symbol
                GROUP BY l0.symbol, t.name
                ORDER BY passed DESC
                LIMIT %s
            """, (n,))
            rows = cur.fetchall()
    finally:
        conn.close()
    return list(rows)


def main():
    top_n = 50
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--top" and i + 1 < len(sys.argv):
                top_n = int(sys.argv[i + 1])

    print(f"=== Layer 1 DeepSeek Processing (top {top_n} tickers) ===\n")

    tickers = get_top_tickers(top_n)

    total_pending = 0
    for t in tickers:
        total_pending += t["passed"]
    print(f"Top {len(tickers)} tickers, ~{total_pending} Layer0-passed articles")
    print(f"(Already processed by Layer1 will be excluded)\n")

    total_stats = {"processed": 0, "relevant": 0, "irrelevant": 0, "errors": 0}

    for t in tickers:
        symbol = t["symbol"]
        pending = get_pending_articles(symbol)
        if not pending:
            print(f"  {symbol}: no pending articles, skip")
            continue

        print(f"  {symbol}: {len(pending)} pending articles")
        stats = run_layer1(symbol)

        total_stats["processed"] += stats.get("processed", 0)
        total_stats["relevant"] += stats.get("relevant", 0)
        total_stats["irrelevant"] += stats.get("irrelevant", 0)
        total_stats["errors"] += stats.get("errors", 0)

    print(f"\n=== Summary ===")
    print(f"Processed: {total_stats['processed']}")
    print(f"Relevant: {total_stats['relevant']}")
    print(f"Irrelevant: {total_stats['irrelevant']}")
    print(f"Errors: {total_stats['errors']}")


if __name__ == "__main__":
    main()
