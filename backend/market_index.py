"""Benchmark index helpers for A-share relative-return modeling."""

from backend.database import ensure_market_index_table, get_conn
from backend.tushare.client import fetch_index_ohlc


def get_benchmark_symbol_for_equity(symbol: str) -> str:
    code = (symbol or "").split(".")[0]
    suffix = (symbol or "").split(".")[-1].upper() if "." in (symbol or "") else ""

    if suffix == "BJ" or code.startswith(("4", "8")):
        return "000001.SH"
    if code.startswith(("688", "689")):
        return "000688.SH"
    if code.startswith(("300", "301")):
        return "399006.SZ"
    return "000001.SH"


def ensure_benchmark_history(symbol: str, start: str, end: str) -> int:
    ensure_market_index_table()
    try:
        rows = fetch_index_ohlc(symbol, start, end)
    except Exception:
        return 0
    if not rows:
        return 0

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """INSERT INTO market_index_daily
                       (symbol, `date`, `open`, high, low, `close`, volume, amount)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                         `open` = VALUES(`open`),
                         high = VALUES(high),
                         low = VALUES(low),
                         `close` = VALUES(`close`),
                         volume = VALUES(volume),
                         amount = VALUES(amount)""",
                    (
                        symbol,
                        row["date"],
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                        row["amount"],
                    ),
                )
        conn.commit()
    finally:
        conn.close()

    return len(rows)


def ensure_symbol_benchmark_history(symbol: str, start: str, end: str) -> tuple[str, int]:
    benchmark_symbol = get_benchmark_symbol_for_equity(symbol)
    count = ensure_benchmark_history(benchmark_symbol, start, end)
    return benchmark_symbol, count
