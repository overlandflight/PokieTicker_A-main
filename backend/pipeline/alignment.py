"""News-to-trading-day alignment with A-share session-aware attribution.

Maps published_utc into China A-share trading sessions and computes
T+0/1/3/5/10 returns from the attributed anchor trading day.
"""

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.database import get_conn

try:
    CHINA_TZ = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:
    CHINA_TZ = timezone(timedelta(hours=8))
PRE_MARKET_END = time(9, 30)
MORNING_SESSION_END = time(11, 30)
MIDDAY_BREAK_END = time(13, 0)
MARKET_CLOSE = time(15, 0)

ATTRIBUTION_COLUMN_DEFS = {
    "session_bucket": (
        "ALTER TABLE news_aligned "
        "ADD COLUMN session_bucket VARCHAR(30) COMMENT 'A股发布时间归因桶' AFTER published_utc"
    ),
    "label_anchor": (
        "ALTER TABLE news_aligned "
        "ADD COLUMN label_anchor VARCHAR(30) COMMENT '收益标签锚点' AFTER session_bucket"
    ),
}


def align_news_for_symbol(symbol: str) -> dict:
    """Align all unaligned news for a symbol to trading days with forward returns."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            _ensure_attribution_columns(cur)

            # Load OHLC dates and closes
            cur.execute(
                "SELECT `date`, `close` FROM ohlc WHERE symbol = %s ORDER BY `date` ASC",
                (symbol,),
            )
            ohlc_rows = cur.fetchall()

            if not ohlc_rows:
                return {"error": "No OHLC data", "aligned": 0}

            dates = [r["date"] for r in ohlc_rows]
            idx = {d: i for i, d in enumerate(dates)}
            close = {r["date"]: float(r["close"]) for r in ohlc_rows}

            # Get news not yet aligned for this symbol
            cur.execute(
                """SELECT nr.id, nr.published_utc
                   FROM news_raw nr
                   JOIN news_ticker nt ON nr.id = nt.news_id
                   WHERE nt.symbol = %s
                   AND nr.id NOT IN (
                       SELECT news_id FROM news_aligned WHERE symbol = %s
                   )""",
                (symbol, symbol),
            )
            news_rows = cur.fetchall()

            aligned_count = 0
            horizons = (1, 3, 5, 10)

            for row in news_rows:
                attribution = _classify_published_attribution(row["published_utc"], idx)
                if attribution is None:
                    continue

                trade_date = attribution["trade_date"]

                i = idx[trade_date]
                prev_d = dates[i - 1] if i > 0 else None

                ret_t0 = _pct(close.get(prev_d), close.get(trade_date)) if prev_d else None

                returns = {}
                for h in horizons:
                    j = i + h
                    if 0 <= j < len(dates):
                        returns[f"ret_t{h}"] = _pct(close.get(trade_date), close.get(dates[j]))
                    else:
                        returns[f"ret_t{h}"] = None

                cur.execute(
                    """INSERT IGNORE INTO news_aligned
                       (news_id, symbol, trade_date, published_utc, session_bucket, label_anchor,
                        ret_t0, ret_t1, ret_t3, ret_t5, ret_t10)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        row["id"],
                        symbol,
                        trade_date,
                        row["published_utc"],
                        attribution["session_bucket"],
                        attribution["label_anchor"],
                        ret_t0,
                        returns.get("ret_t1"),
                        returns.get("ret_t3"),
                        returns.get("ret_t5"),
                        returns.get("ret_t10"),
                    ),
                )
                aligned_count += 1

        conn.commit()
    finally:
        conn.close()
    return {"aligned": aligned_count, "total_news": len(news_rows)}


def _ensure_attribution_columns(cur) -> None:
    for column, ddl in ATTRIBUTION_COLUMN_DEFS.items():
        cur.execute("SHOW COLUMNS FROM news_aligned LIKE %s", (column,))
        if cur.fetchone() is None:
            cur.execute(ddl)


def _parse_published_local(published_utc: Optional[str]) -> Optional[datetime]:
    if not published_utc:
        return None
    try:
        pub = published_utc.strip()
        if not pub:
            return None

        parsed = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=CHINA_TZ)
        return parsed.astimezone(CHINA_TZ)
    except (ValueError, AttributeError):
        return None


def _classify_published_attribution(published_utc: Optional[str], idx: dict) -> Optional[dict[str, str]]:
    published_local = _parse_published_local(published_utc)
    if published_local is None:
        return None

    local_date = published_local.date()
    local_date_str = local_date.isoformat()
    local_time = published_local.timetz().replace(tzinfo=None)

    if local_date_str not in idx:
        trade_date = _find_trade_day(local_date, idx, include_current=True)
        if trade_date is None:
            return None
        return {
            "trade_date": trade_date,
            "session_bucket": "non_trading",
            "label_anchor": "next_open",
        }

    if local_time < PRE_MARKET_END:
        return {
            "trade_date": local_date_str,
            "session_bucket": "pre_market",
            "label_anchor": "same_day_open",
        }

    if local_time < MORNING_SESSION_END:
        return {
            "trade_date": local_date_str,
            "session_bucket": "intraday_morning",
            "label_anchor": "same_day_close",
        }

    if local_time < MIDDAY_BREAK_END:
        return {
            "trade_date": local_date_str,
            "session_bucket": "midday_break",
            "label_anchor": "afternoon_open",
        }

    if local_time < MARKET_CLOSE:
        return {
            "trade_date": local_date_str,
            "session_bucket": "intraday_afternoon",
            "label_anchor": "same_day_close",
        }

    trade_date = _find_trade_day(local_date, idx, include_current=False)
    if trade_date is None:
        return None
    return {
        "trade_date": trade_date,
        "session_bucket": "post_market",
        "label_anchor": "next_open",
    }


def _find_trade_day(start_date: date, idx: dict, *, include_current: bool) -> Optional[str]:
    dt = start_date if include_current else start_date + timedelta(days=1)
    for _ in range(7):
        ds = dt.isoformat()
        if ds in idx:
            return ds
        dt += timedelta(days=1)
    return None


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a
