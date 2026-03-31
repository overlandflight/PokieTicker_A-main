import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

from backend.config import settings
from backend.database import ensure_ohlc_a_share_columns, get_conn
from backend.tushare.client import fetch_ohlc, fetch_news  # noqa: F401
from backend.pipeline.layer0 import run_layer0
from backend.pipeline.layer1 import run_layer1
from backend.pipeline.alignment import align_news_for_symbol
from backend.market_index import ensure_symbol_benchmark_history

import json

router = APIRouter()
PIPELINE_TASK_TABLE = "pipeline_tasks"


class FetchRequest(BaseModel):
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None


class ProcessRequest(BaseModel):
    symbol: str
    batch_size: int = 1000


class TrainRequest(BaseModel):
    symbol: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _task_tracking_is_available(cur) -> bool:
    cur.execute(
        """SELECT 1
           FROM information_schema.tables
           WHERE table_schema = %s AND table_name = %s
           LIMIT 1""",
        (settings.mysql_database, PIPELINE_TASK_TABLE),
    )
    return cur.fetchone() is not None


def _create_pipeline_task(symbol: str, task_type: str, params: Optional[dict] = None, message: str = "Queued") -> str | None:
    task_id = str(uuid4())
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if not _task_tracking_is_available(cur):
                return None
            now = _utc_now_iso()
            cur.execute(
                f"""INSERT INTO {PIPELINE_TASK_TABLE}
                    (task_id, symbol, task_type, status, message, params_json, requested_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (task_id, symbol, task_type, "queued", message, json.dumps(params or {}), now),
            )
        conn.commit()
        return task_id
    finally:
        conn.close()


def _update_pipeline_task(
    task_id: Optional[str],
    *,
    status: str,
    message: Optional[str] = None,
    error_text: Optional[str] = None,
    mark_started: bool = False,
    mark_finished: bool = False,
):
    if not task_id:
        return

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if not _task_tracking_is_available(cur):
                return
            fields = ["status = %s"]
            params: list[object] = [status]
            if message is not None:
                fields.append("message = %s")
                params.append(message)
            if error_text is not None:
                fields.append("error_text = %s")
                params.append(error_text[:4000])
            if mark_started:
                fields.append("started_at = %s")
                params.append(_utc_now_iso())
            if mark_finished:
                fields.append("finished_at = %s")
                params.append(_utc_now_iso())
            params.append(task_id)
            cur.execute(
                f"UPDATE {PIPELINE_TASK_TABLE} SET {', '.join(fields)} WHERE task_id = %s",
                params,
            )
        conn.commit()
    finally:
        conn.close()


def _load_latest_pipeline_task(cur, symbol: str) -> tuple[bool, Optional[dict]]:
    if not _task_tracking_is_available(cur):
        return False, None
    cur.execute(
        f"""SELECT task_id, task_type, status, message, error_text,
                   requested_at, started_at, finished_at
            FROM {PIPELINE_TASK_TABLE}
            WHERE symbol = %s
            ORDER BY requested_at DESC
            LIMIT 1""",
        (symbol,),
    )
    task = cur.fetchone()
    return True, task


@router.post("/train")
def trigger_train(req: TrainRequest, background_tasks: BackgroundTasks):
    """Train XGBoost models (t1 + t3 + t5) for a symbol."""
    symbol = req.symbol.upper()
    task_id = _create_pipeline_task(symbol, "train", {"symbol": symbol}, "Queued model training")
    background_tasks.add_task(_do_train, symbol, task_id)
    return {
        "symbol": symbol,
        "status": "training_started",
        "task_id": task_id,
        "task_tracking_enabled": task_id is not None,
    }


def _do_train(symbol: str, task_id: Optional[str] = None):
    """Background model training. Auto-fetches data if insufficient."""
    from backend.ml.model import train
    from backend.ml.features import build_features

    _update_pipeline_task(task_id, status="running", message="Training models", mark_started=True)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MIN(`date`) AS min_date, MAX(`date`) AS max_date FROM ohlc WHERE symbol = %s",
                (symbol,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row and row["min_date"] and row["max_date"]:
        ensure_symbol_benchmark_history(symbol, row["min_date"], row["max_date"])

    # Check if we have enough data
    df = build_features(symbol)
    if df.empty or len(df) < 60:
        logger.info("Train %s: only %d rows, auto-fetching history...", symbol, len(df))
        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=2 * 366)).isoformat()
        end = today.isoformat()

        # Ensure ticker exists in DB
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT IGNORE INTO tickers (symbol, name) VALUES (%s, %s)",
                    (symbol, symbol),
                )
            conn.commit()
        finally:
            conn.close()

        # Fetch data synchronously (we're already in background)
        _do_fetch(symbol, start, end, auto_train=False)

        # Re-check after fetch
        df = build_features(symbol)
        if df.empty or len(df) < 60:
            logger.warning("Train %s: still only %d rows after fetch, skipping", symbol, len(df))
            message = f"Not enough data ({len(df)} rows) even after fetching"
            _update_pipeline_task(task_id, status="failed", message=message, error_text=message, mark_finished=True)
            return {"error": message}

    results = {}
    for horizon in ["t1", "t3", "t5"]:
        try:
            result = train(symbol, horizon)
            results[horizon] = result
            if "error" in result:
                logger.warning("Train %s/%s failed: %s", symbol, horizon, result["error"])
            else:
                logger.info("Trained %s/%s: accuracy=%.4f", symbol, horizon, result["accuracy"])
        except Exception:
            logger.exception("Train error %s/%s", symbol, horizon)
            results[horizon] = {"error": "training exception"}

    failures = [h for h, result in results.items() if "error" in result]
    failure_details = [f"{h}: {results[h]['error']}" for h in failures]
    if failures and len(failures) == len(results):
        message = f"Training failed for all horizons: {', '.join(failures)}"
        error_text = "; ".join(failure_details)[:4000]
        _update_pipeline_task(task_id, status="failed", message=message, error_text=error_text, mark_finished=True)
    elif failures:
        message = f"Training partially succeeded; failed horizons: {', '.join(failures)}"
        error_text = "; ".join(failure_details)[:4000]
        _update_pipeline_task(
            task_id,
            status="partial_success",
            message=message,
            error_text=error_text,
            mark_finished=True,
        )
    else:
        _update_pipeline_task(task_id, status="success", message="Training completed", mark_finished=True)
    return results


@router.post("/analyze/{symbol}")
def trigger_layer1(symbol: str, background_tasks: BackgroundTasks):
    """手动触发 Layer1 AI 情感分析，只分析最近10篇未分析的新闻（按发布时间倒序）"""
    symbol = symbol.upper()
    # 确保 ticker 存在
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT IGNORE INTO tickers (symbol, name) VALUES (%s, %s)", (symbol, symbol))
        conn.commit()
    finally:
        conn.close()
    # 在后台运行 Layer1，最多分析 10 篇，不限制日期
    background_tasks.add_task(run_layer1, symbol, 10, None)
    return {"symbol": symbol, "status": "analysis_started", "message": "AI 情感分析已开始（最多10篇最新新闻），请稍后刷新页面"}


@router.post("/fetch")
def trigger_fetch(req: FetchRequest, background_tasks: BackgroundTasks):
    """Trigger Tushare data fetch for a symbol (incremental by default)."""
    symbol = req.symbol.upper()
    today = datetime.now(timezone.utc).date()

    # If no explicit start, do incremental: fetch from last_news_fetch + 1 day
    start = req.start
    end = req.end or today.isoformat()
    if not start:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT last_news_fetch FROM tickers WHERE symbol = %s",
                    (symbol,),
                )
                row = cur.fetchone()
                if row and row.get("last_news_fetch"):
                    last = row["last_news_fetch"]
                    # Normalize to date string for comparison
                    last_str = str(last)[:10]
                    if last_str >= end:
                        cur.execute(
                            """SELECT COUNT(*) AS c
                               FROM news_aligned na
                               LEFT JOIN layer0_results l0
                                 ON l0.news_id = na.news_id AND l0.symbol = na.symbol
                               LEFT JOIN layer1_results l1
                                 ON l1.news_id = na.news_id AND l1.symbol = na.symbol
                               WHERE na.symbol = %s
                                 AND COALESCE(l0.passed, 1) = 1
                                 AND l1.news_id IS NULL""",
                            (symbol,),
                        )
                        pending_row = cur.fetchone()
                        pending = int((pending_row or {}).get("c") or 0)
                        if pending > 0:
                            logger.info("Symbol %s up-to-date but has %d pending Layer1 items, processing...", symbol, pending)
                            task_id = _create_pipeline_task(
                                symbol,
                                "process",
                                {"symbol": symbol, "pending": pending},
                                "Queued processing for pending Layer1 items",
                            )
                            background_tasks.add_task(_do_process_only, symbol, task_id)
                            return {
                                "symbol": symbol,
                                "status": "processing_started",
                                "pending": pending,
                                "task_id": task_id,
                                "task_tracking_enabled": task_id is not None,
                            }
                        return {"symbol": symbol, "status": "up_to_date"}
                    start = (datetime.fromisoformat(last_str) + timedelta(days=1)).date().isoformat()
                else:
                    start = (today - timedelta(days=2 * 366)).isoformat()
        finally:
            conn.close()

    logger.info("Triggering fetch for %s (%s ~ %s)", symbol, start, end)
    task_id = _create_pipeline_task(
        symbol,
        "fetch",
        {"symbol": symbol, "start": start, "end": end},
        f"Queued fetch for {start} ~ {end}",
    )
    background_tasks.add_task(_do_fetch, symbol, start, end, True, task_id)
    return {
        "symbol": symbol,
        "status": "fetch_started",
        "start": start,
        "end": end,
        "task_id": task_id,
        "task_tracking_enabled": task_id is not None,
    }


@router.get("/status/{symbol}")
def get_pipeline_status(symbol: str):
    """Get fetch/process progress for a symbol."""
    symbol = symbol.upper()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_ohlc_fetch, last_news_fetch FROM tickers WHERE symbol = %s",
                (symbol,),
            )
            ticker = cur.fetchone() or {}

            cur.execute("SELECT COUNT(*) AS c FROM ohlc WHERE symbol = %s", (symbol,))
            ohlc_count = int((cur.fetchone() or {}).get("c") or 0)

            cur.execute(
                "SELECT COUNT(*) AS c FROM news_ticker WHERE symbol = %s",
                (symbol,),
            )
            raw_news_count = int((cur.fetchone() or {}).get("c") or 0)

            cur.execute(
                "SELECT COUNT(*) AS c FROM news_aligned WHERE symbol = %s",
                (symbol,),
            )
            aligned_count = int((cur.fetchone() or {}).get("c") or 0)

            cur.execute(
                """SELECT COUNT(*) AS c
                   FROM news_ticker nt
                   WHERE nt.symbol = %s
                     AND NOT EXISTS (
                        SELECT 1
                        FROM news_aligned na
                        WHERE na.news_id = nt.news_id AND na.symbol = nt.symbol
                     )""",
                (symbol,),
            )
            pending_alignment = int((cur.fetchone() or {}).get("c") or 0)

            cur.execute(
                "SELECT COUNT(*) AS c FROM layer1_results WHERE symbol = %s",
                (symbol,),
            )
            layer1_count = int((cur.fetchone() or {}).get("c") or 0)

            cur.execute(
                """SELECT COUNT(*) AS c
                   FROM news_aligned na
                   LEFT JOIN layer0_results l0
                     ON l0.news_id = na.news_id AND l0.symbol = na.symbol
                   LEFT JOIN layer1_results l1
                     ON l1.news_id = na.news_id AND l1.symbol = na.symbol
                   WHERE na.symbol = %s
                     AND COALESCE(l0.passed, 1) = 1
                     AND l1.news_id IS NULL""",
                (symbol,),
            )
            pending_layer1 = int((cur.fetchone() or {}).get("c") or 0)

            task_tracking_enabled, latest_task = _load_latest_pipeline_task(cur, symbol)
    finally:
        conn.close()

    last_ohlc = ticker.get("last_ohlc_fetch")
    last_news = ticker.get("last_news_fetch")

    return {
        "symbol": symbol,
        "last_ohlc_fetch": str(last_ohlc)[:10] if last_ohlc else None,
        "last_news_fetch": str(last_news)[:10] if last_news else None,
        "ohlc_count": ohlc_count,
        "raw_news_count": raw_news_count,
        "aligned_count": aligned_count,
        "pending_alignment": pending_alignment,
        "layer1_count": layer1_count,
        "pending_layer1": pending_layer1,
        "deepseek_enabled": bool(settings.deepseek_api_key),
        "task_tracking_enabled": task_tracking_enabled,
        "latest_task": latest_task,
    }


def _run_post_fetch_pipeline(symbol: str, auto_train: bool = True):
    # Run alignment
    logger.info("Running alignment for %s...", symbol)
    align_result = align_news_for_symbol(symbol)
    logger.info("Alignment done for %s: %s", symbol, align_result)

    # Run Layer0 only (过滤新闻，不调用 AI)
    try:
        l0_stats = run_layer0(symbol)
        logger.info("Layer0 done for %s: %s", symbol, l0_stats)
    except Exception:
        logger.exception("Layer0 error for %s", symbol)

    # 注释掉自动 Layer1 分析，改为手动触发
    # if settings.deepseek_api_key:
    #     try:
    #         l1_stats = run_layer1(symbol, max_articles=1000)
    #         logger.info("Layer1 done for %s: %s", symbol, l1_stats)
    #     except Exception:
    #         logger.exception("Layer1 error for %s", symbol)
    # else:
    #     logger.warning("Layer1 skipped for %s: deepseek_api_key is empty", symbol)

    # Auto-train model if enough data
    if auto_train:
        try:
            from backend.ml.model import train
            for horizon in ["t1", "t3", "t5"]:
                result = train(symbol, horizon)
                if "error" in result:
                    logger.info("Skip training %s/%s: %s", symbol, horizon, result["error"])
                else:
                    logger.info("Trained %s/%s: accuracy=%.4f", symbol, horizon, result["accuracy"])
        except Exception:
            logger.exception("Auto-train error for %s", symbol)


def _do_process_only(symbol: str, task_id: Optional[str] = None):
    """Run alignment + Layer0/1 (+ optional training) without fetching remote data."""
    _update_pipeline_task(task_id, status="running", message="Running alignment and analysis", mark_started=True)
    try:
        _run_post_fetch_pipeline(symbol, auto_train=True)
        logger.info("Process-only pipeline complete for %s", symbol)
        _update_pipeline_task(task_id, status="success", message="Processing completed", mark_finished=True)
    except Exception:
        logger.exception("Process-only pipeline error for %s", symbol)
        _update_pipeline_task(
            task_id,
            status="failed",
            message="Processing failed",
            error_text=f"Process-only pipeline error for {symbol}",
            mark_finished=True,
        )


def _do_fetch(symbol: str, start: str, end: str, auto_train: bool = True, task_id: Optional[str] = None):
    """Background fetch of OHLC + news data."""
    _update_pipeline_task(task_id, status="running", message="Fetching OHLC and news", mark_started=True)
    try:
        ensure_ohlc_a_share_columns()

        # Ensure ticker exists in DB first
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT IGNORE INTO tickers (symbol, name) VALUES (%s, %s)",
                    (symbol, symbol),
                )
            conn.commit()
        finally:
            conn.close()

        # OHLC
        logger.info("Fetching OHLC for %s (%s ~ %s)...", symbol, start, end)
        ohlc_rows = fetch_ohlc(symbol, start, end)
        logger.info("Fetched %d OHLC rows for %s", len(ohlc_rows), symbol)
        benchmark_symbol, benchmark_rows = ensure_symbol_benchmark_history(symbol, start, end)
        logger.info("Ensured benchmark %s rows=%d for %s", benchmark_symbol, benchmark_rows, symbol)

        conn = get_conn()
        news_error: Optional[str] = None
        try:
            with conn.cursor() as cur:
                for row in ohlc_rows:
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
                    (end, symbol),
                )
            conn.commit()

            # News - 通过东方财富爬取个股新闻，不限日期范围
            logger.info("Fetching news for %s (unbounded)...", symbol)
            try:
                articles = fetch_news(symbol, start="", end="", max_items=500)
                logger.info("Fetched %d news articles for %s", len(articles), symbol)
                with conn.cursor() as cur:
                    for art in articles:
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
                conn.commit()
                # Only update last_news_fetch after successful news commit
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE tickers SET last_news_fetch = %s WHERE symbol = %s",
                        (end, symbol),
                    )
                conn.commit()
            except Exception as exc:
                logger.exception("News fetch error for symbol=%s", symbol)
                conn.rollback()
                news_error = str(exc)
        finally:
            conn.close()

        if news_error:
            raise RuntimeError(f"News fetch failed for {symbol}: {news_error}")

        _run_post_fetch_pipeline(symbol, auto_train=auto_train)

        logger.info("Fetch pipeline complete for %s", symbol)
        _update_pipeline_task(task_id, status="success", message="Fetch completed", mark_finished=True)
    except Exception as exc:
        logger.exception("Fetch error for %s", symbol)
        _update_pipeline_task(
            task_id,
            status="failed",
            message="Fetch failed",
            error_text=str(exc),
            mark_finished=True,
        )


@router.post("/process")
def trigger_process(req: ProcessRequest):
    """Run Layer 0 filter, then submit Layer 1 for remaining articles."""
    symbol = req.symbol.upper()

    # Step 1: Alignment
    align_result = align_news_for_symbol(symbol)

    # Step 2: Layer 0
    l0_stats = run_layer0(symbol)

    # Step 3: Run Layer 1 (50 articles per API call)
    l1_stats = run_layer1(symbol, max_articles=req.batch_size)

    return {
        "symbol": symbol,
        "alignment": align_result,
        "layer0": l0_stats,
        "layer1": l1_stats,
    }