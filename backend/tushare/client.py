"""数据客户端：使用 Ashare 获取 A股行情，保留东方财富新闻接口。

替代原 AkShare 客户端，提供 A股日线数据、股票搜索等功能。
Ashare 使用腾讯数据源，自动故障转移至新浪备用源，在 Railway 环境中稳定可靠。
新闻获取仍使用原东方财富 API，不受影响。
"""

import hashlib
import time
import logging
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional

import requests as _requests
import akshare as ak
import Ashare as ashare
import pandas as pd

from backend.config import settings

logger = logging.getLogger(__name__)

# ========== 股票基础信息缓存 ==========
_stock_basic_records_cache: Optional[List[Dict[str, str]]] = None
_stock_basic_records_fetched_at = 0.0
_stock_basic_cache_lock = Lock()
STOCK_BASIC_CACHE_TTL_SECONDS = 6 * 60 * 60


def _normalize_stock_basic_row(row: pd.Series) -> Dict[str, str]:
    """将 AkShare 返回的行转为统一格式，并补全市场后缀。

    AkShare 返回的 code 是纯数字（如 '000001'），需要根据交易所规则添加后缀：
    - 上海：以 6、9 开头 -> .SH
    - 深圳：其他 -> .SZ
    """
    raw_code = str(row.get("code") or "")
    if not raw_code:
        return {"ts_code": "", "symbol": "", "name": "", "industry": ""}
    pure_code = raw_code
    # 添加后缀
    if pure_code.startswith(('6', '9')):
        ts_code = f"{pure_code}.SH"
    else:
        ts_code = f"{pure_code}.SZ"
    name = str(row.get("name") or "")
    return {
        "ts_code": ts_code,
        "symbol": pure_code,
        "name": name,
        "industry": "",   # AkShare 基础表不含行业
    }


def _fetch_stock_basic_records() -> Optional[List[Dict[str, str]]]:
    """获取全市场股票基础信息（代码+名称）"""
    try:
        df = ak.stock_info_a_code_name()
        if df.empty:
            return []
        records = [_normalize_stock_basic_row(row) for _, row in df.iterrows()]
        return records
    except Exception as e:
        logger.error("AkShare stock_info_a_code_name error: %s", e)
        return None


def _get_stock_basic_records(force_refresh: bool = False) -> List[Dict[str, str]]:
    global _stock_basic_records_cache, _stock_basic_records_fetched_at

    now = time.monotonic()
    if (
        not force_refresh
        and _stock_basic_records_cache is not None
        and now - _stock_basic_records_fetched_at < STOCK_BASIC_CACHE_TTL_SECONDS
    ):
        return [row.copy() for row in _stock_basic_records_cache]

    with _stock_basic_cache_lock:
        now = time.monotonic()
        if (
            not force_refresh
            and _stock_basic_records_cache is not None
            and now - _stock_basic_records_fetched_at < STOCK_BASIC_CACHE_TTL_SECONDS
        ):
            return [row.copy() for row in _stock_basic_records_cache]

        cached_records = _stock_basic_records_cache
        records = _fetch_stock_basic_records()
        if records is None:
            return [row.copy() for row in cached_records] if cached_records is not None else []

        if records or cached_records is None:
            _stock_basic_records_cache = records
            _stock_basic_records_fetched_at = now
            return [row.copy() for row in records]

        return [row.copy() for row in cached_records]


def _match_stock_basic_records(records: List[Dict[str, str]], query: str, limit: int) -> List[Dict[str, str]]:
    q_upper = query.upper().strip()
    q_lower = query.lower().strip()
    matched: List[Dict[str, str]] = []

    for row in records:
        ts_code = row["ts_code"].upper()
        symbol = row["symbol"].upper()
        name = row["name"].lower()
        if q_upper in ts_code or q_upper in symbol or q_lower in name:
            matched.append({
                "symbol": row["ts_code"],   # 带后缀，用于前端和后续操作
                "name": row["name"],
                "sector": row["industry"],
            })
            if len(matched) >= limit:
                break

    return matched


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_ts_code(ts_code: str) -> str:
    """将 Tushare 格式代码转为纯数字代码"""
    return ts_code.split(".")[0]


def _ts_code_to_ashare(ts_code: str) -> str:
    """将 Tushare 格式代码转为 Ashare 格式代码。

    Tushare 格式: '000001.SZ' / '000001.SH'
    Ashare 格式:  '000001.XSHE' (深圳) / '000001.XSHG' (上海)
    """
    parts = ts_code.split(".")
    if len(parts) == 2:
        code, exchange = parts
        suffix = "XSHG" if exchange.upper() == "SH" else "XSHE"
        return f"{code}.{suffix}"
    # 无后缀时按首位数字推断
    code = ts_code.strip()
    if code.startswith(("6", "9")):
        return f"{code}.XSHG"
    return f"{code}.XSHE"


def _ts_code_to_eastmoney(ts_code: str) -> str:
    """将 Tushare 代码转为东方财富 mTypeAndCode 格式（新闻用，保留原样）"""
    parts = ts_code.split(".")
    if len(parts) == 2:
        code, exchange = parts
        prefix = "1" if exchange.upper() == "SH" else "0"
        return f"{prefix}.{code}"
    code = ts_code.strip()
    if code.startswith("6") or code.startswith("9"):
        return f"1.{code}"
    return f"0.{code}"


def fetch_ohlc(ts_code: str, start: str, end: str) -> List[Dict[str, Any]]:
    """获取A股日线行情数据（使用 Ashare，腾讯数据源，自动故障转移至新浪）。

    Args:
        ts_code: Tushare 股票代码，如 '000001.SZ'
        start: 开始日期 'YYYY-MM-DD'
        end: 结束日期 'YYYY-MM-DD'

    Returns:
        日线数据列表，字段与原 ohlc 表一致。
    """
    ashare_code = _ts_code_to_ashare(ts_code)
    logger.info("Ashare fetch_ohlc: %s -> %s [%s, %s]", ts_code, ashare_code, start, end)

    try:
        df = ashare.get_price(
            ashare_code,
            start_date=start,
            end_date=end,
            frequency="1d",
            fields=["open", "close", "high", "low", "volume"],
        )
    except Exception as e:
        logger.error("Ashare fetch_ohlc error for %s (%s): %s", ts_code, ashare_code, e)
        return []

    if df is None or df.empty:
        logger.warning("Ashare fetch_ohlc returned empty for %s (%s)", ts_code, ashare_code)
        return []

    # Ashare 返回的 DataFrame 以日期为索引（或含 date 列），列名为英文小写
    # 统一将索引重置为 date 列
    if "date" not in df.columns:
        df = df.reset_index()
        # 索引列可能叫 'index' 或 'date'，统一重命名
        if "index" in df.columns:
            df.rename(columns={"index": "date"}, inplace=True)

    # 确保日期列为字符串 YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # 按日期过滤（Ashare 内部已过滤，此处作为保险）
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    df = df.sort_values("date").reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        # volume: Ashare 返回单位为股，转换为手（除以 100）
        raw_volume = _safe_float(row.get("volume"))
        volume_hands = raw_volume / 100.0 if raw_volume is not None else None

        rows.append({
            "date": row["date"],
            "open": _safe_float(row.get("open")),
            "high": _safe_float(row.get("high")),
            "low": _safe_float(row.get("low")),
            "close": _safe_float(row.get("close")),
            "volume": volume_hands,                  # 手
            "vwap": None,                            # Ashare 不提供成交额，特征工程会用 0 填充
            "turnover_rate": None,                   # Ashare 不提供换手率
            "circ_mv": None,                         # Ashare 不提供流通市值
            "total_mv": None,                        # Ashare 不提供总市值
            "transactions": None,                    # Ashare 不提供成交笔数
        })

    logger.info("Ashare fetch_ohlc: fetched %d rows for %s", len(rows), ts_code)
    return rows


def fetch_index_ohlc(ts_code: str, start: str, end: str) -> List[Dict[str, Any]]:
    """获取A股指数日线数据，用于超额收益基准。

    Args:
        ts_code: 指数代码，如 '000001.SH' (上证指数)
        start: 开始日期 'YYYY-MM-DD'
        end: 结束日期 'YYYY-MM-DD'
    """
    # 将 Tushare 指数代码转为 AkShare 代码
    code_part = _normalize_ts_code(ts_code)
    if ts_code.upper().endswith(".SH"):
        ak_symbol = f"sh{code_part}"
    elif ts_code.upper().endswith(".SZ"):
        ak_symbol = f"sz{code_part}"
    else:
        ak_symbol = f"sh{code_part}"

    try:
        df = ak.stock_zh_index_daily(symbol=ak_symbol)
        if df.empty:
            return []
        # 过滤日期范围
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start) & (df["date"] <= end)
        df = df.loc[mask].copy()
        if df.empty:
            return []
        df = df.sort_values("date")
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
                "amount": float(row.get("amount", 0.0)),
            })
        return rows
    except Exception as e:
        logger.error("AkShare fetch_index_ohlc error for %s: %s", ts_code, e)
        return []


def fetch_news(
    ts_code: str,
    start: str = "",
    end: str = "",
    max_items: int = 200,
) -> List[Dict[str, Any]]:
    """获取个股新闻（通过东方财富资讯接口）—— 此部分完全保留原实现，未改动。"""
    import logging
    import requests as _requests
    logger = logging.getLogger(__name__)

    m_type_and_code = _ts_code_to_eastmoney(ts_code)
    start_date = datetime.strptime(start, "%Y-%m-%d") if start else None
    end_date = datetime.strptime(end, "%Y-%m-%d") if end else None

    articles = []
    page_index = 1
    page_size = 100

    while len(articles) < max_items:
        try:
            resp = _requests.get(
                "https://np-listapi.eastmoney.com/comm/wap/getListInfo",
                params={
                    "client": "wap",
                    "type": 1,
                    "mTypeAndCode": m_type_and_code,
                    "pageSize": page_size,
                    "pageIndex": page_index,
                    "param": "list",
                    "name": "zixunlist",
                },
                timeout=15,
            )
            data = resp.json()
        except Exception as e:
            logger.warning("eastmoney news API error for %s: %s", ts_code, e)
            break

        items = (data.get("data") or {}).get("list") or []
        if not items:
            break

        for item in items:
            title = (item.get("Art_Title") or "").strip()
            pub_time = (item.get("Art_ShowTime") or "").strip()
            source = (item.get("Art_MediaName") or "").strip()
            url = (item.get("Art_Url") or "").strip()

            if not title or not pub_time:
                continue

            try:
                article_date = datetime.strptime(pub_time[:10], "%Y-%m-%d")
            except ValueError:
                continue

            #if start_date and article_date < start_date:
             #   continue
            #if end_date and article_date > end_date:
             #   continue

            news_id = hashlib.md5(
                f"{title}_{pub_time}_{url}".encode("utf-8")
            ).hexdigest()

            articles.append({
                "id": news_id,
                "title": title,
                "description": title,
                "publisher": source or "东方财富",
                "author": "",
                "published_utc": pub_time,
                "article_url": url,
                "amp_url": "",
                "tickers": [ts_code],
                "insights": None,
            })

            if len(articles) >= max_items:
                break

        if len(items) < page_size:
            break
        page_index += 1

    logger.info("eastmoney fetched %d news for %s", len(articles), ts_code)
    return articles


def search_tickers(query: str, limit: int = 20) -> List[Dict[str, str]]:
    """搜索A股股票代码（基于本地缓存）。"""
    records = _get_stock_basic_records()
    if not records:
        return []

    matched = _match_stock_basic_records(records, query, limit)
    if matched:
        return matched

    refreshed_records = _get_stock_basic_records(force_refresh=True)
    if refreshed_records == records:
        return []
    return _match_stock_basic_records(refreshed_records, query, limit)


def get_ticker_name(ts_code: str) -> str:
    """根据股票代码获取名称。"""
    target = ts_code.upper()
    for force_refresh in (False, True):
        for row in _get_stock_basic_records(force_refresh=force_refresh):
            if row["ts_code"].upper() == target:
                return row["name"]
    return ""
