"""Layer 1: DeepSeek — 50 articles packed into 1 API call.

Strategy:
1. Local keyword extraction: for long descriptions (>500 chars), extract only
   sentences mentioning the company (ticker, name, etc.)
2. Pack 50 articles into a single prompt → 1 API call
3. Get back a compact JSON array
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

from openai import OpenAI

from backend.config import settings
from backend.database import ensure_layer1_event_columns, ensure_ticker_alias_table, get_conn
from backend.news_events import classify_event_types, event_types_to_json

logger = logging.getLogger(__name__)

MODEL = settings.deepseek_model
BATCH_SIZE = 20  # start smaller to reduce timeout risk
MIN_BATCH_SIZE = 5
MAX_OUTPUT_TOKENS = 4096  # enough for 50 articles (~70 tokens each)

# Comprehensive keyword mappings for extraction
# A股常见股票关键词
TICKER_KEYWORDS: Dict[str, List[str]] = {
    "000001.SZ": ["平安银行", "平安", "000001"],
    "600519.SH": ["贵州茅台", "茅台", "600519"],
    "000858.SZ": ["五粮液", "000858"],
    "600036.SH": ["招商银行", "招行", "600036"],
    "601318.SH": ["中国平安", "平安集团", "平安保险", "601318"],
    "000333.SZ": ["美的集团", "美的", "000333"],
    "002594.SZ": ["比亚迪", "byd", "002594"],
    "601888.SH": ["中国中免", "中免", "601888"],
    "300750.SZ": ["宁德时代", "宁德", "catl", "300750"],
    "600900.SH": ["长江电力", "长电", "600900"],
}

_ticker_keyword_cache: Dict[str, List[str]] = {}

# Minimum description length to trigger extraction (shorter ones sent in full)
EXTRACT_THRESHOLD = 500


def _get_keywords(symbol: str) -> List[str]:
    """Get all keywords for a ticker. Falls back to just the symbol."""
    symbol = symbol.upper()
    if symbol in _ticker_keyword_cache:
        return list(_ticker_keyword_cache[symbol])

    kws = [symbol.lower()]
    # 也添加纯数字代码
    code = symbol.split(".")[0] if "." in symbol else symbol
    kws.append(code)
    kws.extend(TICKER_KEYWORDS.get(symbol, []))

    ensure_ticker_alias_table()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM tickers WHERE symbol = %s", (symbol,))
            ticker_row = cur.fetchone()
            if ticker_row and ticker_row.get("name"):
                kws.append(str(ticker_row["name"]))

            cur.execute("SELECT alias FROM ticker_aliases WHERE symbol = %s", (symbol,))
            alias_rows = cur.fetchall()
            kws.extend(str(row["alias"]) for row in alias_rows if row.get("alias"))
    finally:
        conn.close()

    deduped: list[str] = []
    seen = set()
    for kw in kws:
        normalized = str(kw).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)

    _ticker_keyword_cache[symbol] = deduped
    return list(deduped)


def get_keywords(symbol: str) -> List[str]:
    """Public wrapper used by API/debug tooling."""
    return _get_keywords(symbol)


def invalidate_keyword_cache(symbol: str | None = None) -> None:
    """Invalidate cached keyword expansions when ticker metadata changes."""
    global _ticker_keyword_cache

    if symbol is None:
        _ticker_keyword_cache.clear()
        return

    _ticker_keyword_cache.pop(symbol.upper(), None)


def _extract_relevant_text(description: str, symbol: str) -> str:
    """For long descriptions, extract only sentences mentioning the company."""
    if not description:
        return ""

    desc = description.strip()
    if len(desc) < EXTRACT_THRESHOLD:
        return desc

    keywords = _get_keywords(symbol)
    # 中文按句号、感叹号、问号分句
    sentences = re.split(r'(?<=[。！？.!?])\s*', desc)

    relevant: set = set()
    for i, sent in enumerate(sentences):
        lower = sent.lower()
        if any(kw in lower for kw in keywords):
            for j in range(max(0, i - 1), min(len(sentences), i + 2)):
                relevant.add(j)

    if not relevant:
        return " ".join(sentences[:2])

    return " ".join(sentences[i] for i in sorted(relevant))


def _build_batch_prompt(symbol: str, articles: List[Dict[str, Any]]) -> str:
    """Build a single prompt containing up to 50 articles."""
    lines = []
    for i, art in enumerate(articles):
        extract = _extract_relevant_text(art.get("description") or "", symbol)
        lines.append(f"[{i}] {art['title']}")
        if extract:
            lines.append(f"  > {extract}")

    return f"""请对以下 {len(articles)} 篇新闻文章与股票 {symbol} 的相关性进行评级。仅返回JSON数组。

{chr(10).join(lines)}

格式: [{{"i":0,"r":"y"|"n","s":"+"|"-"|"0","e":"摘要","u":"利好原因","d":"利空原因"}}]
r: "y" = 文章具体讨论了 {symbol}, "n" = 无关/仅简单提及
s: "+" 利好, "-" 利空, "0" 中性
e: 10字以内摘要（无关则留空）
u: 可能推动 {symbol} 股价上涨的原因（无则留空）
d: 可能导致 {symbol} 股价下跌的原因（无则留空）

请使用简体中文。
JSON:"""


def get_pending_articles(symbol: str, limit: int = 10000, start_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get articles that passed Layer 0 but haven't been processed by Layer 1.
    
    Args:
        symbol: 股票代码
        limit: 最大返回数量
        start_date: 可选，只返回 published_utc >= start_date 的新闻（格式 YYYY-MM-DD）
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT nr.id, nr.title, nr.description
                FROM news_raw nr
                JOIN news_aligned na ON nr.id = na.news_id AND na.symbol = %s
                JOIN layer0_results l0 ON nr.id = l0.news_id AND l0.symbol = %s
                WHERE l0.passed = 1
                  AND nr.id NOT IN (SELECT news_id FROM layer1_results WHERE symbol = %s)
            """
            params = [symbol, symbol, symbol]
            if start_date:
                sql += " AND na.published_utc >= %s"
                params.append(start_date)
            # 按发布时间倒序，取最新的新闻
            sql += " ORDER BY na.published_utc DESC LIMIT %s"
            params.append(limit)
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()
    return list(rows)


def process_batch_group(
    symbol: str, articles: List[Dict[str, Any]]
) -> Dict[str, int]:
    """Process a group of up to 50 articles in a single API call."""
    client = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=90.0,
    )
    conn = get_conn()
    ensure_layer1_event_columns()

    stats = {"processed": 0, "relevant": 0, "irrelevant": 0, "errors": 0}

    prompt = _build_batch_prompt(symbol, articles)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(response, "usage", None)
        logger.info(
            "DeepSeek Layer1 success: symbol=%s batch=%d model=%s response_id=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            symbol,
            len(articles),
            MODEL,
            getattr(response, "id", None),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
        )

        text = response.choices[0].message.content if response.choices else "[]"

        # Parse JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= start:
            stats["errors"] = len(articles)
            conn.close()
            return stats

        results = json.loads(text[start:end])

        with conn.cursor() as cur:
            for item in results:
                idx = item.get("i")
                if idx is None or idx >= len(articles):
                    stats["errors"] += 1
                    continue

                art = articles[idx]
                is_relevant = item.get("r") in ("y", "relevant")
                relevance = "relevant" if is_relevant else "irrelevant"
                raw_s = item.get("s", "0")
                sentiment = {"+": "positive", "-": "negative"}.get(raw_s, "neutral")
                event_types = classify_event_types(
                    art.get("title"),
                    art.get("description"),
                    item.get("e", ""),
                    item.get("u", ""),
                    item.get("d", ""),
                )

                cur.execute(
                    """INSERT INTO layer1_results
                       (news_id, symbol, relevance, key_discussion, sentiment,
                        event_type, event_type_tags_json, reason_growth, reason_decrease)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                        relevance=VALUES(relevance), key_discussion=VALUES(key_discussion),
                        sentiment=VALUES(sentiment), event_type=VALUES(event_type),
                        event_type_tags_json=VALUES(event_type_tags_json),
                        reason_growth=VALUES(reason_growth),
                        reason_decrease=VALUES(reason_decrease)""",
                    (
                        art["id"],
                        symbol,
                        relevance,
                        item.get("e", ""),
                        sentiment,
                        event_types[0],
                        event_types_to_json(event_types),
                        item.get("u", ""),
                        item.get("d", ""),
                    ),
                )
                stats["processed"] += 1
                if is_relevant:
                    stats["relevant"] += 1
                else:
                    stats["irrelevant"] += 1

        conn.commit()

    except (json.JSONDecodeError, KeyError, Exception) as e:
        stats["errors"] = len(articles)
        print(f"Batch error for {symbol}: {e}")

    conn.close()
    return stats


def _process_batch_with_fallback(
    symbol: str,
    articles: List[Dict[str, Any]],
    batch_size: int,
) -> Dict[str, int]:
    """Process a batch; if the whole request fails, split into smaller batches."""
    if not articles:
        return {"processed": 0, "relevant": 0, "irrelevant": 0, "errors": 0, "api_calls": 0}

    stats = process_batch_group(symbol, articles)
    total = len(articles)
    succeeded = stats["processed"] > 0
    failed_all = stats["errors"] >= total and not succeeded

    if failed_all and total > MIN_BATCH_SIZE and batch_size > MIN_BATCH_SIZE:
        next_batch = max(MIN_BATCH_SIZE, min(batch_size // 2, total - 1))
        merged = {"processed": 0, "relevant": 0, "irrelevant": 0, "errors": 0, "api_calls": 1}
        for i in range(0, total, next_batch):
            child = _process_batch_with_fallback(
                symbol,
                articles[i:i + next_batch],
                next_batch,
            )
            merged["processed"] += child["processed"]
            merged["relevant"] += child["relevant"]
            merged["irrelevant"] += child["irrelevant"]
            merged["errors"] += child["errors"]
            merged["api_calls"] += child["api_calls"]
        return merged

    stats["api_calls"] = 1
    return stats


def run_layer1(symbol: str, max_articles: int = 10000, start_date: Optional[str] = None) -> Dict[str, Any]:
    """Run Layer 1 on all pending articles for a symbol.

    Args:
        symbol: 股票代码
        max_articles: 最多处理文章数
        start_date: 可选，只处理 published_utc >= start_date 的新闻（格式 YYYY-MM-DD）
    """
    articles = get_pending_articles(symbol, limit=max_articles, start_date=start_date)
    if not articles:
        return {"status": "no_pending", "total": 0}

    total_stats = {
        "total": len(articles), "processed": 0, "relevant": 0,
        "irrelevant": 0, "errors": 0, "api_calls": 0,
    }

    for i in range(0, len(articles), BATCH_SIZE):
        chunk = articles[i : i + BATCH_SIZE]
        stats = _process_batch_with_fallback(symbol, chunk, BATCH_SIZE)

        total_stats["processed"] += stats["processed"]
        total_stats["relevant"] += stats["relevant"]
        total_stats["irrelevant"] += stats["irrelevant"]
        total_stats["errors"] += stats["errors"]
        total_stats["api_calls"] += stats["api_calls"]

        print(f"  [{symbol}] Batch {total_stats['api_calls']}: "
              f"{stats['processed']}/{len(chunk)} ok, {stats['relevant']} relevant")

    return total_stats