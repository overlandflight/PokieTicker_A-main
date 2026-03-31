"""Layer 2: On-demand DeepSeek deep analysis.

Triggered when user clicks a news article. Cached in layer2_results.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from openai import OpenAI

from backend.config import settings
from backend.database import get_conn

MODEL = settings.deepseek_model


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "；".join(str(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def get_cached(news_id: str, symbol: str) -> Optional[Dict[str, Any]]:
    """Check if a deep analysis is already cached."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM layer2_results WHERE news_id = %s AND symbol = %s",
                (news_id, symbol),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return row


def analyze_article(news_id: str, symbol: str) -> Dict[str, Any]:
    """Run deep DeepSeek analysis on a single article. Returns cached if available."""
    cached = get_cached(news_id, symbol)
    if cached:
        return cached

    # Fetch article data
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title, description, article_url FROM news_raw WHERE id = %s",
                (news_id,),
            )
            article = cur.fetchone()
    finally:
        conn.close()

    if not article:
        return {"error": "Article not found"}

    client = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=90.0,
    )

    prompt = f"""你是一位资深金融分析师。请对以下新闻文章对 {symbol} 股票的影响进行深度分析。

标题: {article['title']}

内容: {article['description'] or '无详细内容'}

请以JSON格式返回分析:
{{
  "discussion": "详细分析该新闻对 {symbol} 的影响（200-300字）",
  "growth_reasons": "该新闻中可能推动 {symbol} 股价上涨的具体因素（要点列举）",
  "decrease_reasons": "该新闻中可能导致 {symbol} 股价下跌的风险因素（要点列举）"
}}

仅返回JSON。请确保所有分析内容使用简体中文。"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content if response.choices else ""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        parsed = json.loads(text[start:end]) if start >= 0 and end > start else {}
    except json.JSONDecodeError:
        parsed = {"discussion": text, "growth_reasons": "", "decrease_reasons": ""}

    discussion_text = _to_text(parsed.get("discussion", ""))
    growth_text = _to_text(parsed.get("growth_reasons", ""))
    decrease_text = _to_text(parsed.get("decrease_reasons", ""))

    # Cache result
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO layer2_results
                   (news_id, symbol, discussion, growth_reasons, decrease_reasons, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                    discussion=VALUES(discussion), growth_reasons=VALUES(growth_reasons),
                    decrease_reasons=VALUES(decrease_reasons), created_at=VALUES(created_at)""",
                (
                    news_id,
                    symbol,
                    discussion_text,
                    growth_text,
                    decrease_text,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        conn.commit()
    finally:
        conn.close()

    return {
        "news_id": news_id,
        "symbol": symbol,
        "discussion": discussion_text,
        "growth_reasons": growth_text,
        "decrease_reasons": decrease_text,
    }


def generate_story(symbol: str, csv_content: str) -> str:
    """Generate an AI story about stock price movements."""
    client = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=90.0,
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": f"""以下是 {symbol} 的OHLC数据和相关新闻。请根据这些数据生成一篇有深度的投资故事。

数据:
```
{csv_content[-50000:]}
```

故事要求:
1. 从头到尾讲述股价的完整变化历程，重点突出关键转折点
2. 结合新闻事件分析背后的商业和经济因素
3. 开头用1-2句话简要概述该股票的走势
4. 分析市场情绪变化和投资机会
5. 输出HTML格式，使用 <h3> 标题, <p> 段落, <strong> 强调标签

请使用简体中文，约500-1000字，语言生动有叙事感。重点关注:
- 主要价格波动时期及时间线
- 关键新闻事件的影响
- 与竞争对手的比较
- 监管环境和政策影响"""
            }
        ],
    )

    return response.choices[0].message.content if response.choices else ""


def analyze_range(symbol: str, start_date: str, end_date: str, question: Optional[str] = None) -> Dict[str, Any]:
    """Analyze what drove price movement in a date range using DeepSeek."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Get OHLC data for range
            cur.execute(
                "SELECT `date`, `open`, high, low, `close`, volume FROM ohlc WHERE symbol = %s AND `date` >= %s AND `date` <= %s ORDER BY `date` ASC",
                (symbol, start_date, end_date),
            )
            ohlc_rows = cur.fetchall()

            if not ohlc_rows:
                return {"error": "No OHLC data for this range"}

            open_price = ohlc_rows[0]["open"]
            close_price = ohlc_rows[-1]["close"]
            high_price = max(r["high"] for r in ohlc_rows)
            low_price = min(r["low"] for r in ohlc_rows)
            price_change_pct = round((close_price - open_price) / open_price * 100, 2)

            # Get news in range
            cur.execute(
                """SELECT nr.title, l1.chinese_summary, l1.key_discussion,
                          l1.sentiment, l1.reason_growth, l1.reason_decrease,
                          na.trade_date, na.ret_t0
                   FROM news_aligned na
                   JOIN layer1_results l1 ON na.news_id = l1.news_id AND l1.symbol = na.symbol
                   JOIN news_raw nr ON na.news_id = nr.id
                   WHERE na.symbol = %s AND na.trade_date >= %s AND na.trade_date <= %s
                     AND l1.relevance = 'relevant'
                   ORDER BY ABS(COALESCE(na.ret_t0, 0)) DESC
                   LIMIT 30""",
                (symbol, start_date, end_date),
            )
            news_rows = cur.fetchall()
    finally:
        conn.close()

    news_count = len(news_rows)

    # Build news context for prompt
    news_context = ""
    for i, row in enumerate(news_rows[:30], 1):
        ret = f"当日涨跌: {row['ret_t0']*100:.2f}%" if row["ret_t0"] is not None else ""
        news_context += f"\n{i}. [{row['trade_date']}] {row['title']}\n"
        if row.get("chinese_summary"):
            news_context += f"   摘要: {row['chinese_summary']}\n"
        if ret:
            news_context += f"   {ret}\n"

    ohlc_summary = f"开盘: ¥{open_price:.2f}, 收盘: ¥{close_price:.2f}, 最高: ¥{high_price:.2f}, 最低: ¥{low_price:.2f}, 涨跌: {price_change_pct:+.2f}%, 交易日: {len(ohlc_rows)}"

    client = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=90.0,
    )

    question_part = f"用户的具体问题是: {question}。请重点回答这个问题。\n\n" if question else ""

    prompt = f"""你是一位资深金融分析师。请分析 {symbol} 从 {start_date} 到 {end_date} 的股价走势。

价格数据:
{ohlc_summary}

相关新闻（共 {news_count} 篇）:
{news_context if news_context else "该时段无相关新闻"}

{question_part}请以JSON格式返回分析:
{{
  "summary": "1-2句话简要概述",
  "key_events": ["关键事件1", "关键事件2", ...],
  "bullish_factors": ["利好因素1", ...],
  "bearish_factors": ["利空因素1", ...],
  "trend_analysis": "详细趋势分析（100-150字）"
}}

仅返回JSON。请确保所有内容使用简体中文。"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content if response.choices else ""
    except Exception as e:
        logger.warning("DeepSeek API call failed: %s", e)
        text = ""

    analysis = {}
    if text:
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])

        try:
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                analysis = json.loads(cleaned[start_idx:end_idx])
        except json.JSONDecodeError:
            analysis = {
                "summary": text[:200],
                "key_events": [],
                "bullish_factors": [],
                "bearish_factors": [],
                "trend_analysis": text,
            }

    # Fallback: if AI returned empty, generate basic analysis from data
    if not analysis or not analysis.get("summary"):
        direction = "上涨" if price_change_pct > 0 else "下跌" if price_change_pct < 0 else "持平"
        analysis = {
            "summary": f"{symbol} 在 {start_date} 至 {end_date} 期间{direction} {abs(price_change_pct):.2f}%，共 {len(ohlc_rows)} 个交易日，{news_count} 条相关新闻。（AI 分析暂不可用，显示基础数据）",
            "key_events": [f"[{r['trade_date']}] {r['title'][:60]}" for r in news_rows[:5] if r.get("title")],
            "bullish_factors": [r.get("chinese_summary") or r.get("title", "")[:60] for r in news_rows if r.get("sentiment") == "positive"][:3],
            "bearish_factors": [r.get("chinese_summary") or r.get("title", "")[:60] for r in news_rows if r.get("sentiment") == "negative"][:3],
            "trend_analysis": f"期间最高 ¥{high_price:.2f}，最低 ¥{low_price:.2f}，振幅 {(high_price - low_price) / low_price * 100:.1f}%。",
        }

    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "price_change_pct": price_change_pct,
        "open_price": open_price,
        "close_price": close_price,
        "high_price": high_price,
        "low_price": low_price,
        "news_count": news_count,
        "trading_days": len(ohlc_rows),
        "analysis": analysis,
    }
