"""新浪财经新闻爬取模块。

直接爬取新浪财经个股新闻页面，完全免费无限制。
目标URL: http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?symbol=sz000001&page=1
"""

import hashlib
import logging
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_thread_local = threading.local()

BASE_URL = "http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session
    return session


def _ts_code_to_sina_symbol(ts_code: str) -> str:
    """将 Tushare 股票代码转为新浪格式。

    000001.SZ -> sz000001
    600519.SH -> sh600519
    """
    parts = ts_code.split(".")
    if len(parts) == 2:
        code, exchange = parts
        prefix = exchange.lower()  # SZ -> sz, SH -> sh
        return f"{prefix}{code}"

    # Fallback: infer exchange from code prefix
    code = ts_code.strip()
    if code.startswith("6") or code.startswith("9"):
        return f"sh{code}"
    elif code.startswith("0") or code.startswith("3") or code.startswith("2"):
        return f"sz{code}"
    else:
        logger.warning("Cannot determine exchange for symbol: %s, defaulting to sh", ts_code)
        return f"sh{code}"


def fetch_sina_news(
    ts_code: str,
    start: str = "",
    end: str = "",
    max_pages: int = 5,
    max_items: int = 200,
    fetch_content: bool = True,
) -> List[Dict[str, Any]]:
    """爬取新浪财经个股新闻。

    Args:
        ts_code: Tushare 格式股票代码，如 '000001.SZ'
        start: 开始日期 'YYYY-MM-DD'（可选，用于过滤）
        end: 结束日期 'YYYY-MM-DD'（可选，用于过滤）
        max_pages: 最大爬取页数
        max_items: 最大获取条数
        fetch_content: 是否抓取文章正文（默认True）

    Returns:
        新闻列表，字段适配 news_raw 表结构。
    """
    sina_symbol = _ts_code_to_sina_symbol(ts_code)
    session = _get_session()
    articles = []

    start_date = datetime.strptime(start, "%Y-%m-%d") if start else None
    end_date = datetime.strptime(end, "%Y-%m-%d") if end else None

    for page in range(1, max_pages + 1):
        if len(articles) >= max_items:
            break

        url = f"{BASE_URL}?symbol={sina_symbol}&page={page}"
        try:
            resp = session.get(url, timeout=15)
            resp.encoding = "gb2312"
            if resp.status_code != 200:
                logger.warning("Sina news page %d returned %d", page, resp.status_code)
                break
        except Exception as e:
            logger.warning("Sina news request error page=%d: %s", page, e)
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # 新闻列表在 class="datelist" 的 ul 中
        date_list = soup.find("div", class_="datelist")
        if not date_list:
            # 备用：尝试查找包含新闻链接的列表区域
            date_list = soup.find("div", {"id": "js_ggzx"})
        if not date_list:
            # 再备用：直接在页面中查找所有带日期的链接
            date_list = soup

        # 解析新闻条目：通常格式为 "日期 时间 标题(链接)"
        items = date_list.find_all("a") if date_list else []
        page_has_items = False

        for link in items:
            href = link.get("href", "")
            title = link.get_text(strip=True)

            if not title or not href:
                continue
            # 过滤非新闻链接
            if "finance.sina.com.cn" not in href and "stock.sina.com.cn" not in href:
                continue
            if len(title) < 5:
                continue

            # 提取日期：在链接前面的文本节点中
            date_str = ""
            prev = link.previous_sibling
            if prev and isinstance(prev, str):
                # 通常格式: "2024-01-15 09:30  "
                date_match = re.search(r"(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2})?", prev)
                if date_match:
                    date_str = date_match.group(1)
                    time_str = date_match.group(2) or "00:00"
                    date_str = f"{date_str} {time_str}:00"

            if not date_str:
                # 尝试从 href 中提取日期
                url_date = re.search(r"/(\d{4})-(\d{2})-(\d{2})/", href)
                if url_date:
                    date_str = f"{url_date.group(1)}-{url_date.group(2)}-{url_date.group(3)} 00:00:00"
                else:
                    url_date2 = re.search(r"/(\d{4})(\d{2})(\d{2})/", href)
                    if url_date2:
                        date_str = f"{url_date2.group(1)}-{url_date2.group(2)}-{url_date2.group(3)} 00:00:00"

            if not date_str:
                continue

            page_has_items = True

            # 日期过滤
            try:
                article_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
            except ValueError:
                continue

            if start_date and article_date < start_date:
                # 新浪按时间倒序，如果已经早于起始日期，后面的更早，可以停止
                break

            if end_date and article_date > end_date:
                continue

            # 抓取文章正文
            description = ""
            if fetch_content and href:
                try:
                    description = fetch_sina_news_detail(href)
                    time.sleep(0.3)  # 礼貌延迟
                except Exception as e:
                    logger.debug("Failed to fetch article content: %s", e)

            # 生成唯一ID
            news_id = hashlib.md5(f"{title}_{date_str}_{href}".encode("utf-8")).hexdigest()

            articles.append({
                "id": news_id,
                "title": title,
                "description": description or title,  # 如果正文获取失败，用标题作为描述
                "publisher": "新浪财经",
                "author": "",
                "published_utc": date_str,
                "article_url": href,
                "amp_url": "",
                "tickers": [ts_code],
                "insights": None,
            })

            if len(articles) >= max_items:
                break

        if not page_has_items:
            break

        # 礼貌延迟
        time.sleep(0.5)

    return articles[:max_items]


def fetch_sina_news_detail(article_url: str) -> str:
    """获取单篇新闻正文。

    Args:
        article_url: 新闻文章URL

    Returns:
        新闻正文文本。
    """
    session = _get_session()
    try:
        resp = session.get(article_url, timeout=15)
        resp.encoding = resp.apparent_encoding
        if resp.status_code != 200:
            return ""
    except Exception as e:
        logger.warning("Sina article fetch error: %s", e)
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # 新浪新闻正文通常在 id="artibody" 或 class="article" 中
    content_div = (
        soup.find("div", {"id": "artibody"})
        or soup.find("div", class_="article")
        or soup.find("div", {"id": "article"})
        or soup.find("div", class_="art_t")
    )

    if content_div:
        # 移除脚本和样式标签
        for tag in content_div.find_all(["script", "style"]):
            tag.decompose()
        return content_div.get_text(separator="\n", strip=True)

    return ""
