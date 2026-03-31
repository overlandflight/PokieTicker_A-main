import logging

import pymysql
from backend.config import settings, PROJECT_ROOT

logger = logging.getLogger(__name__)


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

OHLC_A_SHARE_COLUMN_DEFS = {
    "turnover_rate": (
        "ALTER TABLE ohlc "
        "ADD COLUMN turnover_rate DOUBLE COMMENT '换手率(%)' AFTER vwap"
    ),
    "circ_mv": (
        "ALTER TABLE ohlc "
        "ADD COLUMN circ_mv DOUBLE COMMENT '流通市值(万元)' AFTER turnover_rate"
    ),
    "total_mv": (
        "ALTER TABLE ohlc "
        "ADD COLUMN total_mv DOUBLE COMMENT '总市值(万元)' AFTER circ_mv"
    ),
}

LAYER1_EVENT_COLUMN_DEFS = {
    "event_type": (
        "ALTER TABLE layer1_results "
        "ADD COLUMN event_type VARCHAR(50) COMMENT '主事件类型' AFTER sentiment"
    ),
    "event_type_tags_json": (
        "ALTER TABLE layer1_results "
        "ADD COLUMN event_type_tags_json TEXT COMMENT '事件类型标签JSON' AFTER event_type"
    ),
}

MARKET_INDEX_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS market_index_daily (
    symbol        VARCHAR(20) NOT NULL COMMENT '指数代码',
    `date`        VARCHAR(20) NOT NULL COMMENT '交易日期',
    `open`        DOUBLE COMMENT '开盘价',
    high          DOUBLE COMMENT '最高价',
    low           DOUBLE COMMENT '最低价',
    `close`       DOUBLE COMMENT '收盘价',
    volume        DOUBLE COMMENT '成交量',
    amount        DOUBLE COMMENT '成交额(千元)',
    PRIMARY KEY (symbol, `date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

TICKER_ALIAS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS ticker_aliases (
    symbol        VARCHAR(20) NOT NULL COMMENT '股票代码',
    alias         VARCHAR(100) NOT NULL COMMENT '别名/简称/产品名',
    alias_type    VARCHAR(30) COMMENT '别名类型',
    PRIMARY KEY (symbol, alias)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_news_aligned_attribution_columns_ready = False
_ohlc_a_share_columns_ready = False
_market_index_table_ready = False
_layer1_event_columns_ready = False
_ticker_alias_table_ready = False


def get_conn(database: str | None = None) -> pymysql.connections.Connection:
    target_database = settings.mysql_database if database is None else database
    connect_kwargs = dict(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        charset=settings.mysql_charset,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )
    if target_database:
        connect_kwargs["database"] = target_database
    return pymysql.connect(**connect_kwargs)


def check_db_connection():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            cur.fetchone()
    finally:
        conn.close()


def ensure_news_aligned_attribution_columns(force: bool = False) -> None:
    global _news_aligned_attribution_columns_ready

    if _news_aligned_attribution_columns_ready and not force:
        return

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES LIKE 'news_aligned'")
            if cur.fetchone() is None:
                return

            added_columns = []
            for column, ddl in ATTRIBUTION_COLUMN_DEFS.items():
                cur.execute("SHOW COLUMNS FROM news_aligned LIKE %s", (column,))
                if cur.fetchone() is None:
                    cur.execute(ddl)
                    added_columns.append(column)

        conn.commit()
        if added_columns:
            logger.info("Added news_aligned attribution columns: %s", ", ".join(added_columns))
        _news_aligned_attribution_columns_ready = True
    finally:
        conn.close()


def ensure_ohlc_a_share_columns(force: bool = False) -> None:
    global _ohlc_a_share_columns_ready

    if _ohlc_a_share_columns_ready and not force:
        return

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES LIKE 'ohlc'")
            if cur.fetchone() is None:
                return

            added_columns = []
            for column, ddl in OHLC_A_SHARE_COLUMN_DEFS.items():
                cur.execute("SHOW COLUMNS FROM ohlc LIKE %s", (column,))
                if cur.fetchone() is None:
                    cur.execute(ddl)
                    added_columns.append(column)

        conn.commit()
        if added_columns:
            logger.info("Added ohlc A-share columns: %s", ", ".join(added_columns))
        _ohlc_a_share_columns_ready = True
    finally:
        conn.close()


def ensure_market_index_table(force: bool = False) -> None:
    global _market_index_table_ready

    if _market_index_table_ready and not force:
        return

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(MARKET_INDEX_TABLE_DDL)
        conn.commit()
        _market_index_table_ready = True
    finally:
        conn.close()


def ensure_layer1_event_columns(force: bool = False) -> None:
    global _layer1_event_columns_ready

    if _layer1_event_columns_ready and not force:
        return

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES LIKE 'layer1_results'")
            if cur.fetchone() is None:
                return

            added_columns = []
            for column, ddl in LAYER1_EVENT_COLUMN_DEFS.items():
                cur.execute("SHOW COLUMNS FROM layer1_results LIKE %s", (column,))
                if cur.fetchone() is None:
                    cur.execute(ddl)
                    added_columns.append(column)

        conn.commit()
        if added_columns:
            logger.info("Added layer1 event columns: %s", ", ".join(added_columns))
        _layer1_event_columns_ready = True
    finally:
        conn.close()


def ensure_ticker_alias_table(force: bool = False) -> None:
    global _ticker_alias_table_ready

    if _ticker_alias_table_ready and not force:
        return

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(TICKER_ALIAS_TABLE_DDL)
        conn.commit()
        _ticker_alias_table_ready = True
    finally:
        conn.close()


def split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []

    in_single_quote = False
    in_double_quote = False
    in_backtick = False
    in_line_comment = False
    in_block_comment = False

    i = 0
    while i < len(sql_text):
        ch = sql_text[i]
        nxt = sql_text[i + 1] if i + 1 < len(sql_text) else ""
        prev = sql_text[i - 1] if i > 0 else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                buffer.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_single_quote:
            buffer.append(ch)
            if ch == "'" and nxt == "'":
                buffer.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single_quote = False
            i += 1
            continue

        if in_double_quote:
            buffer.append(ch)
            if ch == '"' and nxt == '"':
                buffer.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_double_quote = False
            i += 1
            continue

        if in_backtick:
            buffer.append(ch)
            if ch == "`" and nxt == "`":
                buffer.append(nxt)
                i += 2
                continue
            if ch == "`":
                in_backtick = False
            i += 1
            continue

        if ch == "#" or (ch == "-" and nxt == "-" and (not prev or prev.isspace())):
            in_line_comment = True
            i += 1 if ch == "#" else 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if ch == "'":
            in_single_quote = True
            buffer.append(ch)
            i += 1
            continue

        if ch == '"':
            in_double_quote = True
            buffer.append(ch)
            i += 1
            continue

        if ch == "`":
            in_backtick = True
            buffer.append(ch)
            i += 1
            continue

        if ch == ";":
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            i += 1
            continue

        buffer.append(ch)
        i += 1

    statement = "".join(buffer).strip()
    if statement:
        statements.append(statement)
    return statements


def init_db():
    """执行 init.sql 初始化数据库（如果表不存在）。"""
    errors = settings.validate_for_startup()
    if errors:
        raise RuntimeError("Cannot initialize database:\n- " + "\n- ".join(errors))

    sql_path = PROJECT_ROOT / "init.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"init.sql not found at {sql_path}")

    # 先连接无数据库，确保数据库存在
    conn = get_conn(database="")
    try:
        with conn.cursor() as cur:
            sql_text = sql_path.read_text(encoding="utf-8")
            for statement in split_sql_statements(sql_text):
                stmt = statement.strip()
                if stmt:
                    try:
                        cur.execute(stmt)
                    except pymysql.err.OperationalError as e:
                        # 忽略"已存在"类错误（如重复索引、重复表）
                        if e.args[0] in (1061, 1050):
                            pass
                        else:
                            raise
        conn.commit()
    finally:
        conn.close()

    logger.info("Database '%s' initialized successfully.", settings.mysql_database)


if __name__ == "__main__":
    init_db()
