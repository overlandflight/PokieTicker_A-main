-- PokieTicker MySQL 数据库初始化脚本
-- 用法: mysql -u root -p < init.sql

CREATE DATABASE IF NOT EXISTS pokieticker
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE pokieticker;

-- 股票代码表
CREATE TABLE IF NOT EXISTS tickers (
    symbol        VARCHAR(20) PRIMARY KEY COMMENT '股票代码，如 000001.SZ',
    name          VARCHAR(100) COMMENT '股票名称',
    sector        VARCHAR(100) COMMENT '所属行业',
    last_ohlc_fetch   VARCHAR(20) COMMENT '最近OHLC抓取日期',
    last_news_fetch   VARCHAR(20) COMMENT '最近新闻抓取日期'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS ticker_aliases (
    symbol        VARCHAR(20) NOT NULL COMMENT '股票代码',
    alias         VARCHAR(100) NOT NULL COMMENT '别名/简称/产品名',
    alias_type    VARCHAR(30) COMMENT '别名类型',
    PRIMARY KEY (symbol, alias)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 日线行情表
CREATE TABLE IF NOT EXISTS ohlc (
    symbol        VARCHAR(20) NOT NULL COMMENT '股票代码',
    `date`        VARCHAR(20) NOT NULL COMMENT '交易日期',
    `open`        DOUBLE COMMENT '开盘价',
    high          DOUBLE COMMENT '最高价',
    low           DOUBLE COMMENT '最低价',
    `close`       DOUBLE COMMENT '收盘价',
    volume        DOUBLE COMMENT '成交量(手)',
    vwap          DOUBLE COMMENT '成交额(千元)',
    turnover_rate DOUBLE COMMENT '换手率(%)',
    circ_mv       DOUBLE COMMENT '流通市值(万元)',
    total_mv      DOUBLE COMMENT '总市值(万元)',
    transactions  INT COMMENT '预留字段',
    PRIMARY KEY (symbol, `date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 原始新闻表
CREATE TABLE IF NOT EXISTS news_raw (
    id            VARCHAR(64) PRIMARY KEY COMMENT '新闻唯一ID',
    title         TEXT COMMENT '新闻标题',
    description   TEXT COMMENT '新闻内容',
    publisher     VARCHAR(100) COMMENT '来源',
    author        VARCHAR(100) COMMENT '作者',
    published_utc VARCHAR(40) COMMENT '发布时间',
    article_url   TEXT COMMENT '文章链接',
    amp_url       TEXT COMMENT '备用链接',
    tickers_json  TEXT COMMENT '关联股票JSON',
    insights_json TEXT COMMENT '附加信息JSON'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 新闻与股票关联表
CREATE TABLE IF NOT EXISTS news_ticker (
    news_id       VARCHAR(64) NOT NULL,
    symbol        VARCHAR(20) NOT NULL,
    PRIMARY KEY (news_id, symbol),
    FOREIGN KEY (news_id) REFERENCES news_raw(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Layer0 规则过滤结果
CREATE TABLE IF NOT EXISTS layer0_results (
    news_id       VARCHAR(64) NOT NULL,
    symbol        VARCHAR(20) NOT NULL,
    passed        INT NOT NULL COMMENT '1=通过, 0=过滤',
    reason        VARCHAR(50) COMMENT '过滤原因',
    PRIMARY KEY (news_id, symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Layer1 AI情感分析结果
CREATE TABLE IF NOT EXISTS layer1_results (
    news_id       VARCHAR(64) NOT NULL,
    symbol        VARCHAR(20) NOT NULL,
    relevance     VARCHAR(20) COMMENT '相关性',
    key_discussion      TEXT COMMENT '关键讨论',
    chinese_summary     TEXT COMMENT '中文摘要',
    sentiment           VARCHAR(20) COMMENT '情感倾向',
    event_type          VARCHAR(50) COMMENT '主事件类型',
    event_type_tags_json TEXT COMMENT '事件类型标签JSON',
    discussion          TEXT COMMENT '讨论内容',
    reason_growth       TEXT COMMENT '上涨原因',
    reason_decrease     TEXT COMMENT '下跌原因',
    PRIMARY KEY (news_id, symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Layer2 深度分析结果
CREATE TABLE IF NOT EXISTS layer2_results (
    news_id       VARCHAR(64) NOT NULL,
    symbol        VARCHAR(20) NOT NULL,
    discussion    TEXT COMMENT '深度分析',
    growth_reasons  TEXT COMMENT '上涨因素',
    decrease_reasons TEXT COMMENT '下跌因素',
    created_at    VARCHAR(40) COMMENT '创建时间',
    PRIMARY KEY (news_id, symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 新闻与交易日对齐表
CREATE TABLE IF NOT EXISTS news_aligned (
    news_id       VARCHAR(64) NOT NULL,
    symbol        VARCHAR(20) NOT NULL,
    trade_date    VARCHAR(20) NOT NULL COMMENT '对齐的交易日',
    published_utc VARCHAR(40) COMMENT '发布时间',
    session_bucket VARCHAR(30) COMMENT 'A股发布时间归因桶: pre_market/intraday_morning/midday_break/intraday_afternoon/post_market/non_trading',
    label_anchor  VARCHAR(30) COMMENT '收益标签锚点: same_day_open/same_day_close/afternoon_open/next_open',
    ret_t0        DOUBLE COMMENT '当日收益率',
    ret_t1        DOUBLE COMMENT 'T+1收益率',
    ret_t3        DOUBLE COMMENT 'T+3收益率',
    ret_t5        DOUBLE COMMENT 'T+5收益率',
    ret_t10       DOUBLE COMMENT 'T+10收益率',
    PRIMARY KEY (news_id, symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE INDEX idx_news_aligned_symbol_date ON news_aligned(symbol, trade_date);

-- 管道任务状态表
CREATE TABLE IF NOT EXISTS pipeline_tasks (
    task_id        VARCHAR(64) PRIMARY KEY,
    symbol         VARCHAR(20) NOT NULL,
    task_type      VARCHAR(20) NOT NULL COMMENT 'fetch/process/train',
    status         VARCHAR(30) NOT NULL COMMENT 'queued/running/success/failed/partial_success',
    message        TEXT COMMENT '当前状态说明',
    error_text     TEXT COMMENT '失败详情',
    params_json    TEXT COMMENT '任务参数JSON',
    requested_at   VARCHAR(40) NOT NULL,
    started_at     VARCHAR(40) COMMENT '开始时间',
    finished_at    VARCHAR(40) COMMENT '结束时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE INDEX idx_pipeline_tasks_symbol_requested ON pipeline_tasks(symbol, requested_at);

-- 批处理任务表
CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id      VARCHAR(100) PRIMARY KEY,
    symbol        VARCHAR(20),
    status        VARCHAR(30),
    total         INT,
    completed     INT DEFAULT 0,
    created_at    VARCHAR(40),
    finished_at   VARCHAR(40)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 批处理请求映射表
CREATE TABLE IF NOT EXISTS batch_request_map (
    batch_id      VARCHAR(100) NOT NULL,
    custom_id     VARCHAR(200) NOT NULL,
    symbol        VARCHAR(20) NOT NULL,
    article_ids   TEXT NOT NULL,
    PRIMARY KEY (batch_id, custom_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
