"""Feature engineering: one row per trading day per ticker."""

import pandas as pd
import numpy as np
from backend.database import ensure_market_index_table, ensure_ohlc_a_share_columns, get_conn
from backend.market_index import get_benchmark_symbol_for_equity

CHINEXT_REFORM_DATE = pd.Timestamp("2020-08-24")
BOARD_BUCKET_IDS = {
    "main_board": 0.0,
    "chinext": 1.0,
    "star_market": 2.0,
    "beijing": 3.0,
}

LEGACY_FEATURE_COLS = [
    # News
    "n_articles", "n_relevant", "n_positive", "n_negative", "n_neutral",
    "sentiment_score", "relevance_ratio", "positive_ratio", "negative_ratio", "has_news",
    # Rolling news
    "sentiment_score_3d", "sentiment_score_5d", "sentiment_score_10d",
    "positive_ratio_3d", "positive_ratio_5d", "positive_ratio_10d",
    "negative_ratio_3d", "negative_ratio_5d", "negative_ratio_10d",
    "news_count_3d", "news_count_5d", "news_count_10d",
    "sentiment_momentum_3d",
    # Price / tech
    "ret_1d", "ret_3d", "ret_5d", "ret_10d",
    "volatility_5d", "volatility_10d",
    "volume_ratio_5d", "gap", "ma5_vs_ma20", "rsi_14", "day_of_week",
]


OPTIONAL_A_SHARE_FEATURE_COLS = [
    "amount_ratio_5d",
    "amount_percentile_20d",
    "turnover_rate_5d",
    "turnover_rate_change",
    "circ_mv_log",
    "total_mv_log",
    "cap_bucket_id",
    "calendar_gap_days",
    "resumed_after_halt",
    "recent_halt_resume_5d",
    "benchmark_ret_1d",
    "benchmark_ret_3d",
    "benchmark_ret_5d",
    "benchmark_ret_10d",
    "benchmark_volatility_5d",
    "benchmark_volatility_10d",
    "excess_strength_5d",
    "excess_strength_10d",
    "mkt_articles",
    "mkt_positive",
    "mkt_negative",
    "mkt_tickers_active",
    "mkt_sentiment",
    "mkt_positive_ratio",
    "mkt_sentiment_3d",
    "mkt_sentiment_5d",
    "mkt_momentum",
    "industry_articles",
    "industry_positive",
    "industry_negative",
    "industry_tickers_active",
    "industry_sentiment",
    "industry_positive_ratio",
    "industry_sentiment_3d",
    "industry_sentiment_5d",
    "industry_momentum",
]


def _pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods, fill_method=None)


def _load_news_features(symbol: str) -> pd.DataFrame:
    """Aggregate news_aligned + layer1_results per trade_date."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT na.trade_date,
                       COUNT(*)                                          AS n_articles,
                       SUM(CASE WHEN l1.relevance = 'relevant' THEN 1 ELSE 0 END) AS n_relevant,
                       SUM(CASE WHEN l1.sentiment = 'positive' THEN 1 ELSE 0 END) AS n_positive,
                       SUM(CASE WHEN l1.sentiment = 'negative' THEN 1 ELSE 0 END) AS n_negative,
                       SUM(CASE WHEN l1.sentiment = 'neutral'  THEN 1 ELSE 0 END) AS n_neutral
                FROM news_aligned na
                JOIN layer1_results l1 ON na.news_id = l1.news_id AND na.symbol = l1.symbol
                WHERE na.symbol = %s
                GROUP BY na.trade_date
                ORDER BY na.trade_date
                """,
                (symbol,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # MySQL returns Decimal/int types; ensure float for numpy compatibility
    for col in ["n_articles", "n_relevant", "n_positive", "n_negative", "n_neutral"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    total = df["n_articles"].clip(lower=1)
    df["sentiment_score"] = (df["n_positive"] - df["n_negative"]) / total
    df["relevance_ratio"] = df["n_relevant"] / total
    df["positive_ratio"] = df["n_positive"] / total
    df["negative_ratio"] = df["n_negative"] / total
    df["has_news"] = 1
    return df


def _finalize_sentiment_context(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return df

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    numeric_cols = [col for col in df.columns if col != "trade_date"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    total = df[f"{prefix}_articles"].clip(lower=1)
    df[f"{prefix}_sentiment"] = (df[f"{prefix}_positive"] - df[f"{prefix}_negative"]) / total
    df[f"{prefix}_positive_ratio"] = df[f"{prefix}_positive"] / total
    df[f"{prefix}_sentiment_3d"] = df[f"{prefix}_sentiment"].rolling(3, min_periods=1).mean()
    df[f"{prefix}_sentiment_5d"] = df[f"{prefix}_sentiment"].rolling(5, min_periods=1).mean()
    df[f"{prefix}_momentum"] = df[f"{prefix}_sentiment_3d"] - df[f"{prefix}_sentiment_5d"]
    return df


def _load_market_sentiment_context() -> pd.DataFrame:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT na.trade_date,
                       COUNT(*) AS mkt_articles,
                       SUM(CASE WHEN l1.sentiment = 'positive' THEN 1 ELSE 0 END) AS mkt_positive,
                       SUM(CASE WHEN l1.sentiment = 'negative' THEN 1 ELSE 0 END) AS mkt_negative,
                       COUNT(DISTINCT na.symbol) AS mkt_tickers_active
                FROM news_aligned na
                JOIN layer1_results l1 ON na.news_id = l1.news_id AND na.symbol = l1.symbol
                GROUP BY na.trade_date
                ORDER BY na.trade_date
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()
    return _finalize_sentiment_context(pd.DataFrame(rows), "mkt")


def _load_industry_sentiment_context(sector: str | None) -> pd.DataFrame:
    if not sector:
        return pd.DataFrame()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT na.trade_date,
                       COUNT(*) AS industry_articles,
                       SUM(CASE WHEN l1.sentiment = 'positive' THEN 1 ELSE 0 END) AS industry_positive,
                       SUM(CASE WHEN l1.sentiment = 'negative' THEN 1 ELSE 0 END) AS industry_negative,
                       COUNT(DISTINCT na.symbol) AS industry_tickers_active
                FROM news_aligned na
                JOIN layer1_results l1 ON na.news_id = l1.news_id AND na.symbol = l1.symbol
                JOIN tickers t ON na.symbol = t.symbol
                WHERE t.sector = %s
                GROUP BY na.trade_date
                ORDER BY na.trade_date
                """,
                (sector,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()
    return _finalize_sentiment_context(pd.DataFrame(rows), "industry")


def _load_ohlc(symbol: str) -> pd.DataFrame:
    ensure_ohlc_a_share_columns()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT o.symbol, o.`date`, o.`open`, o.high, o.low, o.`close`, o.volume,
                          o.vwap AS amount,
                          o.turnover_rate, o.circ_mv, o.total_mv,
                          t.name AS ticker_name, t.sector
                   FROM ohlc o
                   LEFT JOIN tickers t ON o.symbol = t.symbol
                   WHERE o.symbol = %s
                   ORDER BY o.`date`""",
                (symbol,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    # MySQL returns Decimal types; convert to float for numpy/pandas compatibility
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover_rate", "circ_mv", "total_mv"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def _load_benchmark_close(benchmark_symbol: str) -> pd.DataFrame:
    ensure_market_index_table()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT `date`, `close`
                   FROM market_index_daily
                   WHERE symbol = %s
                   ORDER BY `date`""",
                (benchmark_symbol,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=["trade_date", "benchmark_close"])

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["date"])
    df["benchmark_close"] = df["close"].astype(float)
    return df[["trade_date", "benchmark_close"]]


def _add_benchmark_context_features(df: pd.DataFrame) -> pd.DataFrame:
    benchmark_close = pd.to_numeric(df["benchmark_close"], errors="coerce")
    benchmark_returns_1d = _pct_change(benchmark_close)
    df["benchmark_ret_1d"] = _pct_change(benchmark_close, 1).shift(1)
    df["benchmark_ret_3d"] = _pct_change(benchmark_close, 3).shift(1)
    df["benchmark_ret_5d"] = _pct_change(benchmark_close, 5).shift(1)
    df["benchmark_ret_10d"] = _pct_change(benchmark_close, 10).shift(1)
    df["benchmark_volatility_5d"] = benchmark_returns_1d.rolling(5).std().shift(1)
    df["benchmark_volatility_10d"] = benchmark_returns_1d.rolling(10).std().shift(1)
    return df


def _infer_board_bucket(symbol: str) -> str:
    code = (symbol or "").split(".")[0]
    suffix = (symbol or "").split(".")[-1].upper() if "." in (symbol or "") else ""

    if suffix == "BJ" or code.startswith(("4", "8")):
        return "beijing"
    if code.startswith(("688", "689")):
        return "star_market"
    if code.startswith(("300", "301")):
        return "chinext"
    return "main_board"


def _infer_is_st(ticker_name: str | None) -> float:
    name = str(ticker_name or "").upper()
    return 1.0 if name.startswith("ST") or name.startswith("*ST") else 0.0


def _price_limit_ratio(symbol: str, trade_date: pd.Timestamp, is_st: float) -> float:
    if is_st >= 0.5:
        return 0.05

    board_bucket = _infer_board_bucket(symbol)
    if board_bucket == "beijing":
        return 0.30
    if board_bucket == "star_market":
        return 0.20
    if board_bucket == "chinext":
        return 0.20 if trade_date >= CHINEXT_REFORM_DATE else 0.10
    return 0.10


def _percentile_of_last(window_values) -> float:
    series = pd.Series(window_values).dropna()
    if series.empty:
        return np.nan
    last = series.iloc[-1]
    return float((series <= last).sum() / len(series))


def build_features(symbol: str) -> pd.DataFrame:
    """Build feature matrix: one row per trading day.

    All features use shift(1) or past windows to prevent look-ahead leakage.
    Target: whether close > previous close (binary up/down).
    """
    ohlc = _load_ohlc(symbol)
    if ohlc.empty or len(ohlc) < 30:
        return pd.DataFrame()

    news = _load_news_features(symbol)
    benchmark_symbol = get_benchmark_symbol_for_equity(symbol)
    benchmark = _load_benchmark_close(benchmark_symbol)

    # Merge news onto OHLC dates
    df = ohlc.rename(columns={"date": "trade_date"})
    if not news.empty:
        df = df.merge(news, on="trade_date", how="left")
    else:
        for col in ["n_articles", "n_relevant", "n_positive", "n_negative",
                     "n_neutral", "sentiment_score", "relevance_ratio",
                     "positive_ratio", "negative_ratio", "has_news"]:
            df[col] = 0

    # Fill missing news days
    news_cols = ["n_articles", "n_relevant", "n_positive", "n_negative",
                 "n_neutral", "sentiment_score", "relevance_ratio",
                 "positive_ratio", "negative_ratio", "has_news"]
    df[news_cols] = df[news_cols].fillna(0)
    df = df.merge(benchmark, on="trade_date", how="left")
    df["benchmark_close"] = pd.to_numeric(df.get("benchmark_close"), errors="coerce")
    df["benchmark_symbol"] = benchmark_symbol
    df["benchmark_available"] = df["benchmark_close"].notna().astype(float)

    ticker_name = ""
    sector = ""
    if "ticker_name" in df.columns:
        nonempty_names = df["ticker_name"].dropna()
        if not nonempty_names.empty:
            ticker_name = str(nonempty_names.iloc[0])
    if "sector" in df.columns:
        nonempty_sectors = df["sector"].dropna()
        if not nonempty_sectors.empty:
            sector = str(nonempty_sectors.iloc[0])

    market_context = _load_market_sentiment_context()
    if not market_context.empty:
        df = df.merge(market_context, on="trade_date", how="left")

    industry_context = _load_industry_sentiment_context(sector)
    if not industry_context.empty:
        df = df.merge(industry_context, on="trade_date", how="left")

    board_bucket = _infer_board_bucket(symbol)
    df["board_bucket_id"] = BOARD_BUCKET_IDS.get(board_bucket, 0.0)
    df["is_st"] = _infer_is_st(ticker_name)
    df["price_limit_ratio"] = df["trade_date"].apply(
        lambda trade_date: _price_limit_ratio(symbol, trade_date, df["is_st"].iloc[0])
    )
    df["calendar_gap_days"] = df["trade_date"].diff().dt.days.fillna(1.0)
    raw_halt_resume = (df["calendar_gap_days"] > 10).astype(float)
    df["resumed_after_halt"] = raw_halt_resume.shift(1).fillna(0.0)
    df["recent_halt_resume_5d"] = raw_halt_resume.rolling(5, min_periods=1).max().shift(1).fillna(0.0)

    # --- Rolling news features ---
    for w in [3, 5, 10]:
        df[f"sentiment_score_{w}d"] = df["sentiment_score"].rolling(w, min_periods=1).mean()
        df[f"positive_ratio_{w}d"] = df["positive_ratio"].rolling(w, min_periods=1).mean()
        df[f"negative_ratio_{w}d"] = df["negative_ratio"].rolling(w, min_periods=1).mean()
        df[f"news_count_{w}d"] = df["n_articles"].rolling(w, min_periods=1).sum()
    df["sentiment_momentum_3d"] = df["sentiment_score_3d"] - df["sentiment_score_10d"]

    # --- Price / technical features (shifted by 1 to prevent leakage) ---
    close = df["close"]
    df["ret_1d"] = _pct_change(close, 1).shift(1)
    df["ret_3d"] = _pct_change(close, 3).shift(1)
    df["ret_5d"] = _pct_change(close, 5).shift(1)
    df["ret_10d"] = _pct_change(close, 10).shift(1)
    df = _add_benchmark_context_features(df)

    close_returns_1d = _pct_change(close)
    df["volatility_5d"] = close_returns_1d.rolling(5).std().shift(1)
    df["volatility_10d"] = close_returns_1d.rolling(10).std().shift(1)
    df["excess_strength_5d"] = df["ret_5d"] - df["benchmark_ret_5d"]
    df["excess_strength_10d"] = df["ret_10d"] - df["benchmark_ret_10d"]

    avg_vol_5 = df["volume"].rolling(5).mean().shift(1)
    df["volume_ratio_5d"] = (df["volume"].shift(1) / avg_vol_5.clip(lower=1))

    amount = df["amount"].replace(0, np.nan)
    avg_amount_5 = amount.rolling(5).mean().shift(1)
    df["amount_ratio_5d"] = amount.shift(1) / avg_amount_5.clip(lower=1)
    df["amount_percentile_20d"] = amount.shift(1).rolling(20, min_periods=5).apply(_percentile_of_last, raw=False)
    turnover_rate = df["turnover_rate"].replace(0, np.nan)
    df["turnover_rate_5d"] = turnover_rate.rolling(5, min_periods=1).mean().shift(1)
    df["turnover_rate_change"] = turnover_rate.shift(1) - turnover_rate.rolling(5, min_periods=1).mean().shift(1)

    circ_mv = df["circ_mv"].replace(0, np.nan)
    total_mv = df["total_mv"].replace(0, np.nan)
    df["circ_mv_log"] = np.log1p(circ_mv.shift(1))
    df["total_mv_log"] = np.log1p(total_mv.shift(1))
    circ_mv_shifted = circ_mv.shift(1)
    df["cap_bucket_id"] = np.where(
        circ_mv_shifted.isna(),
        0.0,
        np.select(
            [
                circ_mv_shifted < 5e5,
                circ_mv_shifted < 2e6,
            ],
            [0.0, 1.0],
            default=2.0,
        ),
    )

    df["gap"] = (df["open"] / close.shift(1) - 1).shift(1)

    ma5 = close.rolling(5).mean().shift(1)
    ma20 = close.rolling(20).mean().shift(1)
    df["ma5_vs_ma20"] = (ma5 / ma20.clip(lower=0.01) - 1)

    # RSI 14
    delta = close.diff().shift(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.clip(lower=1e-10)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    prev_close = close.shift(1)
    limit_up_price = (prev_close * (1 + df["price_limit_ratio"])).round(2)
    limit_down_price = (prev_close * (1 - df["price_limit_ratio"])).round(2)
    raw_limit_up = ((prev_close.notna()) & (close >= limit_up_price - 0.005)).astype(float)
    raw_limit_down = ((prev_close.notna()) & (close <= limit_down_price + 0.005)).astype(float)
    df["is_limit_up"] = raw_limit_up.shift(1).fillna(0.0)
    df["is_limit_down"] = raw_limit_down.shift(1).fillna(0.0)
    for w in [3, 5, 10]:
        df[f"limit_up_count_{w}d"] = raw_limit_up.rolling(w, min_periods=1).sum().shift(1).fillna(0.0)
        df[f"limit_down_count_{w}d"] = raw_limit_down.rolling(w, min_periods=1).sum().shift(1).fillna(0.0)

    for col in OPTIONAL_A_SHARE_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[OPTIONAL_A_SHARE_FEATURE_COLS] = df[OPTIONAL_A_SHARE_FEATURE_COLS].fillna(0.0)

    df["day_of_week"] = df["trade_date"].dt.dayofweek

    # --- Targets: next-N-day direction ---
    # Keep unavailable future labels as NaN so training can drop them correctly.
    future_t1 = close.shift(-1)
    future_t2 = close.shift(-2)
    future_t3 = close.shift(-3)
    future_t5 = close.shift(-5)
    benchmark_close = df["benchmark_close"]
    benchmark_future_t1 = benchmark_close.shift(-1)
    benchmark_future_t2 = benchmark_close.shift(-2)
    benchmark_future_t3 = benchmark_close.shift(-3)
    benchmark_future_t5 = benchmark_close.shift(-5)
    benchmark_ret_t1 = ((benchmark_future_t1 / benchmark_close) - 1).fillna(0.0)
    benchmark_ret_t2 = ((benchmark_future_t2 / benchmark_close) - 1).fillna(0.0)
    benchmark_ret_t3 = ((benchmark_future_t3 / benchmark_close) - 1).fillna(0.0)
    benchmark_ret_t5 = ((benchmark_future_t5 / benchmark_close) - 1).fillna(0.0)
    stock_ret_t1 = (future_t1 / close) - 1
    stock_ret_t2 = (future_t2 / close) - 1
    stock_ret_t3 = (future_t3 / close) - 1
    stock_ret_t5 = (future_t5 / close) - 1
    df["excess_ret_t1"] = stock_ret_t1 - benchmark_ret_t1
    df["excess_ret_t2"] = stock_ret_t2 - benchmark_ret_t2
    df["excess_ret_t3"] = stock_ret_t3 - benchmark_ret_t3
    df["excess_ret_t5"] = stock_ret_t5 - benchmark_ret_t5
    df["target_t1"] = np.where(future_t1.notna(), (df["excess_ret_t1"] > 0).astype(int), np.nan)
    df["target_t2"] = np.where(future_t2.notna(), (df["excess_ret_t2"] > 0).astype(int), np.nan)
    df["target_t3"] = np.where(future_t3.notna(), (df["excess_ret_t3"] > 0).astype(int), np.nan)
    df["target_t5"] = np.where(future_t5.notna(), (df["excess_ret_t5"] > 0).astype(int), np.nan)

    # Drop rows without enough history
    df = df.dropna(subset=["ret_10d", "rsi_14"]).reset_index(drop=True)

    return df


def build_features_multi(symbols: list[str] | None = None) -> pd.DataFrame:
    """Build combined feature matrix for multiple tickers."""
    if symbols is None:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT symbol FROM ohlc")
                rows = cur.fetchall()
        finally:
            conn.close()
        symbols = [r["symbol"] for r in rows]

    frames = []
    for sym in symbols:
        df = build_features(sym)
        if df.empty:
            continue
        df["symbol"] = sym
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)


FEATURE_COLS = LEGACY_FEATURE_COLS + OPTIONAL_A_SHARE_FEATURE_COLS + [
    "board_bucket_id",
    "is_st",
    "price_limit_ratio",
    "is_limit_up",
    "is_limit_down",
    "limit_up_count_3d",
    "limit_up_count_5d",
    "limit_up_count_10d",
    "limit_down_count_3d",
    "limit_down_count_5d",
    "limit_down_count_10d",
]
