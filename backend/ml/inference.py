"""Forecast module: aggregate recent news window and predict future trend.

Combines:
1. Recent news aggregation (7d or 30d window)
2. XGBoost model prediction
3. Similar historical period search
4. Statistical conclusion generation
"""

import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib

from backend import database
from backend.ml.features import build_features, FEATURE_COLS, LEGACY_FEATURE_COLS
from backend.ml.stratification import derive_row_stratification
from backend.news_events import parse_event_types

MODELS_DIR = Path(__file__).parent / "models"


def _load_recent_news(symbol: str, window_days: int, ref_date: str | None = None) -> list[dict]:
    """Load recent news articles within the window."""
    ensure = getattr(database, "ensure_news_aligned_attribution_columns", None)
    if callable(ensure):
        ensure()
    ensure_layer1 = getattr(database, "ensure_layer1_event_columns", None)
    if callable(ensure_layer1):
        ensure_layer1()
    conn = database.get_conn()
    try:
        with conn.cursor() as cur:
            if ref_date is None:
                cur.execute(
                    "SELECT MAX(trade_date) as max_date FROM news_aligned WHERE symbol = %s", (symbol,)
                )
                row = cur.fetchone()
                if row and row["max_date"]:
                    ref_date = row["max_date"]
                else:
                    ref_date = datetime.now().strftime("%Y-%m-%d")
            ref_dt = datetime.strptime(ref_date, "%Y-%m-%d") if isinstance(ref_date, str) else ref_date
            cutoff = (ref_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
            cur.execute(
                """SELECT na.news_id, na.trade_date, nr.title,
                          na.session_bucket, na.label_anchor,
                          nr.description,
                          l1.sentiment, l1.chinese_summary,
                          l1.relevance, l1.key_discussion,
                          l1.event_type, l1.event_type_tags_json,
                          na.ret_t0, na.ret_t1
                   FROM news_aligned na
                   JOIN news_raw nr ON na.news_id = nr.id
                   LEFT JOIN layer1_results l1 ON na.news_id = l1.news_id AND l1.symbol = na.symbol
                   WHERE na.symbol = %s AND na.trade_date >= %s AND na.trade_date <= %s
                   ORDER BY na.trade_date DESC
                   LIMIT 200""",
                (symbol, cutoff, ref_date),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    normalized_rows = []
    for row in rows:
        event_types = parse_event_types(
            row.get("event_type_tags_json"),
            row.get("title"),
            row.get("description"),
            row.get("key_discussion"),
        )
        normalized = dict(row)
        normalized["event_types"] = event_types
        normalized["event_type"] = row.get("event_type") or event_types[0]
        normalized_rows.append(normalized)
    return normalized_rows


def _compute_window_features(
    df: pd.DataFrame,
    window_days: int,
    feature_cols: list[str] | None = None,
) -> np.ndarray | None:
    """Average the feature vectors over the last `window_days` trading days."""
    feature_cols = FEATURE_COLS if feature_cols is None else feature_cols
    if df.empty:
        return None
    n_rows = min(window_days, len(df))
    window_df = df.iloc[-n_rows:]
    vec = window_df.reindex(columns=feature_cols).mean().values.astype(np.float64)
    np.nan_to_num(vec, copy=False)
    return vec


def _find_similar_periods(
    df: pd.DataFrame, window_vec: np.ndarray, window_days: int, top_k: int = 10
) -> list[dict]:
    """Slide a window over history, find the most similar windows."""
    n = len(df)
    if n < window_days + 10:
        return []

    X_raw = df[FEATURE_COLS].values.astype(np.float64)
    np.nan_to_num(X_raw, copy=False)

    cumsum = np.vstack([np.zeros((1, X_raw.shape[1])), np.cumsum(X_raw, axis=0)])
    max_start = n - window_days
    window_vecs = (cumsum[window_days:n + 1] - cumsum[:max_start + 1]) / window_days

    all_vecs = np.vstack([window_vecs, window_vec.reshape(1, -1)])
    mean = np.mean(all_vecs, axis=0)
    std = np.std(all_vecs, axis=0)
    std[std < 1e-10] = 1.0
    all_norm = (all_vecs - mean) / std

    target_norm = all_norm[-1]
    history_norm = all_norm[:-1]

    norms = np.linalg.norm(history_norm, axis=1)
    norms[norms < 1e-10] = 1.0
    target_n = np.linalg.norm(target_norm)
    if target_n < 1e-10:
        target_n = 1.0
    sims = history_norm @ target_norm / (norms * target_n)

    exclude_start = max(0, len(sims) - window_days * 2)
    sims[exclude_start:] = -999

    top_indices = np.argsort(sims)[::-1][:top_k]

    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()

    results = []
    for idx in top_indices:
        if sims[idx] < -900:
            continue
        period_start = dates[idx]
        period_end = dates[min(idx + window_days - 1, n - 1)]
        after_start = idx + window_days
        after_end_t5 = min(after_start + 5, n)
        after_end_t10 = min(after_start + 10, n)

        if after_start >= n:
            continue

        close_vals = df["close"].values
        period_close = close_vals[min(idx + window_days - 1, n - 1)]

        ret_t5 = None
        ret_t10 = None
        if after_end_t5 > after_start:
            ret_t5 = round((close_vals[after_end_t5 - 1] / period_close - 1) * 100, 2)
        if after_end_t10 > after_start:
            ret_t10 = round((close_vals[after_end_t10 - 1] / period_close - 1) * 100, 2)

        window_slice = df.iloc[idx:idx + window_days]
        avg_sentiment = float(window_slice["sentiment_score"].mean())

        results.append({
            "period_start": period_start,
            "period_end": period_end,
            "similarity": round(float(sims[idx]), 4),
            "avg_sentiment": round(avg_sentiment, 3),
            "n_articles": int(window_slice["n_articles"].sum()),
            "ret_after_5d": ret_t5,
            "ret_after_10d": ret_t10,
        })

    return results


def generate_forecast(symbol: str, window_days: int = 7) -> dict:
    """Generate a complete forecast report for a symbol."""
    symbol = symbol.upper()
    df = build_features(symbol)
    if df.empty:
        return {"error": f"No feature data for {symbol}"}

    last_date = df.iloc[-1]["trade_date"].strftime("%Y-%m-%d")
    current_stratification = derive_row_stratification(df.iloc[-1])

    # 1. Recent news
    recent_news = _load_recent_news(symbol, window_days, ref_date=last_date)
    analyzed_all = [n for n in recent_news if n.get("sentiment") in ("positive", "negative", "neutral")]
    analyzed_relevant = [
        n for n in analyzed_all if n.get("relevance") == "relevant"
    ]
    analyzed_news = analyzed_relevant or analyzed_all
    n_pos = sum(1 for n in analyzed_news if n.get("sentiment") == "positive")
    n_neg = sum(1 for n in analyzed_news if n.get("sentiment") == "negative")
    n_neu = sum(1 for n in analyzed_news if n.get("sentiment") == "neutral")
    n_total = len(recent_news)
    n_analyzed = len(analyzed_news)
    n_pending = max(n_total - len(analyzed_all), 0)

    def _impact_score(n):
        score = 0.0
        if n.get("relevance") == "relevant":
            score += 2.0
        sent = n.get("sentiment")
        if sent in ("positive", "negative"):
            score += 1.5
        elif sent == "neutral":
            score += 0.3
        if n.get("ret_t0") is not None:
            score += min(abs(n["ret_t0"]) * 10, 2.0)
        return score

    impact_candidates = [
        n for n in recent_news
        if n.get("relevance") == "relevant"
        and n.get("sentiment") in ("positive", "negative")
        and n.get("ret_t0") is not None
    ]
    if len(impact_candidates) < 5:
        impact_candidates = [
            n for n in recent_news
            if n.get("sentiment") is not None and n.get("ret_t0") is not None
        ]
    impact_sorted = sorted(impact_candidates, key=_impact_score, reverse=True)

    news_summary = {
        "total": n_total,
        "analyzed": n_analyzed,
        "relevant_analyzed": len(analyzed_relevant),
        "analysis_scope": "relevant" if analyzed_relevant else "all",
        "pending": n_pending,
        "positive": n_pos,
        "negative": n_neg,
        "neutral": n_neu,
        "sentiment_ratio": round((n_pos - n_neg) / max(n_analyzed, 1), 3) if n_analyzed > 0 else 0.0,
        "top_headlines": [
            {
                "date": n["trade_date"],
                "title": (n["title"] or "")[:100],
                "sentiment": n.get("sentiment", "unknown"),
                "summary": (n.get("chinese_summary") or "")[:120],
                "session_bucket": n.get("session_bucket"),
                "label_anchor": n.get("label_anchor"),
                "event_type": n.get("event_type"),
                "event_types": n.get("event_types", []),
            }
            for n in recent_news[:10]
        ],
        "top_impact": [
            {
                "news_id": n["news_id"],
                "date": n["trade_date"],
                "title": (n["title"] or "")[:120],
                "sentiment": n.get("sentiment", "unknown"),
                "relevance": n.get("relevance"),
                "session_bucket": n.get("session_bucket"),
                "label_anchor": n.get("label_anchor"),
                "event_type": n.get("event_type"),
                "event_types": n.get("event_types", []),
                "key_discussion": (n.get("key_discussion") or "")[:150],
                "ret_t0": round(n["ret_t0"] * 100, 2) if n.get("ret_t0") is not None else None,
                "ret_t1": round(n["ret_t1"] * 100, 2) if n.get("ret_t1") is not None else None,
            }
            for n in impact_sorted[:5]
        ],
    }

    # 2. Window feature vector
    similarity_window_vec = _compute_window_features(df, window_days)
    if similarity_window_vec is None:
        return {"error": "Cannot compute features"}

    # 3. Model predictions
    prediction = None

    try:
        from backend.ml.lstm_model import predict_lstm
        lstm_result = predict_lstm(symbol)
    except ImportError:
        lstm_result = None
    if lstm_result is not None:
        h = lstm_result["horizon"]
        if prediction is None:
            prediction = {}
        prediction[h] = {
            "direction": lstm_result["direction"],
            "confidence": lstm_result["confidence"],
            "model_type": "LSTM",
            "top_drivers": [],
            "model_accuracy": None,
            "baseline_accuracy": None,
        }

    for horizon in ["t1", "t3", "t5"]:
        model_path = MODELS_DIR / f"{symbol}_{horizon}.joblib"
        meta_path = MODELS_DIR / f"{symbol}_{horizon}_meta.json"
        if not model_path.exists():
            model_path = MODELS_DIR / f"UNIFIED_{horizon}.joblib"
            meta_path = MODELS_DIR / f"UNIFIED_{horizon}_meta.json"
            if not model_path.exists():
                continue

        model = joblib.load(model_path)
        meta = json.loads(meta_path.read_text())
        model_feature_cols = meta.get("feature_cols") or LEGACY_FEATURE_COLS

        # Use window-level averaged features for forecast prediction so different
        # windows (e.g. 7d vs 30d) map to different model inputs.
        model_window_vec = _compute_window_features(df, window_days, feature_cols=model_feature_cols)
        if model_window_vec is None:
            continue
        X = model_window_vec.reshape(1, -1).astype(np.float64)
        np.nan_to_num(X, copy=False)

        proba = model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        feature_frame = df.reindex(columns=model_feature_cols)
        feature_means = feature_frame.mean().values.astype(np.float64)
        feature_stds = feature_frame.std().values.astype(np.float64)
        feature_stds[feature_stds < 1e-10] = 1.0
        importances = model.feature_importances_
        vec = X[0]

        contributions = []
        for i, col in enumerate(model_feature_cols):
            val = float(vec[i])
            z = (val - feature_means[i]) / feature_stds[i]
            contrib = abs(z) * importances[i]
            contributions.append({
                "name": col,
                "value": round(val, 4),
                "importance": round(float(importances[i]), 4),
                "z_score": round(float(z), 2),
                "contribution": round(float(contrib), 4),
            })
        contributions.sort(key=lambda x: x["contribution"], reverse=True)

        if prediction is None:
            prediction = {}
        prediction[horizon] = {
            "direction": "up" if pred_class == 1 else "down",
            "confidence": round(confidence, 4),
            "model_type": "XGBoost",
            "target_definition": meta.get("target_definition", "absolute_direction"),
            "benchmark_symbol": meta.get("benchmark_symbol"),
            "train_stratification": meta.get("train_stratification"),
            "test_stratification": meta.get("test_stratification"),
            "test_stratified_metrics": meta.get("test_stratified_metrics"),
            "top_drivers": contributions[:6],
            "model_accuracy": meta.get("accuracy", 0),
            "baseline_accuracy": meta.get("baseline", 0),
        }

    if prediction is None:
        # No trained model: return news-only summary instead of error
        wlabel = f"过去{window_days}天" if window_days <= 7 else f"过去{window_days}天（约1个月）"
        return {
            "symbol": symbol,
            "window_days": window_days,
            "forecast_date": last_date,
            "news_summary": news_summary,
            "prediction": {},
            "current_stratification": current_stratification,
            "similar_periods": [],
            "similar_stats": {"count": 0, "up_ratio_5d": 0, "up_ratio_10d": 0, "avg_ret_5d": None, "avg_ret_10d": None},
            "conclusion": (
                f"{symbol} 暂无训练好的预测模型。{wlabel}共 {n_total} 条新闻，"
                f"已分析 {n_analyzed} 条（利好 {n_pos} / 利空 {n_neg}）。"
                + (f"仍有 {n_pending} 条待分析。" if n_pending > 0 else "")
                + "需要先运行模型训练。"
            ),
            "no_model": True,
        }

    # 4. Similar historical periods
    similar_periods = _find_similar_periods(df, similarity_window_vec, window_days, top_k=10)

    rets_5 = [p["ret_after_5d"] for p in similar_periods if p["ret_after_5d"] is not None]
    rets_10 = [p["ret_after_10d"] for p in similar_periods if p["ret_after_10d"] is not None]

    similar_stats = {
        "count": len(similar_periods),
        "up_ratio_5d": round(sum(1 for r in rets_5 if r > 0) / max(len(rets_5), 1), 2),
        "up_ratio_10d": round(sum(1 for r in rets_10 if r > 0) / max(len(rets_10), 1), 2),
        "avg_ret_5d": round(float(np.mean(rets_5)), 2) if rets_5 else None,
        "avg_ret_10d": round(float(np.mean(rets_10)), 2) if rets_10 else None,
    }

    # 5. Conclusion
    conclusion = _build_conclusion(
        symbol, window_days, news_summary, prediction, similar_stats
    )

    last_date = df.iloc[-1]["trade_date"].strftime("%Y-%m-%d")

    return {
        "symbol": symbol,
        "window_days": window_days,
        "forecast_date": last_date,
        "news_summary": news_summary,
        "prediction": prediction,
        "current_stratification": current_stratification,
        "similar_periods": similar_periods,
        "similar_stats": similar_stats,
        "conclusion": conclusion,
    }


def _build_conclusion(
    symbol: str,
    window_days: int,
    news_summary: dict,
    prediction: dict,
    similar_stats: dict,
) -> str:
    """Build a Chinese-language conclusion from statistical signals."""
    parts = []

    window_label = f"过去{window_days}天" if window_days <= 7 else f"过去{window_days}天（约1个月）"
    n_total = int(news_summary.get("total", 0) or 0)
    n_analyzed = int(news_summary.get("analyzed", n_total) or 0)
    n_pending = int(news_summary.get("pending", max(n_total - n_analyzed, 0)) or 0)
    ratio = float(news_summary.get("sentiment_ratio", 0.0) or 0.0)

    if n_total == 0:
        parts.append(f"{symbol} 在{window_label}内无相关新闻。")
    elif n_analyzed == 0:
        parts.append(f"{symbol} 在{window_label}共有 {n_total} 条相关新闻，但尚未完成情绪分析。")
    else:
        tone = "偏多" if ratio > 0.1 else "偏空" if ratio < -0.1 else "中性"
        parts.append(
            f"{symbol} {window_label}共有 {n_total} 条相关新闻（已分析 {n_analyzed} 条），"
            f"{news_summary['positive']} 条利好 / {news_summary['negative']} 条利空，"
            f"整体情绪{tone}（{ratio:+.2f}）。"
        )
        if n_pending > 0:
            parts.append(f"仍有 {n_pending} 条新闻尚未完成情绪标注。")

    horizon_labels = [
        ("短期（T+1）", "t1"),
        ("中期（T+3）", "t3"),
        ("中期（T+5）", "t5"),
    ]
    for h_label, h_key in horizon_labels:
        p = prediction.get(h_key)
        if not p:
            continue
        target_definition = p.get("target_definition", "absolute_direction")
        benchmark_symbol = p.get("benchmark_symbol")
        if target_definition == "excess_return_vs_benchmark" and benchmark_symbol:
            direction = "相对基准偏强" if p["direction"] == "up" else "相对基准偏弱"
            benchmark_text = f"（相对基准 {benchmark_symbol}）"
        else:
            direction = "看多" if p["direction"] == "up" else "看空"
            benchmark_text = ""
        confidence = p["confidence"] * 100
        model_tag = f"[{p.get('model_type', 'XGBoost')}]" if p.get("model_type") else ""
        parts.append(f"{model_tag} 模型{h_label}预测: {direction}{benchmark_text}，置信度 {confidence:.0f}%。")

    if similar_stats["count"] > 0:
        up_ratio_5d = similar_stats.get("up_ratio_5d")
        avg_ret_5d = similar_stats.get("avg_ret_5d")
        if up_ratio_5d is not None and avg_ret_5d is not None:
            parts.append(
                f"在 {similar_stats['count']} 个历史相似时段中，"
                f"{up_ratio_5d * 100:.0f}% 在后续5个交易日上涨，"
                f"平均收益 {avg_ret_5d:+.1f}%。"
            )

    signals = []
    t1 = prediction.get("t1", {})
    if t1:
        signals.append(1 if t1["direction"] == "up" else -1)
    t3 = prediction.get("t3", {})
    if t3:
        signals.append(1 if t3["direction"] == "up" else -1)
    t5 = prediction.get("t5", {})
    if t5:
        signals.append(1 if t5["direction"] == "up" else -1)
    if similar_stats.get("up_ratio_5d") is not None:
        signals.append(1 if similar_stats["up_ratio_5d"] > 0.5 else -1)
    if ratio > 0.1:
        signals.append(1)
    elif ratio < -0.1:
        signals.append(-1)

    if signals:
        avg_signal = sum(signals) / len(signals)
        if avg_signal > 0.3:
            parts.append("多信号综合评估：偏向看多。")
        elif avg_signal < -0.3:
            parts.append("多信号综合评估：偏向看空。")
        else:
            parts.append("多信号综合评估：方向不明确，建议观望。")

    return " ".join(parts)
