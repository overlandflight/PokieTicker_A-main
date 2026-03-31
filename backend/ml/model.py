"""XGBoost model training and prediction."""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

from backend.ml.features import build_features, build_features_multi, FEATURE_COLS, LEGACY_FEATURE_COLS
from backend.ml.stratification import (
    derive_row_stratification,
    summarize_prediction_stratification,
    summarize_sample_stratification,
)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
MIN_SINGLE_SYMBOL_ROWS = 60
MIN_UNIFIED_ROWS = 100
MIN_SPLIT_ROWS = 10
MIN_MINORITY_CLASS_ROWS = 5


def _prepare_training_dataset(
    df,
    *,
    target_col: str,
    min_rows: int,
    sort_cols: list[str],
    feature_cols: list[str] | None = None,
) -> tuple[dict | None, str | None]:
    """Validate training inputs before fitting a model."""
    feature_cols = FEATURE_COLS if feature_cols is None else feature_cols

    if df.empty:
        return None, "No feature data available"

    working = df.dropna(subset=[target_col]).sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if len(working) < min_rows:
        return None, f"Not enough data ({len(working)} rows)"

    feature_frame = working.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    empty_feature_cols = [col for col in feature_cols if feature_frame[col].notna().sum() == 0]
    if empty_feature_cols:
        cols = ", ".join(empty_feature_cols[:3])
        return None, f"Invalid features: no usable values for {cols}"

    informative_feature_count = sum(
        1 for col in feature_cols if feature_frame[col].nunique(dropna=True) > 1
    )
    if informative_feature_count == 0:
        return None, "Invalid features: all feature columns are constant or empty"

    nonempty_rows = ~feature_frame.isna().all(axis=1)
    dropped_rows = int((~nonempty_rows).sum())
    if dropped_rows:
        working = working.loc[nonempty_rows].reset_index(drop=True)
        feature_frame = feature_frame.loc[nonempty_rows].reset_index(drop=True)
        if len(working) < min_rows:
            return None, f"Not enough valid data after dropping empty feature rows ({len(working)} rows)"

    split_idx = int(len(working) * 0.8)
    test_size = len(working) - split_idx
    if split_idx < MIN_SPLIT_ROWS or test_size < MIN_SPLIT_ROWS:
        return None, f"Time-series split is too small (train={split_idx}, test={test_size})"

    y = working[target_col].astype(int).to_numpy()
    class_counts = np.bincount(y, minlength=2)
    if np.count_nonzero(class_counts) < 2:
        return None, "Target has only one class"
    if int(class_counts.min()) < MIN_MINORITY_CLASS_ROWS:
        return None, f"Target minority class is too small ({int(class_counts.min())} rows)"

    train_counts = np.bincount(y[:split_idx], minlength=2)
    if np.count_nonzero(train_counts) < 2:
        return None, "Training split has only one class"

    test_counts = np.bincount(y[split_idx:], minlength=2)
    if np.count_nonzero(test_counts) < 2:
        return None, "Test split has only one class"

    return {
        "df": working,
        "X": feature_frame.to_numpy(dtype=np.float64),
        "y": y,
        "dates": working["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
        "symbols": working["symbol"].tolist() if "symbol" in working.columns else None,
        "split_idx": split_idx,
        "feature_cols": feature_cols,
    }, None


def _resolve_model_feature_cols(meta: dict | None) -> list[str]:
    if meta and isinstance(meta.get("feature_cols"), list) and meta["feature_cols"]:
        return [str(col) for col in meta["feature_cols"]]
    return list(LEGACY_FEATURE_COLS)


def train(symbol: str, horizon: str = "t1") -> dict:
    """Train XGBoost for a single symbol/horizon. Returns metrics dict."""
    target_col = f"target_{horizon}"

    df = build_features(symbol)
    prepared, error = _prepare_training_dataset(
        df,
        target_col=target_col,
        min_rows=MIN_SINGLE_SYMBOL_ROWS,
        sort_cols=["trade_date"],
    )
    if error:
        return {"error": f"{symbol}: {error}"}

    from xgboost import XGBClassifier

    # Time-series split: last 20% for test
    X = prepared["X"]
    y = prepared["y"]
    dates = prepared["dates"]
    split_idx = prepared["split_idx"]
    feature_cols = prepared["feature_cols"]
    benchmark_symbol = None
    target_definition = "absolute_direction_fallback"
    if "benchmark_symbol" in prepared["df"].columns:
        benchmark_values = prepared["df"]["benchmark_symbol"].dropna().astype(str)
        if not benchmark_values.empty:
            benchmark_symbol = benchmark_values.mode().iloc[0]
    if "benchmark_available" in prepared["df"].columns and float(prepared["df"]["benchmark_available"].sum()) > 0:
        target_definition = "excess_return_vs_benchmark"
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    # Feature importance
    importances = model.feature_importances_
    top_features = sorted(
        zip(feature_cols, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    meta = {
        "symbol": symbol,
        "horizon": horizon,
        "accuracy": round(accuracy, 4),
        "baseline": round(baseline, 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "train_size": split_idx,
        "test_size": len(y_test),
        "train_start": dates[0],
        "train_end": dates[split_idx - 1],
        "test_start": dates[split_idx],
        "test_end": dates[-1],
        "train_stratification": summarize_sample_stratification(prepared["df"], range(0, split_idx)),
        "test_stratification": summarize_sample_stratification(prepared["df"], range(split_idx, len(prepared["df"]))),
        "test_stratified_metrics": summarize_prediction_stratification(
            prepared["df"],
            range(split_idx, len(prepared["df"])),
            y_test,
            y_pred,
        ),
        "feature_cols": feature_cols,
        "target_definition": target_definition,
        "benchmark_symbol": benchmark_symbol,
        "top_features": [{"name": n, "importance": round(v, 4)} for n, v in top_features],
        "trained_at": datetime.now().isoformat(),
    }

    # Save
    model_path = MODELS_DIR / f"{symbol}_{horizon}.joblib"
    meta_path = MODELS_DIR / f"{symbol}_{horizon}_meta.json"
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(meta, indent=2))

    return meta


def train_unified(horizon: str = "t1", symbols: list[str] | None = None) -> dict:
    """Train a single XGBoost on ALL tickers combined. Returns metrics dict."""
    target_col = f"target_{horizon}"

    df = build_features_multi(symbols)
    prepared, error = _prepare_training_dataset(
        df,
        target_col=target_col,
        min_rows=MIN_UNIFIED_ROWS,
        sort_cols=["trade_date", "symbol"],
    )
    if error:
        return {"error": error}

    from xgboost import XGBClassifier

    X = prepared["X"]
    y = prepared["y"]
    dates = prepared["dates"]
    syms = prepared["symbols"] or []
    feature_cols = prepared["feature_cols"]
    benchmark_symbols = []
    target_definition = "absolute_direction_fallback"
    if "benchmark_symbol" in prepared["df"].columns:
        benchmark_symbols = sorted(set(prepared["df"]["benchmark_symbol"].dropna().astype(str)))
    if "benchmark_available" in prepared["df"].columns and float(prepared["df"]["benchmark_available"].sum()) > 0:
        target_definition = "excess_return_vs_benchmark"

    # Time-series split on globally sorted chronological rows.
    split_idx = prepared["split_idx"]
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = XGBClassifier(
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    importances = model.feature_importances_
    top_features = sorted(
        zip(feature_cols, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    meta = {
        "symbol": "UNIFIED",
        "horizon": horizon,
        "accuracy": round(accuracy, 4),
        "baseline": round(baseline, 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "train_size": split_idx,
        "test_size": len(y_test),
        "train_start": dates[0],
        "train_end": dates[split_idx - 1],
        "test_start": dates[split_idx],
        "test_end": dates[-1],
        "train_stratification": summarize_sample_stratification(prepared["df"], range(0, split_idx)),
        "test_stratification": summarize_sample_stratification(prepared["df"], range(split_idx, len(prepared["df"]))),
        "test_stratified_metrics": summarize_prediction_stratification(
            prepared["df"],
            range(split_idx, len(prepared["df"])),
            y_test,
            y_pred,
        ),
        "tickers": sorted(set(syms)),
        "feature_cols": feature_cols,
        "target_definition": target_definition,
        "benchmark_symbols": benchmark_symbols,
        "top_features": [{"name": n, "importance": round(v, 4)} for n, v in top_features],
        "trained_at": datetime.now().isoformat(),
    }

    model_path = MODELS_DIR / f"UNIFIED_{horizon}.joblib"
    meta_path = MODELS_DIR / f"UNIFIED_{horizon}_meta.json"
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(meta, indent=2))

    return meta


def predict(symbol: str, horizon: str = "t1") -> dict:
    """Load model and predict direction for the latest trading day."""
    model_path = MODELS_DIR / f"{symbol}_{horizon}.joblib"
    meta_path = MODELS_DIR / f"{symbol}_{horizon}_meta.json"

    # Fall back to unified model if per-ticker model missing
    if not model_path.exists():
        model_path = MODELS_DIR / f"UNIFIED_{horizon}.joblib"
        meta_path = MODELS_DIR / f"UNIFIED_{horizon}_meta.json"
    if not model_path.exists():
        return {"error": f"No model for {symbol}/{horizon}. Run training first."}

    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    feature_cols = _resolve_model_feature_cols(meta)

    df = build_features(symbol)
    if df.empty:
        return {"error": f"No feature data for {symbol}"}

    # Use the last row (most recent trading day with complete features)
    last_row = df.iloc[-1]
    feature_frame = df.tail(1).reindex(columns=feature_cols)
    X = feature_frame.to_numpy(dtype=np.float64)
    np.nan_to_num(X, copy=False)

    proba = model.predict_proba(X)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])

    # Top feature contributions for this prediction
    importances = model.feature_importances_
    feature_values = {
        col: round(float(np.nan_to_num(feature_frame.iloc[0][col], nan=0.0)), 4)
        for col in feature_cols
    }
    top = sorted(
        zip(feature_cols, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        "symbol": symbol,
        "horizon": horizon,
        "direction": "up" if pred_class == 1 else "down",
        "confidence": round(confidence, 4),
        "date": str(last_row["trade_date"].date()),
        "stratification": derive_row_stratification(last_row),
        "target_definition": meta.get("target_definition", "absolute_direction"),
        "benchmark_symbol": meta.get("benchmark_symbol"),
        "top_features": [
            {"name": n, "value": round(feature_values[n], 4), "importance": round(imp, 4)}
            for n, imp in top
        ],
        "model_accuracy": meta["accuracy"],
        "baseline_accuracy": meta["baseline"],
    }
