"""Expanding-window cross-validation backtest."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from backend.ml.features import build_features, build_features_multi, FEATURE_COLS
from backend.ml.stratification import derive_row_stratification, summarize_prediction_stratification, summarize_trade_stratification

MODELS_DIR = Path(__file__).parent / "models"
LIMIT_EPSILON = 0.01


@dataclass(slots=True)
class BacktestConstraints:
    min_entry_amount_k: float = 20_000.0
    min_entry_turnover_rate_pct: float = 0.5
    max_exit_extension_days: int = 5
    halt_gap_days: int = 10


def _extract_horizon_days(horizon: str) -> int:
    return int(horizon.removeprefix("t"))


def _is_limit_up_entry(prev_close: float | None, entry_open: float | None, limit_ratio: float | None) -> bool:
    if prev_close in (None, 0) or entry_open is None or limit_ratio is None:
        return False
    limit_up_price = round(prev_close * (1 + limit_ratio), 2)
    return entry_open >= limit_up_price - LIMIT_EPSILON


def _is_limit_down_exit(prev_close: float | None, exit_close: float | None, limit_ratio: float | None) -> bool:
    if prev_close in (None, 0) or exit_close is None or limit_ratio is None:
        return False
    limit_down_price = round(prev_close * (1 - limit_ratio), 2)
    return exit_close <= limit_down_price + LIMIT_EPSILON


def _is_low_liquidity(amount: float | None, turnover_rate: float | None, constraints: BacktestConstraints) -> bool:
    amount_value = 0.0 if amount is None else float(amount)
    turnover_value = 0.0 if turnover_rate is None else float(turnover_rate)
    return (
        amount_value < constraints.min_entry_amount_k
        or turnover_value < constraints.min_entry_turnover_rate_pct
    )


def _is_resumed_after_halt(calendar_gap_days: float | None, constraints: BacktestConstraints) -> bool:
    if calendar_gap_days is None:
        return False
    return float(calendar_gap_days) > constraints.halt_gap_days


def _build_symbol_trade_lookup(df):
    working = df.copy().reset_index(drop=True)
    working["trade_date_str"] = working["trade_date"].dt.strftime("%Y-%m-%d")
    if "symbol" not in working.columns:
        working["symbol"] = "__SINGLE__"

    per_symbol = {}
    position_lookup = {}
    for symbol, group in working.groupby("symbol", sort=False):
        ordered = group.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
        per_symbol[symbol] = ordered
        for idx, row in ordered.iterrows():
            position_lookup[(symbol, row["trade_date_str"])] = idx
    return per_symbol, position_lookup


def _summarize_trade_returns(returns: list[float]) -> dict:
    if not returns:
        return {
            "trades": 0,
            "win_rate": None,
            "avg_return_pct": None,
            "total_return_pct": None,
        }

    compounded = 1.0
    for trade_return in returns:
        compounded *= 1 + trade_return

    return {
        "trades": len(returns),
        "win_rate": round(sum(1 for value in returns if value > 0) / len(returns), 4),
        "avg_return_pct": round(float(np.mean(returns)) * 100, 2),
        "total_return_pct": round((compounded - 1) * 100, 2),
    }


def _evaluate_trade_constraints(
    df,
    prediction_rows: list[dict],
    horizon_days: int,
    constraints: BacktestConstraints,
) -> dict:
    per_symbol, position_lookup = _build_symbol_trade_lookup(df)
    theoretical_returns: list[float] = []
    tradable_returns: list[float] = []
    blocked_limit_up_entries = 0
    blocked_low_liquidity_entries = 0
    blocked_halt_resume_entries = 0
    deferred_limit_down_exits = 0
    unresolved_limit_down_exits = 0
    unresolved_halt_overlap_trades = 0
    trade_records: list[dict] = []

    for item in prediction_rows:
        if int(item["predicted"]) != 1:
            continue

        symbol = item.get("symbol") or "__SINGLE__"
        trade_date = item["date"]
        ordered = per_symbol.get(symbol)
        current_pos = position_lookup.get((symbol, trade_date))
        if ordered is None or current_pos is None:
            continue

        entry_pos = current_pos + 1
        exit_pos = current_pos + horizon_days
        if entry_pos >= len(ordered) or exit_pos >= len(ordered):
            continue

        entry_row = ordered.iloc[entry_pos]
        exit_row = ordered.iloc[exit_pos]
        entry_open = float(entry_row["open"])
        exit_close = float(exit_row["close"])
        theoretical_return = exit_close / entry_open - 1
        theoretical_returns.append(theoretical_return)
        trade_record = {
            "symbol": symbol,
            "signal_date": trade_date,
            "entry_date": entry_row["trade_date_str"],
            "exit_date": exit_row["trade_date_str"],
            "theoretical_return": theoretical_return,
            "tradable_return": None,
            "skipped_reason": None,
        }
        trade_record.update(derive_row_stratification(entry_row))

        signal_close = float(ordered.iloc[current_pos]["close"])
        entry_limit_ratio = float(entry_row.get("price_limit_ratio", 0.1))
        if _is_limit_up_entry(signal_close, entry_open, entry_limit_ratio):
            blocked_limit_up_entries += 1
            trade_record["skipped_reason"] = "limit_up_entry"
            trade_records.append(trade_record)
            continue

        if _is_low_liquidity(entry_row.get("amount"), entry_row.get("turnover_rate"), constraints):
            blocked_low_liquidity_entries += 1
            trade_record["skipped_reason"] = "low_liquidity"
            trade_records.append(trade_record)
            continue

        if _is_resumed_after_halt(entry_row.get("calendar_gap_days"), constraints):
            blocked_halt_resume_entries += 1
            trade_record["skipped_reason"] = "halt_resume_entry"
            trade_records.append(trade_record)
            continue

        if any(
            _is_resumed_after_halt(value, constraints)
            for value in ordered.iloc[entry_pos + 1:exit_pos + 1]["calendar_gap_days"].tolist()
        ):
            unresolved_halt_overlap_trades += 1
            trade_record["skipped_reason"] = "halt_overlap"
            trade_records.append(trade_record)
            continue

        final_exit_row = exit_row
        final_exit_pos = exit_pos
        while True:
            prev_close = float(ordered.iloc[final_exit_pos - 1]["close"]) if final_exit_pos > 0 else None
            limit_ratio = float(final_exit_row.get("price_limit_ratio", 0.1))
            if not _is_limit_down_exit(prev_close, final_exit_row.get("close"), limit_ratio):
                break
            next_exit_pos = final_exit_pos + 1
            if (
                next_exit_pos >= len(ordered)
                or next_exit_pos > exit_pos + constraints.max_exit_extension_days
            ):
                unresolved_limit_down_exits += 1
                final_exit_row = None
                break
            deferred_limit_down_exits += 1
            final_exit_pos = next_exit_pos
            final_exit_row = ordered.iloc[final_exit_pos]

        if final_exit_row is None:
            trade_record["skipped_reason"] = "limit_down_exit_unresolved"
            trade_records.append(trade_record)
            continue

        trade_record["exit_date"] = final_exit_row["trade_date_str"]
        trade_record["tradable_return"] = float(final_exit_row["close"]) / entry_open - 1
        tradable_returns.append(trade_record["tradable_return"])
        trade_records.append(trade_record)

    theoretical = _summarize_trade_returns(theoretical_returns)
    tradable = _summarize_trade_returns(tradable_returns)
    tradable.update({
        "blocked_limit_up_entries": blocked_limit_up_entries,
        "blocked_low_liquidity_entries": blocked_low_liquidity_entries,
        "blocked_halt_resume_entries": blocked_halt_resume_entries,
        "deferred_limit_down_exits": deferred_limit_down_exits,
        "unresolved_limit_down_exits": unresolved_limit_down_exits,
        "unresolved_halt_overlap_trades": unresolved_halt_overlap_trades,
        "skipped_trades": (
            blocked_limit_up_entries
            + blocked_low_liquidity_entries
            + unresolved_limit_down_exits
            + blocked_halt_resume_entries
            + unresolved_halt_overlap_trades
        ),
        "constraints": asdict(constraints),
    })
    return {
        "theoretical_long_only": theoretical,
        "tradable_long_only": tradable,
        "trade_stratification": summarize_trade_stratification(trade_records),
    }


def _run_cv(X, y, dates, n_folds, min_train, labels=None):
    """Core expanding-window CV logic. Returns folds + aggregate."""
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    n = len(X)
    test_size = (n - min_train) // n_folds
    if test_size < 10:
        n_folds = max(1, (n - min_train) // 10)
        test_size = (n - min_train) // n_folds

    folds = []
    all_preds = []
    all_true = []
    all_dates = []
    all_labels = []
    all_positions = []

    for fold in range(n_folds):
        train_end = min_train + fold * test_size
        test_end = train_end + test_size if fold < n_folds - 1 else n

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[train_end:test_end], y[train_end:test_end]

        model = XGBClassifier(
            max_depth=4, n_estimators=300, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X_tr, y_tr, verbose=False)

        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        baseline = max(y_te.mean(), 1 - y_te.mean())

        folds.append({
            "fold": fold + 1,
            "train_size": int(train_end),
            "test_size": int(test_end - train_end),
            "test_start": dates[train_end],
            "test_end": dates[test_end - 1],
            "accuracy": round(acc, 4),
            "baseline": round(baseline, 4),
            "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_te, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_te, y_pred, zero_division=0), 4),
        })

        for i in range(len(y_te)):
            all_preds.append(int(y_pred[i]))
            all_true.append(int(y_te[i]))
            all_dates.append(dates[train_end + i])
            all_positions.append(train_end + i)
            if labels is not None:
                all_labels.append(labels[train_end + i])

    all_true_arr = np.array(all_true)
    all_preds_arr = np.array(all_preds)

    return folds, all_preds, all_true, all_dates, all_labels, all_positions, all_true_arr, all_preds_arr


def run_backtest(
    symbol: str,
    horizon: str = "t1",
    n_folds: int = 5,
    min_train: int = 200,
    constraints: BacktestConstraints | None = None,
) -> dict:
    """Expanding-window CV for a single ticker. Returns per-fold and aggregate metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    constraints = constraints or BacktestConstraints()
    target_col = f"target_{horizon}"

    df = build_features(symbol)
    if df.empty:
        return {"error": f"No data for {symbol}"}

    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    n = len(df)

    if n < min_train + 20:
        return {"error": f"Too few rows ({n}) for backtest"}

    X = df[FEATURE_COLS].values
    y = df[target_col].values
    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()

    folds, all_preds, all_true, all_dates, _, all_positions, all_true_arr, all_preds_arr = _run_cv(
        X, y, dates, n_folds, min_train
    )

    overall_acc = accuracy_score(all_true_arr, all_preds_arr)
    overall_baseline = max(all_true_arr.mean(), 1 - all_true_arr.mean())

    prediction_rows = [
        {
            "date": d,
            "predicted": p,
            "actual": a,
        }
        for d, p, a in zip(all_dates, all_preds, all_true)
    ]
    trading_metrics = _evaluate_trade_constraints(df, prediction_rows, _extract_horizon_days(horizon), constraints)

    result = {
        "symbol": symbol,
        "horizon": horizon,
        "n_folds": len(folds),
        "total_predictions": len(all_true),
        "overall_accuracy": round(overall_acc, 4),
        "overall_baseline": round(overall_baseline, 4),
        "overall_precision": round(precision_score(all_true_arr, all_preds_arr, zero_division=0), 4),
        "overall_recall": round(recall_score(all_true_arr, all_preds_arr, zero_division=0), 4),
        "overall_f1": round(f1_score(all_true_arr, all_preds_arr, zero_division=0), 4),
        "prediction_stratification": summarize_prediction_stratification(df, all_positions, all_true_arr, all_preds_arr),
        "folds": folds,
        "daily_predictions": prediction_rows,
    }
    result.update(trading_metrics)

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / f"{symbol}_{horizon}_backtest.json"
    out_path.write_text(json.dumps(result, indent=2))

    return result


def run_backtest_unified(horizon: str = "t1", n_folds: int = 5, min_train: int = 800,
                         symbols: list[str] | None = None,
                         constraints: BacktestConstraints | None = None) -> dict:
    """Expanding-window CV on combined multi-ticker data."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    constraints = constraints or BacktestConstraints()
    target_col = f"target_{horizon}"

    df = build_features_multi(symbols)
    if df.empty:
        return {"error": "No combined data"}

    # Sort by date (mixing tickers chronologically)
    df = df.sort_values("trade_date").reset_index(drop=True)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    n = len(df)

    if n < min_train + 50:
        return {"error": f"Too few rows ({n}) for unified backtest"}

    X = df[FEATURE_COLS].values
    y = df[target_col].values
    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()
    syms = df["symbol"].tolist()

    folds, all_preds, all_true, all_dates, all_labels, all_positions, all_true_arr, all_preds_arr = _run_cv(
        X, y, dates, n_folds, min_train, labels=syms
    )

    overall_acc = accuracy_score(all_true_arr, all_preds_arr)
    overall_baseline = max(all_true_arr.mean(), 1 - all_true_arr.mean())

    # Per-ticker breakdown
    per_ticker = {}
    for i in range(len(all_true)):
        sym = all_labels[i]
        if sym not in per_ticker:
            per_ticker[sym] = {"true": [], "pred": []}
        per_ticker[sym]["true"].append(all_true[i])
        per_ticker[sym]["pred"].append(all_preds[i])

    ticker_metrics = {}
    for sym, data in sorted(per_ticker.items()):
        t = np.array(data["true"])
        p = np.array(data["pred"])
        ticker_metrics[sym] = {
            "n": len(t),
            "accuracy": round(accuracy_score(t, p), 4),
            "baseline": round(max(t.mean(), 1 - t.mean()), 4),
            "precision": round(precision_score(t, p, zero_division=0), 4),
            "recall": round(recall_score(t, p, zero_division=0), 4),
            "f1": round(f1_score(t, p, zero_division=0), 4),
        }

    prediction_rows = [
        {
            "date": d,
            "symbol": s,
            "predicted": p,
            "actual": a,
        }
        for d, s, p, a in zip(all_dates, all_labels, all_preds, all_true)
    ]
    trading_metrics = _evaluate_trade_constraints(df, prediction_rows, _extract_horizon_days(horizon), constraints)

    result = {
        "symbol": "UNIFIED",
        "horizon": horizon,
        "tickers": sorted(set(syms)),
        "n_folds": len(folds),
        "total_predictions": len(all_true),
        "overall_accuracy": round(overall_acc, 4),
        "overall_baseline": round(overall_baseline, 4),
        "overall_precision": round(precision_score(all_true_arr, all_preds_arr, zero_division=0), 4),
        "overall_recall": round(recall_score(all_true_arr, all_preds_arr, zero_division=0), 4),
        "overall_f1": round(f1_score(all_true_arr, all_preds_arr, zero_division=0), 4),
        "prediction_stratification": summarize_prediction_stratification(df, all_positions, all_true_arr, all_preds_arr),
        "per_ticker": ticker_metrics,
        "folds": folds,
    }
    result.update(trading_metrics)

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / f"UNIFIED_{horizon}_backtest.json"
    out_path.write_text(json.dumps(result, indent=2))

    return result
