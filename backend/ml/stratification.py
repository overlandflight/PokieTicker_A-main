"""Shared A-share stratification helpers for training and backtest reporting."""

from __future__ import annotations

import math

BOARD_BUCKET_LABELS = {
    0.0: "main_board",
    1.0: "chinext",
    2.0: "star_market",
    3.0: "beijing",
}

CAP_BUCKET_LABELS = {
    0.0: "small_cap",
    1.0: "mid_cap",
    2.0: "large_cap",
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isnan(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _ratio(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(value / total, 4)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _compound_return_pct(returns: list[float]) -> float | None:
    if not returns:
        return None
    compounded = 1.0
    for trade_return in returns:
        compounded *= 1 + trade_return
    return round((compounded - 1) * 100, 2)


def _get_row(source, position: int):
    if hasattr(source, "iloc"):
        return source.iloc[position]
    return source[position]


def _iter_rows(source, positions=None):
    if source is None:
        return
    positions = range(len(source)) if positions is None else positions
    for position in positions:
        yield position, _get_row(source, position)


def _board_bucket_label(value) -> str:
    return BOARD_BUCKET_LABELS.get(_safe_float(value), "unknown")


def _cap_bucket_label(value) -> str:
    return CAP_BUCKET_LABELS.get(_safe_float(value), "unknown")


def _liquidity_bucket(amount, turnover_rate) -> str:
    amount_value = _safe_float(amount, 0.0)
    turnover_value = _safe_float(turnover_rate, 0.0)
    if amount_value < 20_000 or turnover_value < 0.5:
        return "illiquid"
    if amount_value < 100_000 or turnover_value < 2.0:
        return "mid_liquidity"
    return "high_liquidity"


def derive_row_stratification(row) -> dict[str, str]:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: row[key] if key in row else default
    return {
        "board": _board_bucket_label(getter("board_bucket_id")),
        "cap": _cap_bucket_label(getter("cap_bucket_id")),
        "liquidity": _liquidity_bucket(getter("amount"), getter("turnover_rate")),
    }


def summarize_sample_stratification(source, positions=None) -> dict[str, dict[str, dict[str, float | int]]]:
    buckets = {"board": {}, "cap": {}, "liquidity": {}}
    total = 0
    for _, row in _iter_rows(source, positions):
        total += 1
        strata = derive_row_stratification(row)
        for dimension, bucket in strata.items():
            buckets[dimension][bucket] = buckets[dimension].get(bucket, 0) + 1

    summary = {}
    for dimension, counts in buckets.items():
        summary[dimension] = {
            bucket: {"count": count, "ratio": _ratio(count, total)}
            for bucket, count in sorted(counts.items())
        }
    return summary


def summarize_prediction_stratification(source, positions, y_true, y_pred) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped = {"board": {}, "cap": {}, "liquidity": {}}
    positions_list = list(positions)

    for position, actual, predicted in zip(positions_list, y_true, y_pred):
        row = _get_row(source, position)
        strata = derive_row_stratification(row)
        actual_int = int(actual)
        predicted_int = int(predicted)
        for dimension, bucket in strata.items():
            item = grouped[dimension].setdefault(
                bucket,
                {"actual": [], "predicted": []},
            )
            item["actual"].append(actual_int)
            item["predicted"].append(predicted_int)

    summary = {}
    for dimension, buckets in grouped.items():
        total = sum(len(item["actual"]) for item in buckets.values())
        summary[dimension] = {}
        for bucket, item in sorted(buckets.items()):
            actual_values = item["actual"]
            predicted_values = item["predicted"]
            count = len(actual_values)
            actual_up_ratio = sum(actual_values) / count if count else 0.0
            predicted_up_ratio = sum(predicted_values) / count if count else 0.0
            accuracy = sum(
                1 for actual_value, predicted_value in zip(actual_values, predicted_values)
                if actual_value == predicted_value
            ) / count if count else 0.0
            baseline = max(actual_up_ratio, 1 - actual_up_ratio) if count else 0.0
            summary[dimension][bucket] = {
                "count": count,
                "ratio": _ratio(count, total),
                "accuracy": round(accuracy, 4),
                "baseline": round(baseline, 4),
                "actual_up_ratio": round(actual_up_ratio, 4),
                "predicted_up_ratio": round(predicted_up_ratio, 4),
            }
    return summary


def summarize_trade_stratification(trade_records: list[dict]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    grouped = {"board": {}, "cap": {}, "liquidity": {}}

    for record in trade_records:
        for dimension in grouped:
            bucket = str(record.get(dimension, "unknown"))
            grouped[dimension].setdefault(bucket, []).append(record)

    summary = {}
    for dimension, buckets in grouped.items():
        total = sum(len(records) for records in buckets.values())
        summary[dimension] = {}
        for bucket, records in sorted(buckets.items()):
            theoretical_returns = [float(r["theoretical_return"]) for r in records if r.get("theoretical_return") is not None]
            tradable_returns = [float(r["tradable_return"]) for r in records if r.get("tradable_return") is not None]
            skipped_reason_counts: dict[str, int] = {}
            for record in records:
                skipped_reason = record.get("skipped_reason")
                if not skipped_reason:
                    continue
                reason = str(skipped_reason)
                skipped_reason_counts[reason] = skipped_reason_counts.get(reason, 0) + 1
            summary[dimension][bucket] = {
                "count": len(records),
                "ratio": _ratio(len(records), total),
                "tradable_trades": len(tradable_returns),
                "skipped_trades": sum(1 for r in records if r.get("tradable_return") is None),
                "tradable_ratio": _ratio(len(tradable_returns), len(records)),
                "skipped_ratio": _ratio(sum(1 for r in records if r.get("tradable_return") is None), len(records)),
                "avg_theoretical_return_pct": round((_mean(theoretical_returns) or 0.0) * 100, 2) if theoretical_returns else None,
                "avg_tradable_return_pct": round((_mean(tradable_returns) or 0.0) * 100, 2) if tradable_returns else None,
                "tradable_win_rate": round(sum(1 for value in tradable_returns if value > 0) / len(tradable_returns), 4) if tradable_returns else None,
                "total_tradable_return_pct": _compound_return_pct(tradable_returns),
                "skipped_reason_counts": skipped_reason_counts,
            }
    return summary
