"""Prediction API endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "models"


@router.get("/{symbol}")
def get_prediction(symbol: str, horizon: str = Query("t1", pattern="^t[135]$")):
    """Get direction prediction for a symbol."""
    from backend.ml.model import predict

    result = predict(symbol.upper(), horizon)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{symbol}/backtest")
def get_backtest(
    symbol: str,
    horizon: str = Query("t1", pattern="^t[135]$"),
    min_entry_amount_k: float | None = Query(None, ge=0),
    min_entry_turnover_rate_pct: float | None = Query(None, ge=0),
    max_exit_extension_days: int | None = Query(None, ge=0, le=20),
    halt_gap_days: int | None = Query(None, ge=3, le=60),
):
    """Get backtest results for a symbol."""
    sym = symbol.upper()
    from backend.ml.backtest import BacktestConstraints, run_backtest

    constraints = BacktestConstraints(
        min_entry_amount_k=20_000.0 if min_entry_amount_k is None else min_entry_amount_k,
        min_entry_turnover_rate_pct=0.5 if min_entry_turnover_rate_pct is None else min_entry_turnover_rate_pct,
        max_exit_extension_days=5 if max_exit_extension_days is None else max_exit_extension_days,
        halt_gap_days=10 if halt_gap_days is None else halt_gap_days,
    )

    if any(
        value is not None
        for value in (
            min_entry_amount_k,
            min_entry_turnover_rate_pct,
            max_exit_extension_days,
            halt_gap_days,
        )
    ):
        result = run_backtest(sym, horizon, constraints=constraints)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    path = MODELS_DIR / f"{sym}_{horizon}_backtest.json"
    if path.exists():
        return json.loads(path.read_text())

    result = run_backtest(sym, horizon, constraints=constraints)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{symbol}/forecast")
def get_forecast(symbol: str, window: int = Query(7, ge=3, le=60)):
    """Generate forecast based on recent news window (7d or 30d)."""
    from backend.ml.inference import generate_forecast

    result = generate_forecast(symbol.upper(), window)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{symbol}/similar-days")
def get_similar_days(symbol: str, date: str = Query(...), top_k: int = Query(10, ge=1, le=30)):
    """Find historically similar trading days based on ML features."""
    from backend.ml.similar import find_similar_days

    result = find_similar_days(symbol.upper(), date, top_k)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
