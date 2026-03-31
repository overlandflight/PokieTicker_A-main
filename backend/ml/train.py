"""CLI entry point: python -m backend.ml.train [--symbol SYM] [--backtest] [--lstm]"""

import argparse
import time

from backend.database import get_conn
from backend.ml.model import train
from backend.ml.backtest import BacktestConstraints, run_backtest

HORIZONS = ["t1", "t5"]

# Best LSTM configs per ticker (from experiments)
LSTM_CONFIGS = {
    "TSLA": {"target_col": "target_t3", "seq_len": 10, "exclude_neutral": False},
    # Add more tickers here as LSTM proves beneficial
}


def _format_bucket_counts(section: dict | None) -> str:
    if not section:
        return "-"
    parts = []
    for bucket, item in section.items():
        parts.append(f"{bucket}:{item.get('count', 0)}")
    return ", ".join(parts) if parts else "-"


def get_symbols() -> list[str]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT symbol FROM tickers WHERE last_ohlc_fetch IS NOT NULL ORDER BY symbol"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [r["symbol"] for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--symbol", type=str, help="Train only this ticker")
    parser.add_argument("--backtest", action="store_true", help="Run backtest after training")
    parser.add_argument("--lstm", action="store_true", help="Also train LSTM for configured tickers")
    parser.add_argument("--min-entry-amount-k", type=float, default=20_000.0, help="Minimum tradable entry amount in thousand yuan")
    parser.add_argument("--min-entry-turnover-rate-pct", type=float, default=0.5, help="Minimum tradable entry turnover rate percent")
    parser.add_argument("--max-exit-extension-days", type=int, default=5, help="Max days to defer exit when limit down blocks selling")
    parser.add_argument("--halt-gap-days", type=int, default=10, help="Calendar gap threshold treated as halt/resume")
    args = parser.parse_args()

    backtest_constraints = BacktestConstraints(
        min_entry_amount_k=args.min_entry_amount_k,
        min_entry_turnover_rate_pct=args.min_entry_turnover_rate_pct,
        max_exit_extension_days=args.max_exit_extension_days,
        halt_gap_days=args.halt_gap_days,
    )

    symbols = [args.symbol.upper()] if args.symbol else get_symbols()
    print(f"Training for {len(symbols)} ticker(s): {', '.join(symbols)}")

    t0 = time.time()
    for sym in symbols:
        for h in HORIZONS:
            result = train(sym, h)
            if "error" in result:
                print(f"  {sym}/{h}: {result['error']}")
            else:
                print(f"  {sym}/{h}: acc={result['accuracy']:.1%} baseline={result['baseline']:.1%} "
                      f"(train={result['train_size']}, test={result['test_size']})")
                strat = result.get("test_stratification", {})
                print(
                    "    strata: "
                    f"board[{_format_bucket_counts(strat.get('board'))}] "
                    f"cap[{_format_bucket_counts(strat.get('cap'))}] "
                    f"liquidity[{_format_bucket_counts(strat.get('liquidity'))}]"
                )

            if args.backtest and "error" not in result:
                bt = run_backtest(sym, h, constraints=backtest_constraints)
                if "error" in bt:
                    print(f"    backtest: {bt['error']}")
                else:
                    theoretical = bt.get("theoretical_long_only", {})
                    tradable = bt.get("tradable_long_only", {})
                    print(f"    backtest: {bt['n_folds']} folds, "
                          f"acc={bt['overall_accuracy']:.1%} baseline={bt['overall_baseline']:.1%}")
                    if theoretical.get("trades"):
                        print(
                            "      theoretical: "
                            f"trades={theoretical['trades']} "
                            f"avg={theoretical['avg_return_pct']:+.2f}% "
                            f"total={theoretical['total_return_pct']:+.2f}%"
                        )
                    if tradable.get("trades") is not None:
                        print(
                            "      tradable: "
                            f"trades={tradable['trades']} "
                            f"avg={tradable.get('avg_return_pct') if tradable.get('avg_return_pct') is not None else 0:+.2f}% "
                            f"total={tradable.get('total_return_pct') if tradable.get('total_return_pct') is not None else 0:+.2f}% "
                            f"skipped={tradable.get('skipped_trades', 0)}"
                        )
                    pred_strata = bt.get("prediction_stratification", {})
                    print(
                        "      pred-strata: "
                        f"board[{_format_bucket_counts(pred_strata.get('board'))}] "
                        f"cap[{_format_bucket_counts(pred_strata.get('cap'))}] "
                        f"liquidity[{_format_bucket_counts(pred_strata.get('liquidity'))}]"
                    )

        # LSTM training for configured tickers
        if args.lstm and sym in LSTM_CONFIGS:
            from backend.ml.lstm_model import train_and_save_lstm
            cfg = LSTM_CONFIGS[sym]
            print(f"  {sym}/LSTM: training {cfg['target_col']} seq={cfg['seq_len']}...")
            result = train_and_save_lstm(sym, **cfg, epochs=50)
            if "error" in result:
                print(f"    LSTM: {result['error']}")
            else:
                print(f"    LSTM: saved ({result['train_size']} sequences)")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
