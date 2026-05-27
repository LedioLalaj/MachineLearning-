"""Step 3 — Benchmark a trained model and log the iteration.

Run:  python 03_benchmark.py --tag v1

What it does:
  1. Runs evaluation on multiple map seeds.
  2. Logs results to benchmarks/<tag>.json.
  3. Saves diagnostic plots for analysis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from drive2win.benchmark import run_benchmark
from drive2win import viz


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tag", required=True,
                    help="Iteration tag, e.g. v1, v2, v3.")

    ap.add_argument("--weights", default=None,
                    help="Path to weights file. Defaults to nav_<tag>.npz.")

    ap.add_argument("--module", default=None,
                    help="Optional model module override.")

    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Map seeds to evaluate on. If not set, uses fixed test suite.")

    ap.add_argument("--runs", type=int, default=5,
                    help="Number of runs per seed.")

    ap.add_argument("--duration", type=float, default=60.0,
                    help="Max duration per run.")

    ap.add_argument("--data", default=None,
                    help="Optional data_<tag>.npz for overlay plots.")

    args = ap.parse_args()

    # ------------------------------------------------------------
    # Default evaluation seeds (fixed benchmark suite)
    # ------------------------------------------------------------
    if args.seeds is None:
        args.seeds = [7, 21, 42, 84, 1337]

    weights = args.weights or f"nav_{args.tag}.npz"
    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)

    all_results = []

    for seed in args.seeds:
        print(f"\n=== Evaluating on seed {seed} ===")

        result = run_benchmark(
            weights=weights,
            runs=args.runs,
            seed=seed,
            duration=args.duration,
            module=args.module,
        )

        all_results.append({"seed": seed, **result})

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Iteration: {args.tag} | Weights: {weights}")

    for r in all_results:
        s = r["summary"]
        print(
            f"seed {r['seed']:>5} | "
            f"completion={int(s['completion_rate'] * s['n_runs'])}/{s['n_runs']} | "
            f"median_lap={s['median_lap_time']:.2f}s | "
            f"crashes={s['mean_crashes']:.2f} | "
            f"max_cp={s['max_checkpoints']}"
        )

    print("=" * 60)

    # ------------------------------------------------------------
    # Save JSON log
    # ------------------------------------------------------------
    log_path = out_dir / f"{args.tag}.json"

    log = {
        "tag": args.tag,
        "weights": weights,
        "module": args.module,
        "runs_per_seed": args.runs,
        "duration_s": args.duration,
        "seeds": [
            {
                "seed": r["seed"],
                "summary": r["summary"],
                "runs": r["runs"],
            }
            for r in all_results
        ],
    }

    log_path.write_text(json.dumps(log, indent=2, default=float))
    print(f"\nSaved log → {log_path}")

    # ------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------
    flat_runs = [run for r in all_results for run in r["runs"]]

    viz.plot_multi_run_paths(
        flat_runs,
        out=str(out_dir / f"{args.tag}_paths.png"),
        title=f"All runs — {args.tag}"
    )

    viz.plot_checkpoint_progress(
        flat_runs,
        out=str(out_dir / f"{args.tag}_progress.png")
    )

    # ------------------------------------------------------------
    # Optional overlay (train vs test trajectories)
    # ------------------------------------------------------------
    if args.data:
        d = np.load(args.data, allow_pickle=False)
        train_xz = d["positions"] if "positions" in d.files else None
        first_track = flat_runs[0].get("track") or []

        viz.plot_path_overlay(
            train_xz,
            first_track,
            out=str(out_dir / f"{args.tag}_overlay.png"),
            title=f"{args.tag} — train vs test"
        )


if __name__ == "__main__":
    main()