#!/usr/bin/env python3
import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

LOSS_KEYS = [
    "Loss Sum",
    "Delta Noise Loss",
    "Beta Loss",
    "KL Loss",
    "Gamma Loss",
]


def parse_log(log_path: str) -> Tuple[List[dict], List[str]]:
    epoch_step_re = re.compile(r"Epoch\s+(\d+)\s*\|\s*Step\s+(\d+)")
    scalar_re = re.compile(r"^\s*([A-Za-z0-9_ ]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")

    records: List[dict] = []
    metric_order: List[str] = []

    current_epoch: Optional[int] = None
    current_step: Optional[int] = None

    pending_losses: Dict[str, float] = {}
    pending_metrics: Dict[str, float] = {}
    in_metric_block = False

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            epoch_match = epoch_step_re.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                current_step = int(epoch_match.group(2))

            scalar_match = scalar_re.match(line)
            if scalar_match:
                key = scalar_match.group(1).strip()
                value = float(scalar_match.group(2))

                if key in LOSS_KEYS:
                    pending_losses[key] = value
                    continue

                if in_metric_block:
                    pending_metrics[key] = value
                    if key not in metric_order:
                        metric_order.append(key)
                    continue

            if line == "Train Metrics:":
                in_metric_block = True
                continue

            if in_metric_block and (not line or line.startswith("[")):
                if current_epoch is not None and current_step is not None and pending_losses and pending_metrics:
                    records.append(
                        {
                            "epoch": current_epoch,
                            "step": current_step,
                            "losses": dict(pending_losses),
                            "metrics": dict(pending_metrics),
                        }
                    )
                pending_losses = {}
                pending_metrics = {}
                in_metric_block = False

        if in_metric_block and current_epoch is not None and current_step is not None and pending_losses and pending_metrics:
            records.append(
                {
                    "epoch": current_epoch,
                    "step": current_step,
                    "losses": dict(pending_losses),
                    "metrics": dict(pending_metrics),
                }
            )

    return records, metric_order


def aggregate_by_epoch(records: List[dict], metric_keys: List[str]) -> List[dict]:
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["epoch"]].append(rec)

    out_rows: List[dict] = []
    for epoch in sorted(grouped.keys()):
        recs = grouped[epoch]
        row = {"epoch": epoch, "n_samples": len(recs)}

        for loss_key in LOSS_KEYS:
            vals = [r["losses"].get(loss_key) for r in recs if loss_key in r["losses"]]
            row[loss_key] = sum(vals) / len(vals) if vals else float("nan")

        for metric_key in metric_keys[:5]:
            vals = [r["metrics"].get(metric_key) for r in recs if metric_key in r["metrics"]]
            row[metric_key] = sum(vals) / len(vals) if vals else float("nan")

        out_rows.append(row)

    return out_rows


def save_csv(rows: List[dict], metric_keys: List[str], out_csv: str) -> None:
    fieldnames = ["epoch", "n_samples"] + LOSS_KEYS + metric_keys[:5]
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_curves(rows: List[dict], metric_keys: List[str], out_png: str) -> None:
    epochs = [r["epoch"] for r in rows]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True)

    for col_idx, loss_key in enumerate(LOSS_KEYS):
        ax = axes[0, col_idx]
        vals = [r[loss_key] for r in rows]
        ax.plot(epochs, vals, marker="o", linewidth=1.4, markersize=3)
        ax.set_title(loss_key)
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    for col_idx, metric_key in enumerate(metric_keys[:5]):
        ax = axes[1, col_idx]
        vals = [r[metric_key] for r in rows]
        ax.plot(epochs, vals, marker="o", linewidth=1.4, markersize=3)
        ax.set_title(metric_key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse CVDM slurm train log and plot 5 losses + 5 train metrics by epoch.")
    parser.add_argument("--log", required=True, help="Path to slurm output log")
    parser.add_argument("--out-dir", default=None, help="Output directory for CSV and PNG (default: log directory)")
    args = parser.parse_args()

    log_path = os.path.abspath(args.log)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(log_path)
    os.makedirs(out_dir, exist_ok=True)

    records, metric_order = parse_log(log_path)
    if not records:
        raise RuntimeError("No train records found. Check that the log contains 'Train Metrics:' blocks and loss lines.")

    if len(metric_order) < 5:
        raise RuntimeError(f"Expected at least 5 train metrics, found {len(metric_order)}: {metric_order}")

    rows = aggregate_by_epoch(records, metric_order)

    log_base = os.path.splitext(os.path.basename(log_path))[0]
    out_csv = os.path.join(out_dir, f"{log_base}_epoch_train_curves.csv")
    out_png = os.path.join(out_dir, f"{log_base}_epoch_train_curves.png")

    save_csv(rows, metric_order, out_csv)
    plot_curves(rows, metric_order, out_png)

    print(f"Parsed records: {len(records)}")
    print(f"Epochs: {len(rows)} ({rows[0]['epoch']}..{rows[-1]['epoch']})")
    print(f"Loss keys: {LOSS_KEYS}")
    print(f"Metric keys used: {metric_order[:5]}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
