#!/usr/bin/env python3
"""Aggregate single-domain, pooled, and CORAL RX comparisons across groups.

Creates a macro-average cross-domain heatmap and a paired per-group model
comparison.  Every controlled group has equal weight; raw pixels are never
pooled across groups.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from domain_1_aligned_rx import DOMAIN_FACTORS, parse_domain_factor


MODEL_ORDER = ("single_weather_rx", "pooled_rx", "coral_aligned_rx")
MODEL_LABELS = {
    "single_weather_rx": "Single weather\n(cross-weather only)",
    "pooled_rx": "Pooled RX",
    "coral_aligned_rx": "CORAL-aligned RX",
}
MODEL_COLORS = {"single_weather_rx": "#d95f02", "pooled_rx": "#7570b3", "coral_aligned_rx": "#1b9e77"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_metrics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """One equally weighted FAR summary for each group and model."""
    by_group_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        model = row.get("model")
        if model not in MODEL_ORDER:
            continue
        # A single-weather model is meaningful for domain shift only off diagonal.
        if model == "single_weather_rx" and row.get("train_domain", row.get("train_weather")) == row.get("test_domain", row.get("test_weather")):
            continue
        by_group_model[(row["group"], model)].append(float(row["false_alarm_rate"]))
    metrics = []
    for (group, model), values in sorted(by_group_model.items()):
        metrics.append({
            "group": group,
            "model": model,
            "mean_false_alarm_rate": float(np.mean(values)),
            "evaluations": len(values),
        })
    return metrics


def macro_heatmap(rows: list[dict[str, str]], factor: str) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Average each model/test-domain cell across groups, not raw pixels."""
    single = [row for row in rows if row.get("model") == "single_weather_rx"]
    pooled = [row for row in rows if row.get("model") == "pooled_rx"]
    aligned = [row for row in rows if row.get("model") == "coral_aligned_rx"]
    train_weather = sorted({row.get("train_domain", row.get("train_weather", "")) for row in single})
    test_weather = sorted(
        {row.get("test_domain", row.get("test_weather", "")) for row in single}
        | {row.get("test_domain", row.get("test_weather", "")) for row in pooled}
        | {row.get("test_domain", row.get("test_weather", "")) for row in aligned}
    )
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in single:
        values[(row.get("train_domain", row.get("train_weather", "")), row.get("test_domain", row.get("test_weather", "")))].append(float(row["false_alarm_rate"]))
    for row in aligned:
        values[("all (CORAL)", row.get("test_domain", row.get("test_weather", "")))].append(float(row["false_alarm_rate"]))
    for row in pooled:
        values[("all (Pooled)", row.get("test_domain", row.get("test_weather", "")))].append(float(row["false_alarm_rate"]))
    labels = [f"{factor} {weather}" for weather in train_weather] + ["all (Pooled)", "all (CORAL)"]
    keys = train_weather + ["all (Pooled)", "all (CORAL)"]
    matrix = np.full((len(keys), len(test_weather)), np.nan)
    counts = np.zeros(matrix.shape, dtype=int)
    for row, train in enumerate(keys):
        for col, test in enumerate(test_weather):
            cell = values[(train, test)]
            if cell:
                matrix[row, col] = np.mean(cell)
                counts[row, col] = len(cell)
    return labels, test_weather, matrix, counts


def plot_macro_heatmap(labels: list[str], weather: list[str], matrix: np.ndarray, counts: np.ndarray, path: Path, factor: str) -> None:
    masked = np.ma.masked_invalid(matrix)
    maximum = max(0.05, float(np.nanmax(matrix)))
    cmap = plt.colormaps["YlOrRd"].copy()
    cmap.set_bad("#eeeeee")
    fig, axis = plt.subplots(figsize=(max(7, len(weather) * 1.55), max(5, len(labels) * 1.05)), constrained_layout=True)
    image = axis.imshow(masked, vmin=0, vmax=maximum, cmap=cmap, aspect="auto")
    axis.set(
        title=f"Macro-average cross-{factor} normal-road false alarms",
        xlabel=f"Test {factor}",
        ylabel=f"Train {factor}",
        xticks=range(len(weather)),
        yticks=range(len(labels)),
    )
    axis.set_xticklabels(weather)
    axis.set_yticklabels(labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text = "N/A" if not np.isfinite(matrix[row, col]) else f"{matrix[row, col]:.1%}\n(n={counts[row, col]})"
            axis.text(col, row, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Macro-average false-alarm rate at train p99")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_group_distribution(metrics: list[dict[str, object]], path: Path, factor: str) -> None:
    by_model = {model: {row["group"]: float(row["mean_false_alarm_rate"]) for row in metrics if row["model"] == model} for model in MODEL_ORDER}
    common_groups = sorted(set.intersection(*(set(values) for values in by_model.values())))
    values = [[by_model[model][group] for group in common_groups] for model in MODEL_ORDER]
    fig, axis = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    labels = [f"Single {factor}\n(cross-{factor} only)", MODEL_LABELS["pooled_rx"], MODEL_LABELS["coral_aligned_rx"]]
    box = axis.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.28)
    rng = np.random.default_rng(42)
    positions = np.arange(1, len(MODEL_ORDER) + 1)
    for group in common_groups:
        line = [by_model[model][group] for model in MODEL_ORDER]
        axis.plot(positions, line, color="#777777", alpha=0.22, linewidth=0.8, zorder=1)
    for position, model, model_values in zip(positions, MODEL_ORDER, values):
        jitter = rng.uniform(-0.07, 0.07, len(model_values))
        axis.scatter(np.full(len(model_values), position) + jitter, model_values, color=MODEL_COLORS[model], s=22, alpha=0.80, zorder=2)
        axis.scatter(position, np.mean(model_values), marker="D", color="black", s=40, zorder=3)
    axis.axhline(0.01, color="tab:red", linestyle="--", label="nominal 1% train p99")
    axis.set(title=f"Per-group cross-{factor} false alarms (n={len(common_groups)} common groups)", ylabel="Mean false-alarm rate per group", ylim=(0, 1.02))
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summary_rows(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for model in MODEL_ORDER:
        values = np.asarray([float(row["mean_false_alarm_rate"]) for row in metrics if row["model"] == model])
        output.append({
            "model": model,
            "groups": len(values),
            "macro_mean_false_alarm_rate": float(values.mean()),
            "median_false_alarm_rate": float(np.median(values)),
            "q25_false_alarm_rate": float(np.percentile(values, 25)),
            "q75_false_alarm_rate": float(np.percentile(values, 75)),
            "worst_group_false_alarm_rate": float(values.max()),
        })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--domain-factor", type=parse_domain_factor, default=None, metavar="FACTOR", help="Run one factor; omit to run all factors.")
    parser.add_argument("--all-domain-factors", action="store_true", help="Run all four factors sequentially.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to a factor-specific output folder.")
    return parser.parse_args()


def run_factor(args: argparse.Namespace) -> Path:
    if args.results_dir is None:
        args.results_dir = Path("domain_2_single_domain_rx_results") / args.domain_factor
    if args.output_dir is None:
        args.output_dir = Path("domain_3_overall_rx_comparisons") / args.domain_factor
    rows = read_csv(args.results_dir / "cross_weather_results.csv")
    metrics = group_metrics(rows)
    labels, weather, matrix, counts = macro_heatmap(rows, args.domain_factor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_macro_heatmap(labels, weather, matrix, counts, args.output_dir / "macro_average_cross_fpr_heatmap.png", args.domain_factor)
    plot_group_distribution(metrics, args.output_dir / "per_group_model_comparison.png", args.domain_factor)
    write_csv(args.output_dir / "per_group_model_metrics.csv", metrics)
    write_csv(args.output_dir / "overall_model_summary.csv", summary_rows(metrics))
    print(f"Wrote aggregate figures and CSV summaries to {args.output_dir.resolve()}.")
    return args.output_dir


def main() -> None:
    args = parse_args()
    factors = DOMAIN_FACTORS if args.all_domain_factors or args.domain_factor is None else (args.domain_factor,)
    output_dirs: list[Path] = []
    for factor in factors:
        run_args = argparse.Namespace(**vars(args))
        run_args.domain_factor = factor
        if len(factors) > 1 and args.output_dir is not None:
            run_args.output_dir = args.output_dir / factor
        print(f"\n=== Aggregating domain factor: {factor} ===")
        output_dirs.append(run_factor(run_args))
    print("\n=== Saved overall comparison folders ===")
    for output_dir in output_dirs:
        print(output_dir.resolve())


if __name__ == "__main__":
    main()
