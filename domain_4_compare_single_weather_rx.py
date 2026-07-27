#!/usr/bin/env python3
"""Compare cross-weather false alarms of single-, pooled-, and aligned-RX.

The experiment uses only normal Road ROI pixels.  For each controlled group,
an RX background trained on one source weather is evaluated on every held-out
target weather.  Its source-train p99 score is kept as the threshold, so a
large off-diagonal false-alarm rate directly measures weather domain shift.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from domain_1_aligned_rx import (
    capped_rows,
    fit_coral,
    fit_rx,
    load_manifest,
    road_features,
    split_per_weather,
)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_summary(scores: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "false_alarm_rate": float(np.mean(scores > threshold)),
        "score_mean": float(scores.mean()),
        "score_p95": float(np.percentile(scores, 95)),
        "score_p99": float(np.percentile(scores, 99)),
    }


def heatmap(matrix: np.ndarray, labels: list[str], title: str, colorbar_label: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(max(5.5, len(labels) * 1.25), max(4.5, len(labels) * 1.1)), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=max(0.05, float(np.nanmax(matrix))), cmap="YlOrRd")
    axis.set(
        title=title,
        xlabel="Test weather",
        ylabel="Train weather",
        xticks=range(len(labels)),
        yticks=range(len(labels)),
    )
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            axis.text(col, row, f"{matrix[row, col]:.1%}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axis, label=colorbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def grouped_bar(summary: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["model"]) for row in summary]
    values = [float(row["mean_cross_weather_false_alarm_rate"]) for row in summary]
    fig, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    bars = axis.bar(labels, values, color=["#d95f02", "#7570b3", "#1b9e77"][: len(labels)])
    axis.set(title="Average cross-weather normal false-alarm rate", ylabel="False-alarm rate")
    axis.axhline(0.01, color="tab:red", linestyle="--", label="nominal 1% source p99")
    axis.legend()
    axis.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=3)
    axis.set_ylim(0, max(0.02, max(values) * 1.20))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_group(group: str, samples: list[object], args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
    selected = [sample for sample in samples if not args.weather_domains or sample.weather in args.weather_domains]
    weather = sorted({sample.weather for sample in selected})
    if args.weather_domains and not args.weather_domains.issubset(weather):
        return [{"group": group, "status": "skipped", "reason": f"missing requested weather domains: {sorted(args.weather_domains - set(weather))}"}]
    if len(weather) < args.min_domains:
        return [{"group": group, "status": "skipped", "reason": f"only {len(weather)} weather domains"}]

    train, test = split_per_weather(selected, args.test_fraction, args.seed)
    if not train:
        return [{"group": group, "status": "skipped", "reason": "each weather domain needs at least two images"}]

    rng = np.random.default_rng(args.seed)
    train_by_weather: dict[str, list[np.ndarray]] = defaultdict(list)
    test_by_weather: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample in train:
        features, _ = road_features(sample)
        train_by_weather[sample.weather].append(capped_rows(features, args.max_pixels_per_image, rng))
    for sample in test:
        features, _ = road_features(sample)
        test_by_weather[sample.weather].append(capped_rows(features, args.max_pixels_per_image, rng))
    train_x = {domain: np.concatenate(train_by_weather[domain]) for domain in weather}
    test_x = {domain: np.concatenate(test_by_weather[domain]) for domain in weather}

    group_dir = output_dir / f"group_{group}"
    group_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "group": group,
        "train": [{"sample": sample.base, "weather": sample.weather} for sample in train],
        "test": [{"sample": sample.base, "weather": sample.weather} for sample in test],
    }
    (group_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    single_matrix = np.zeros((len(weather), len(weather)), dtype=float)
    for source_index, source_weather in enumerate(weather):
        source_train = capped_rows(train_x[source_weather], args.max_background_pixels, rng)
        model = fit_rx(source_train, args.cov_reg)
        threshold = float(np.percentile(model.score(source_train), 99))
        np.savez(
            group_dir / f"single_weather_{source_weather}_rx_background.npz",
            mean=model.mean,
            inv_cov=model.inv_cov,
            train_score_p99=threshold,
        )
        for target_index, target_weather in enumerate(weather):
            scores = model.score(test_x[target_weather])
            summary = score_summary(scores, threshold)
            single_matrix[source_index, target_index] = summary["false_alarm_rate"]
            rows.append({
                "group": group,
                "model": "single_weather_rx",
                "train_weather": source_weather,
                "test_weather": target_weather,
                "threshold_source_train_p99": threshold,
                "train_images": sum(sample.weather == source_weather for sample in train),
                "test_images": sum(sample.weather == target_weather for sample in test),
                "test_road_pixels": len(scores),
                "status": "ok",
                **summary,
            })
    heatmap(
        single_matrix,
        weather,
        f"Group {group}: single-weather RX cross-weather false alarms",
        "False-alarm rate at source train p99",
        group_dir / "single_weather_cross_fpr_heatmap.png",
    )

    # Fair controls: one model trained on all weather data before/after CORAL.
    pooled_train = np.concatenate([train_x[domain] for domain in weather])
    pooled_train = capped_rows(pooled_train, args.max_background_pixels, rng)
    pooled_model = fit_rx(pooled_train, args.cov_reg)
    pooled_threshold = float(np.percentile(pooled_model.score(pooled_train), 99))
    transforms = fit_coral(train_x, args.cov_reg)
    aligned_train = np.concatenate([transforms[domain].apply(train_x[domain]) for domain in weather])
    aligned_background = capped_rows(aligned_train, args.max_background_pixels, rng)
    aligned_model = fit_rx(aligned_background, args.cov_reg)
    aligned_threshold = float(np.percentile(aligned_model.score(aligned_background), 99))
    np.savez(group_dir / "pooled_rx_background.npz", mean=pooled_model.mean, inv_cov=pooled_model.inv_cov, train_score_p99=pooled_threshold)
    np.savez(group_dir / "aligned_rx_background.npz", mean=aligned_model.mean, inv_cov=aligned_model.inv_cov, train_score_p99=aligned_threshold)

    for model_name, model, threshold in (
        ("pooled_rx", pooled_model, pooled_threshold),
        ("coral_aligned_rx", aligned_model, aligned_threshold),
    ):
        for target_weather in weather:
            values = test_x[target_weather] if model_name == "pooled_rx" else transforms[target_weather].apply(test_x[target_weather])
            scores = model.score(values)
            rows.append({
                "group": group,
                "model": model_name,
                "train_weather": "all",
                "test_weather": target_weather,
                "threshold_source_train_p99": threshold,
                "train_images": len(train),
                "test_images": sum(sample.weather == target_weather for sample in test),
                "test_road_pixels": len(scores),
                "status": "ok",
                **score_summary(scores, threshold),
            })

    single_off_diagonal = [single_matrix[row, col] for row in range(len(weather)) for col in range(len(weather)) if row != col]
    summary_rows = [
        {
            "group": group,
            "model": "single_weather_rx",
            "mean_cross_weather_false_alarm_rate": float(np.mean(single_off_diagonal)),
        }
    ]
    for model_name in ("pooled_rx", "coral_aligned_rx"):
        fprs = [float(row["false_alarm_rate"]) for row in rows if row["model"] == model_name]
        summary_rows.append({"group": group, "model": model_name, "mean_cross_weather_false_alarm_rate": float(np.mean(fprs))})
    write_csv(group_dir / "comparison_summary.csv", summary_rows)
    grouped_bar(summary_rows, group_dir / "model_comparison_false_alarm_rate.png")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("HSI_Drive"))
    parser.add_argument("--output-dir", type=Path, default=Path("single_weather_rx_comparison_results"))
    parser.add_argument("--weather-domains", nargs="*", default=None, help="Optional IDs, e.g. 1 2 3 4.")
    parser.add_argument("--min-domains", type=int, default=2)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--cov-reg", type=float, default=1e-4)
    parser.add_argument("--max-pixels-per-image", type=int, default=20_000)
    parser.add_argument("--max-background-pixels", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    args.weather_domains = set(args.weather_domains) if args.weather_domains else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, skipped = load_manifest(args.dataset_dir)
    groups: dict[str, list[object]] = defaultdict(list)
    for sample in samples:
        groups[sample.group].append(sample)
    manifest = {"dataset_dir": str(args.dataset_dir), "samples_loaded": len(samples), "skipped_samples": skipped}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = [row for group, values in sorted(groups.items()) for row in run_group(group, values, args, args.output_dir)]
    write_csv(args.output_dir / "cross_weather_results.csv", rows)
    completed = sum(row.get("status") == "ok" for row in rows)
    print(f"Wrote {args.output_dir / 'cross_weather_results.csv'}; completed {completed} evaluations.")


if __name__ == "__main__":
    main()
