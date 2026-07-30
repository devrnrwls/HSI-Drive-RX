#!/usr/bin/env python3
"""Visualize normal-Road alignment and RX background results.

Run ``domain_aligned_rx.py`` first.  This script reads its saved CORAL/RX
models and held-out score maps, then recreates held-out Road features from the
original HSI-Drive dataset for visualization only.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from domain_1_aligned_rx import CoralTransform, load_manifest, road_features


def pca_2d(x: np.ndarray) -> np.ndarray:
    """Dependency-free PCA projection for a feature matrix."""
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def capped_rows(x: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    return x if len(x) <= maximum else x[rng.choice(len(x), maximum, replace=False)]


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]


def load_transforms(path: Path) -> dict[str, CoralTransform]:
    saved = np.load(path)
    recoloring = saved["recoloring"]
    target_mean = saved["target_mean"]
    transforms: dict[str, CoralTransform] = {}
    for key in saved.files:
        if not (key.startswith("mean_domain_") or key.startswith("mean_weather_")):
            continue
        domain = key.removeprefix("mean_domain_").removeprefix("mean_weather_")
        prefix = "domain" if f"whitening_domain_{domain}" in saved else "weather"
        transforms[domain] = CoralTransform(
            mean=saved[key],
            whitening=saved[f"whitening_{prefix}_{domain}"],
            recoloring=recoloring,
            target_mean=target_mean,
        )
    return transforms


def plot_feature_scatter(
    features: dict[str, np.ndarray],
    title: str,
    path: Path,
) -> None:
    weather_order = sorted(features)
    all_features = np.vstack([features[weather] for weather in weather_order])
    projected = pca_2d(all_features)
    boundaries = np.cumsum([len(features[weather]) for weather in weather_order])
    start = 0
    fig, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    for weather, end in zip(weather_order, boundaries):
        axis.scatter(projected[start:end, 0], projected[start:end, 1], s=5, alpha=0.30, label=f"Weather {weather}")
        start = end
    axis.set(title=title, xlabel="PC 1", ylabel="PC 2")
    axis.legend(markerscale=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pairwise_covariance_distance(features: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    weather_order = sorted(features)
    covariances = {weather: np.cov(features[weather], rowvar=False) for weather in weather_order}
    matrix = np.zeros((len(weather_order), len(weather_order)), dtype=float)
    for row, first in enumerate(weather_order):
        for col, second in enumerate(weather_order):
            denominator = max(np.linalg.norm(covariances[first], ord="fro"), 1e-12)
            matrix[row, col] = np.linalg.norm(covariances[first] - covariances[second], ord="fro") / denominator
    return weather_order, matrix


def plot_distance_heatmap(before: dict[str, np.ndarray], after: dict[str, np.ndarray], path: Path) -> None:
    weather_order, before_matrix = pairwise_covariance_distance(before)
    _, after_matrix = pairwise_covariance_distance(after)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    maximum = max(float(before_matrix.max()), float(after_matrix.max()), 1e-12)
    for axis, matrix, title in zip(axes, (before_matrix, after_matrix), ("Before alignment", "After alignment")):
        image = axis.imshow(matrix, vmin=0, vmax=maximum, cmap="magma")
        axis.set(title=title, xticks=range(len(weather_order)), yticks=range(len(weather_order)))
        axis.set_xticklabels(weather_order)
        axis.set_yticklabels(weather_order)
        for row in range(len(weather_order)):
            for col in range(len(weather_order)):
                axis.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=axes, label="relative covariance distance")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_score_distribution(scores: dict[str, list[np.ndarray]], threshold: float, path: Path) -> None:
    weather_order = sorted(scores)
    values = [np.concatenate(scores[weather]) for weather in weather_order]
    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.boxplot(values, tick_labels=[f"Weather {weather}" for weather in weather_order], showfliers=False)
    axis.axhline(threshold, color="tab:red", linestyle="--", label="RX train p99 threshold")
    axis.set(ylabel="Held-out normal Road RX score", title="RX score distribution by weather")
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_false_alarm_rate(scores: dict[str, list[np.ndarray]], threshold: float, path: Path) -> None:
    weather_order = sorted(scores)
    rates = [float(np.mean(np.concatenate(scores[weather]) > threshold)) for weather in weather_order]
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bars = axis.bar([f"Weather {weather}" for weather in weather_order], rates, color="tab:orange")
    axis.axhline(0.01, color="tab:red", linestyle="--", label="nominal 1% train p99")
    axis.set(ylabel="False-alarm rate", ylim=(0, max(0.02, max(rates) * 1.2)), title="Held-out normal Road false alarms")
    axis.legend()
    axis.bar_label(bars, labels=[f"{rate:.3f}" for rate in rates], padding=3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cross_weather_fpr_heatmap(
    scores: dict[str, list[np.ndarray]],
    threshold: float,
    group: str,
    path: Path,
) -> list[dict[str, object]]:
    """Plot the domain-4-comparable aligned-RX false-alarm-rate heatmap.

    Aligned RX has one background model trained on all weather domains after
    CORAL, hence this is a one-row matrix rather than the source-weather by
    target-weather matrix used by single-weather RX.
    """
    weather_order = sorted(scores)
    rates = [float(np.mean(np.concatenate(scores[weather]) > threshold)) for weather in weather_order]
    matrix = np.asarray([rates])
    fig, axis = plt.subplots(
        figsize=(max(5.5, len(weather_order) * 1.25), 4.5),
        constrained_layout=True,
    )
    image = axis.imshow(matrix, vmin=0, vmax=max(0.05, float(matrix.max())), cmap="YlOrRd")
    axis.set(
        title=f"Group {group}: CORAL-aligned RX cross-weather false alarms",
        xlabel="Test weather",
        ylabel="Train weather",
        xticks=range(len(weather_order)),
        yticks=[0],
    )
    axis.set_xticklabels(weather_order)
    axis.set_yticklabels(["all (CORAL)"])
    for col, rate in enumerate(rates):
        axis.text(col, 0, f"{rate:.1%}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axis, label="False-alarm rate at train p99")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [
        {
            "group": group,
            "model": "coral_aligned_rx",
            "train_weather": "all",
            "test_weather": weather,
            "threshold_source_train_p99": threshold,
            "false_alarm_rate": rate,
            "test_road_pixels": len(np.concatenate(scores[weather])),
        }
        for weather, rate in zip(weather_order, rates)
    ]


def plot_score_map(score_path: Path, output_path: Path, title: str) -> None:
    score_map = np.load(score_path)
    masked = np.ma.masked_invalid(score_map)
    colormap = plt.colormaps["turbo"].copy()
    colormap.set_bad("black")
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    image = axis.imshow(masked, cmap=colormap, vmin=np.nanpercentile(score_map, 1), vmax=np.nanpercentile(score_map, 99))
    axis.set(title=title)
    axis.axis("off")
    fig.colorbar(image, ax=axis, label="RX score")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def visualize_group(
    group: str,
    rows: list[dict[str, str]],
    samples_by_base: dict[str, object],
    output_dir: Path,
    max_pixels: int,
    seed: int,
    factor: str,
) -> None:
    group_dir = output_dir / f"group_{group}"
    transforms = load_transforms(group_dir / "coral_alignment.npz")
    rx_data = np.load(group_dir / "rx_background.npz")
    threshold = float(rx_data["train_score_p99"])
    rng = np.random.default_rng(seed)
    before: dict[str, list[np.ndarray]] = defaultdict(list)
    after: dict[str, list[np.ndarray]] = defaultdict(list)
    scores: dict[str, list[np.ndarray]] = defaultdict(list)
    visualization_dir = group_dir / "visualizations"
    maps_dir = visualization_dir / "normal_score_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        sample = samples_by_base.get(row["sample"])
        if sample is None:
            raise FileNotFoundError(f"Could not find original nf sample {row['sample']}")
        weather = row.get("domain", row.get("weather", ""))
        features, _ = road_features(sample)
        features = capped_rows(features, max_pixels, rng)
        before[weather].append(features)
        after[weather].append(transforms[weather].apply(features))
        score_path = Path(row["score_path"])
        if not score_path.is_absolute():
            score_path = Path.cwd() / score_path
        score_map = np.load(score_path)
        scores[weather].append(score_map[np.isfinite(score_map)])
        plot_score_map(score_path, maps_dir / f"{row['sample']}_score_map.png", f"{row['sample']} | {factor} {weather}")

    before_joined = {weather: np.concatenate(values) for weather, values in before.items()}
    after_joined = {weather: np.concatenate(values) for weather, values in after.items()}
    plot_feature_scatter(before_joined, "Held-out normal Road features: before alignment", visualization_dir / "feature_before_alignment.png")
    plot_feature_scatter(after_joined, "Held-out normal Road features: after CORAL alignment", visualization_dir / "feature_after_alignment.png")
    plot_distance_heatmap(before_joined, after_joined, visualization_dir / "covariance_distance_before_after.png")
    plot_score_distribution(scores, threshold, visualization_dir / "heldout_normal_rx_scores_by_weather.png")
    plot_false_alarm_rate(scores, threshold, visualization_dir / "false_alarm_rate_by_weather.png")
    cross_weather_rows = plot_cross_weather_fpr_heatmap(
        scores,
        threshold,
        group,
        visualization_dir / "aligned_cross_weather_fpr_heatmap.png",
    )
    write_csv(visualization_dir / "aligned_cross_weather_fpr.csv", cross_weather_rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("HSI_Drive"))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--domain-factor", choices=("season", "weather", "daytime", "roadtype"), default="weather")
    parser.add_argument("--groups", nargs="*", help="Optional controlled group IDs, e.g. 113 111.")
    parser.add_argument("--max-pixels-per-weather", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.results_dir is None:
        args.results_dir = Path(f"domain_{args.domain_factor}_aligned_rx_results")
    results_path = args.results_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Run domain_aligned_rx.py first; missing {results_path}")
    samples, skipped = load_manifest(args.dataset_dir)
    if skipped:
        print(f"Ignoring {len(skipped)} original samples with missing/invalid inputs.")
    samples_by_base = {sample.base: sample for sample in samples}
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_results(results_path):
        if not args.groups or row["group"] in args.groups:
            by_group[row["group"]].append(row)
    if not by_group:
        raise ValueError("No completed groups found in results.csv for the requested selection.")
    for group, rows in sorted(by_group.items()):
        print(f"Visualizing group {group}: {len(rows)} held-out normal images")
        visualize_group(group, rows, samples_by_base, args.results_dir, args.max_pixels_per_weather, args.seed, args.domain_factor)


if __name__ == "__main__":
    main()
