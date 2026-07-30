#!/usr/bin/env python3
"""Compare cross-domain false alarms of single-domain, pooled, and aligned RX.

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
    DOMAIN_FACTORS,
    capped_rows,
    fit_coral,
    fit_rx,
    load_manifest,
    parse_domain_factor,
    road_features,
    save_score_map,
    split_per_domain,
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


def heatmap(matrix: np.ndarray, labels: list[str], title: str, colorbar_label: str, path: Path, factor: str = "weather") -> None:
    fig, axis = plt.subplots(figsize=(max(5.5, len(labels) * 1.25), max(4.5, len(labels) * 1.1)), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=max(0.05, float(np.nanmax(matrix))), cmap="YlOrRd")
    axis.set(
        title=title,
        xlabel=f"Test {factor}",
        ylabel=f"Train {factor}",
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


def plot_alignment_diagnostics(before: dict[str, np.ndarray], after: dict[str, np.ndarray], factor: str, output_dir: Path) -> None:
    """Save PCA before/after plots and covariance-distance comparison."""
    domains = sorted(before)

    def pca_plot(features: dict[str, np.ndarray], title: str, path: Path) -> None:
        joined = np.vstack([features[domain] for domain in domains])
        centered = joined - joined.mean(axis=0, keepdims=True)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vectors[:2].T
        boundaries = np.cumsum([len(features[domain]) for domain in domains])
        fig, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
        start = 0
        for domain, end in zip(domains, boundaries):
            axis.scatter(projected[start:end, 0], projected[start:end, 1], s=5, alpha=0.30, label=f"{factor} {domain}")
            start = end
        axis.set(title=title, xlabel="PC 1", ylabel="PC 2"); axis.legend(markerscale=2)
        fig.savefig(path, dpi=180); plt.close(fig)

    pca_plot(before, f"Held-out normal Road features: before alignment ({factor})", output_dir / "feature_before_alignment.png")
    pca_plot(after, f"Held-out normal Road features: after CORAL alignment ({factor})", output_dir / "feature_after_alignment.png")
    matrices = []
    for features in (before, after):
        covariances = {domain: np.cov(features[domain], rowvar=False) for domain in domains}
        matrices.append(np.asarray([[np.linalg.norm(covariances[first] - covariances[second], ord="fro") / max(np.linalg.norm(covariances[first], ord="fro"), 1e-12) for second in domains] for first in domains]))
    maximum = max(float(matrix.max()) for matrix in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, matrix, title in zip(axes, matrices, ("Before alignment", "After CORAL alignment")):
        image = axis.imshow(matrix, vmin=0, vmax=max(maximum, 1e-12), cmap="magma")
        axis.set(title=title, xticks=range(len(domains)), yticks=range(len(domains)))
        axis.set_xticklabels(domains); axis.set_yticklabels(domains)
    fig.colorbar(image, ax=axes, label="relative covariance distance")
    fig.savefig(output_dir / "covariance_distance_before_after.png", dpi=180); plt.close(fig)


def save_pooled_score_map_visualizations(
    test: list[object],
    model: object,
    threshold: float,
    factor: str,
    group_dir: Path,
) -> None:
    """Save full-resolution held-out score maps for the raw pooled RX baseline."""
    maps_dir = group_dir / "visualizations" / "pooled_rx_score_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    scores_by_domain: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample in test:
        features, _ = road_features(sample)
        scores = model.score(features)
        score_path = group_dir / f"{sample.base}_pooled_scores.npy"
        save_score_map(score_path, scores, sample)
        cmap = plt.colormaps["turbo"].copy()
        cmap.set_bad("black")
        score_map = np.load(score_path)
        fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        image = axis.imshow(
            np.ma.masked_invalid(score_map),
            cmap=cmap,
            vmin=np.nanpercentile(score_map, 1),
            vmax=np.nanpercentile(score_map, 99),
        )
        domain = sample.domain_value(factor)
        scores_by_domain[domain].append(scores)
        axis.set(title=f"{sample.base} | pooled RX | {factor} {domain}")
        axis.axis("off")
        fig.colorbar(image, ax=axis, label="Pooled RX score")
        fig.savefig(maps_dir / f"{sample.base}_pooled_score_map.png", dpi=180)
        plt.close(fig)
        rows.append({
            "group": group_dir.name.removeprefix("group_"),
            "sample": sample.base,
            "domain_factor": factor,
            "domain": domain,
            "threshold_train_p99": threshold,
            "false_alarm_rate": float(np.mean(scores > threshold)),
            "score_mean": float(scores.mean()),
            "score_p99": float(np.percentile(scores, 99)),
            "score_path": str(score_path),
        })
    write_csv(group_dir / "pooled_rx_score_map_results.csv", rows)
    domains = sorted(scores_by_domain)
    values = [np.concatenate(scores_by_domain[domain]) for domain in domains]
    labels = [f"{factor} {domain}" for domain in domains]
    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.boxplot(values, tick_labels=labels, showfliers=False)
    axis.axhline(threshold, color="tab:red", linestyle="--", label="Pooled RX train p99 threshold")
    axis.set(ylabel="Held-out normal Road RX score", title=f"Pooled RX score distribution by {factor}")
    axis.legend()
    fig.savefig(group_dir / "visualizations" / "pooled_rx_scores_by_domain.png", dpi=180)
    plt.close(fig)
    rates = [float(np.mean(values[index] > threshold)) for index in range(len(values))]
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bars = axis.bar(labels, rates, color="tab:blue")
    axis.axhline(0.01, color="tab:red", linestyle="--", label="nominal 1% train p99")
    axis.set(
        ylabel="False-alarm rate",
        ylim=(0, max(0.02, max(rates) * 1.2)),
        title=f"Pooled RX false alarms by {factor}",
    )
    axis.legend()
    axis.bar_label(bars, labels=[f"{rate:.3f}" for rate in rates], padding=3)
    fig.savefig(group_dir / "visualizations" / "pooled_rx_false_alarm_rate_by_domain.png", dpi=180)
    plt.close(fig)


def run_group(group: str, samples: list[object], args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
    factor = args.domain_factor
    selected = [sample for sample in samples if not args.domain_values or sample.domain_value(factor) in args.domain_values]
    domains = sorted({sample.domain_value(factor) for sample in selected})
    if args.domain_values and not args.domain_values.issubset(domains):
        return [{"group": group, "domain_factor": factor, "status": "skipped", "reason": f"missing requested {factor} domains: {sorted(args.domain_values - set(domains))}"}]
    if len(domains) < args.min_domains:
        return [{"group": group, "domain_factor": factor, "status": "skipped", "reason": f"only {len(domains)} {factor} domains"}]

    train, test = split_per_domain(selected, args.test_fraction, args.seed, factor)
    if not train:
        return [{"group": group, "domain_factor": factor, "status": "skipped", "reason": f"each {factor} domain needs at least two images"}]

    rng = np.random.default_rng(args.seed)
    train_by_weather: dict[str, list[np.ndarray]] = defaultdict(list)
    test_by_weather: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample in train:
        features, _ = road_features(sample)
        train_by_weather[sample.domain_value(factor)].append(capped_rows(features, args.max_pixels_per_image, rng))
    for sample in test:
        features, _ = road_features(sample)
        test_by_weather[sample.domain_value(factor)].append(capped_rows(features, args.max_pixels_per_image, rng))
    train_x = {domain: np.concatenate(train_by_weather[domain]) for domain in domains}
    test_x = {domain: np.concatenate(test_by_weather[domain]) for domain in domains}

    group_dir = output_dir / f"group_{group}"
    group_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "group": group,
        "domain_factor": factor,
        "train": [{"sample": sample.base, "domain": sample.domain_value(factor)} for sample in train],
        "test": [{"sample": sample.base, "domain": sample.domain_value(factor)} for sample in test],
    }
    (group_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    single_matrix = np.zeros((len(domains), len(domains)), dtype=float)
    model_prefix = "single_weather" if factor == "weather" else f"single_{factor}"
    for source_index, source_domain in enumerate(domains):
        source_train = capped_rows(train_x[source_domain], args.max_background_pixels, rng)
        model = fit_rx(source_train, args.cov_reg)
        threshold = float(np.percentile(model.score(source_train), 99))
        np.savez(
            group_dir / f"{model_prefix}_{source_domain}_rx_background.npz",
            mean=model.mean,
            inv_cov=model.inv_cov,
            train_score_p99=threshold,
        )
        for target_index, target_domain in enumerate(domains):
            scores = model.score(test_x[target_domain])
            summary = score_summary(scores, threshold)
            single_matrix[source_index, target_index] = summary["false_alarm_rate"]
            rows.append({
                "group": group,
                "domain_factor": factor,
                "model": "single_weather_rx",
                "train_domain": source_domain,
                "test_domain": target_domain,
                "threshold_source_train_p99": threshold,
                "train_images": sum(sample.domain_value(factor) == source_domain for sample in train),
                "test_images": sum(sample.domain_value(factor) == target_domain for sample in test),
                "test_road_pixels": len(scores),
                "status": "ok",
                **summary,
            })
    heatmap(
        single_matrix,
        domains,
        f"Group {group}: single-{factor} RX cross-domain false alarms",
        "False-alarm rate at source train p99",
        group_dir / "single_weather_cross_fpr_heatmap.png",
        factor,
    )

    # Fair controls: one model trained on all weather data before/after CORAL.
    pooled_train = np.concatenate([train_x[domain] for domain in domains])
    pooled_train = capped_rows(pooled_train, args.max_background_pixels, rng)
    pooled_model = fit_rx(pooled_train, args.cov_reg)
    pooled_threshold = float(np.percentile(pooled_model.score(pooled_train), 99))
    transforms = fit_coral(train_x, args.cov_reg)
    diagnostics_dir = group_dir / "visualizations"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    plot_alignment_diagnostics(test_x, {domain: transforms[domain].apply(test_x[domain]) for domain in domains}, factor, diagnostics_dir)
    aligned_train = np.concatenate([transforms[domain].apply(train_x[domain]) for domain in domains])
    aligned_background = capped_rows(aligned_train, args.max_background_pixels, rng)
    aligned_model = fit_rx(aligned_background, args.cov_reg)
    aligned_threshold = float(np.percentile(aligned_model.score(aligned_background), 99))
    np.savez(group_dir / "pooled_rx_background.npz", mean=pooled_model.mean, inv_cov=pooled_model.inv_cov, train_score_p99=pooled_threshold)
    np.savez(group_dir / "aligned_rx_background.npz", mean=aligned_model.mean, inv_cov=aligned_model.inv_cov, train_score_p99=aligned_threshold)
    if args.visualize_pooled_rx:
        save_pooled_score_map_visualizations(test, pooled_model, pooled_threshold, factor, group_dir)

    for model_name, model, threshold in (
        ("pooled_rx", pooled_model, pooled_threshold),
        ("coral_aligned_rx", aligned_model, aligned_threshold),
    ):
        for target_domain in domains:
            values = test_x[target_domain] if model_name == "pooled_rx" else transforms[target_domain].apply(test_x[target_domain])
            scores = model.score(values)
            rows.append({
                "group": group,
                "domain_factor": factor,
                "model": model_name,
                "train_domain": "all",
                "test_domain": target_domain,
                "threshold_source_train_p99": threshold,
                "train_images": len(train),
                "test_images": sum(sample.domain_value(factor) == target_domain for sample in test),
                "test_road_pixels": len(scores),
                "status": "ok",
                **score_summary(scores, threshold),
            })

    single_off_diagonal = [single_matrix[row, col] for row in range(len(domains)) for col in range(len(domains)) if row != col]
    summary_rows = [
        {
            "group": group,
            "domain_factor": factor,
            "model": "single_weather_rx",
            "mean_cross_weather_false_alarm_rate": float(np.mean(single_off_diagonal)),
        }
    ]
    for model_name in ("pooled_rx", "coral_aligned_rx"):
        fprs = [float(row["false_alarm_rate"]) for row in rows if row["model"] == model_name]
        summary_rows.append({"group": group, "domain_factor": factor, "model": model_name, "mean_cross_weather_false_alarm_rate": float(np.mean(fprs))})
    write_csv(group_dir / "comparison_summary.csv", summary_rows)
    grouped_bar(summary_rows, group_dir / "model_comparison_false_alarm_rate.png")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("HSI_Drive"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--domain-factor", type=parse_domain_factor, default=None, metavar="FACTOR", help="Run one factor; omit to run all factors.")
    parser.add_argument("--all-domain-factors", action="store_true", help="Run season, weather, daytime, and roadtype sequentially.")
    parser.add_argument("--domain-values", "--weather-domains", nargs="*", default=None, help="Optional domain IDs, e.g. 1 2 3 4.")
    parser.add_argument("--min-domains", type=int, default=2)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--cov-reg", type=float, default=1e-4)
    parser.add_argument("--max-pixels-per-image", type=int, default=20_000)
    parser.add_argument("--max-background-pixels", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visualize-pooled-rx",
        action="store_true",
        help="Save full-resolution held-out pooled-RX score maps and PNG visualizations.",
    )
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    return args


def run_experiment(args: argparse.Namespace) -> None:
    args.domain_values = set(args.domain_values) if args.domain_values else None
    if args.output_dir is None:
        args.output_dir = Path("domain_2_single_domain_rx_results") / args.domain_factor
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, skipped = load_manifest(args.dataset_dir)
    groups: dict[str, list[object]] = defaultdict(list)
    for sample in samples:
        groups[sample.control_group(args.domain_factor)].append(sample)
    manifest = {"dataset_dir": str(args.dataset_dir), "samples_loaded": len(samples), "skipped_samples": skipped}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = [row for group, values in sorted(groups.items()) for row in run_group(group, values, args, args.output_dir)]
    write_csv(args.output_dir / "cross_weather_results.csv", rows)
    completed = sum(row.get("status") == "ok" for row in rows)
    print(f"Wrote {args.output_dir / 'cross_weather_results.csv'}; completed {completed} evaluations.")
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
        print(f"\n=== Running domain factor: {factor} ===")
        output_dirs.append(run_experiment(run_args))
    print("\n=== Saved domain-2 comparison folders ===")
    for output_dir in output_dirs:
        print(output_dir.resolve())


if __name__ == "__main__":
    main()
