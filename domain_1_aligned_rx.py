#!/usr/bin/env python3
"""Normal HSI-Drive Road ROI weather alignment and fixed-background RX.

This standalone offline experiment reads original HSI-Drive samples named
``nf[Season][Weather][DayTime][RoadType]_[frame]_RC_TC.npy`` and their semantic
labels.  It does not use ``Generated_Anomaly_Dataset`` or synthetic anomaly
labels.  A separate anomaly experiment can later reuse the saved CORAL
transforms and RX background.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


ROAD_LABEL = 1
NAME_PATTERN = re.compile(r"^nf(?P<condition>\d{4})_(?P<frame>.+)_RC_TC$")
DOMAIN_FACTORS = ("season", "weather", "daytime", "roadtype")
FACTOR_INDEX = {"season": 0, "weather": 1, "daytime": 2, "roadtype": 3}
FACTOR_ALIASES = {"0": "season", "1": "weather", "2": "daytime", "3": "roadtype"}


def parse_domain_factor(value: str) -> str:
    """Accept a factor name or its nf[Season][Weather][DayTime][RoadType] index."""
    factor = FACTOR_ALIASES.get(value.strip().lower(), value.strip().lower())
    if factor not in DOMAIN_FACTORS:
        choices = ", ".join(f"{index}={name}" for index, name in FACTOR_ALIASES.items())
        raise argparse.ArgumentTypeError(f"choose {choices}, or one of: {', '.join(DOMAIN_FACTORS)}")
    return factor


@dataclass(frozen=True)
class Sample:
    base: str
    condition: str
    group: str
    weather: str
    cube_path: Path
    label_path: Path

    @property
    def season(self) -> str:
        return self.condition[0]

    @property
    def daytime(self) -> str:
        return self.condition[2]

    @property
    def roadtype(self) -> str:
        return self.condition[3]

    def domain_value(self, factor: str) -> str:
        return self.condition[FACTOR_INDEX[factor]]

    def control_group(self, factor: str) -> str:
        index = FACTOR_INDEX[factor]
        return self.condition[:index] + self.condition[index + 1:]


def parse_sample_base(base: str) -> tuple[str, str, str]:
    """Return condition, controlled group, and weather from an nf filename."""
    match = NAME_PATTERN.fullmatch(base)
    if not match:
        raise ValueError(f"Expected nf[Season][Weather][DayTime][RoadType]_..._RC_TC, got {base!r}")
    condition = match.group("condition")
    return condition, condition[0] + condition[2] + condition[3], condition[1]


def load_manifest(dataset_dir: Path) -> tuple[list[Sample], list[str]]:
    """Discover original normal nf samples and their matching semantic labels."""
    cube_dir = dataset_dir / "Cubes_Scaling"
    label_dir = dataset_dir / "Labels"
    if not cube_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected {cube_dir} and {label_dir}")
    samples: list[Sample] = []
    skipped: list[str] = []
    for cube_path in sorted(cube_dir.glob("nf*_RC_TC.npy")):
        base = cube_path.stem
        try:
            condition, group, weather = parse_sample_base(base)
        except ValueError as exc:
            skipped.append(str(exc))
            continue
        # nf1113_557_RC_TC.npy -> nf1113_557.png
        label_base = base.removesuffix("_RC_TC")
        label_path = label_dir / f"{label_base}.png"
        if not label_path.is_file():
            skipped.append(f"{base}: missing label {label_path}")
            continue
        samples.append(Sample(base, condition, group, weather, cube_path, label_path))
    return samples, skipped


def load_label(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        label = np.asarray(image)
    return label[..., 0] if label.ndim == 3 else label


def cube_to_hwc(cube: np.ndarray) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3-D cube, got {cube.shape}")
    # HSI-Drive Cubes_Scaling stores 25 spectral bands in axis 0.
    return np.moveaxis(cube, 0, -1) if cube.shape[0] == 25 else cube


def road_features(sample: Sample) -> tuple[np.ndarray, tuple[int, int]]:
    hsi = cube_to_hwc(np.load(sample.cube_path)).astype(np.float64, copy=False)
    label = load_label(sample.label_path)
    if hsi.shape[:2] != label.shape:
        raise ValueError(f"Shape mismatch for {sample.base}: cube={hsi.shape}, label={label.shape}")
    mask = label == ROAD_LABEL
    if not np.any(mask):
        raise ValueError(f"{sample.base} has no Road ROI pixels (label {ROAD_LABEL})")
    return hsi[mask], label.shape


def capped_rows(x: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    return x if len(x) <= maximum else x[rng.choice(len(x), size=maximum, replace=False)]


def covariance(x: np.ndarray, reg: float) -> np.ndarray:
    if len(x) < 2:
        raise ValueError("At least two Road ROI pixels are required for covariance")
    cov = np.cov(x, rowvar=False)
    scale = max(float(np.trace(cov)) / cov.shape[0], 1.0)
    return cov + np.eye(cov.shape[0]) * (reg * scale)


def symmetric_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.power(np.clip(values, 1e-12, None), exponent)) @ vectors.T


@dataclass
class CoralTransform:
    mean: np.ndarray
    whitening: np.ndarray
    recoloring: np.ndarray
    target_mean: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.whitening @ self.recoloring + self.target_mean


def fit_coral(domain_features: dict[str, np.ndarray], reg: float) -> dict[str, CoralTransform]:
    """Align each weather distribution to the equal-weight common distribution."""
    means = {domain: values.mean(axis=0) for domain, values in domain_features.items()}
    covs = {domain: covariance(values, reg) for domain, values in domain_features.items()}
    target_mean = np.mean(list(means.values()), axis=0)
    target_cov = np.mean(list(covs.values()), axis=0)
    recoloring = symmetric_power(target_cov, 0.5)
    return {
        domain: CoralTransform(means[domain], symmetric_power(covs[domain], -0.5), recoloring, target_mean)
        for domain in domain_features
    }


@dataclass
class RXModel:
    mean: np.ndarray
    inv_cov: np.ndarray

    def score(self, x: np.ndarray) -> np.ndarray:
        delta = x - self.mean
        return np.einsum("ij,jk,ik->i", delta, self.inv_cov, delta)


def fit_rx(x: np.ndarray, reg: float) -> RXModel:
    return RXModel(x.mean(axis=0), np.linalg.pinv(covariance(x, reg)))


def split_per_domain(samples: list[Sample], test_fraction: float, seed: int, factor: str) -> tuple[list[Sample], list[Sample]]:
    """Hold out normal images independently in every selected domain."""
    by_domain: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_domain[sample.domain_value(factor)].append(sample)
    rng = np.random.default_rng(seed)
    train: list[Sample] = []
    test: list[Sample] = []
    for values in by_domain.values():
        if len(values) < 2:
            return [], []
        values = [values[index] for index in rng.permutation(len(values))]
        n_test = min(max(1, round(len(values) * test_fraction)), len(values) - 1)
        test.extend(values[:n_test])
        train.extend(values[n_test:])
    return train, test


def split_per_weather(samples: list[Sample], test_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Backward-compatible alias for the original weather experiment."""
    return split_per_domain(samples, test_fraction, seed, "weather")


def save_score_map(path: Path, scores: np.ndarray, sample: Sample) -> None:
    label = load_label(sample.label_path)
    road = label == ROAD_LABEL
    score_map = np.full(label.shape, np.nan, dtype=np.float32)
    score_map[road] = scores.astype(np.float32)
    np.save(path, score_map)


def save_alignment(path: Path, transforms: dict[str, CoralTransform], factor: str) -> None:
    values: dict[str, np.ndarray] = {}
    for weather, transform in transforms.items():
        values[f"mean_domain_{weather}"] = transform.mean
        values[f"whitening_domain_{weather}"] = transform.whitening
    first = next(iter(transforms.values()))
    values["recoloring"] = first.recoloring
    values["target_mean"] = first.target_mean
    values["domain_factor"] = np.asarray(factor)
    np.savez(path, **values)


def run_group(group: str, samples: list[Sample], args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
    factor = args.domain_factor
    selected = [sample for sample in samples if not args.domain_values or sample.domain_value(factor) in args.domain_values]
    domains = sorted({sample.domain_value(factor) for sample in selected})
    if args.domain_values and not args.domain_values.issubset(domains):
        missing = sorted(args.domain_values - set(domains))
        return [{"group": group, "domain_factor": factor, "status": "skipped", "reason": f"missing requested {factor} domains: {missing}"}]
    if len(domains) < args.min_domains:
        return [{"group": group, "status": "skipped", "reason": f"only {len(domains)} weather domains: {domains}"}]

    train, test = split_per_domain(selected, args.test_fraction, args.seed, factor)
    if not train:
        return [{"group": group, "status": "skipped", "reason": "each weather domain needs at least two normal nf samples"}]

    rng = np.random.default_rng(args.seed)
    by_domain: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample in train:
        features, _ = road_features(sample)
        by_domain[sample.domain_value(factor)].append(capped_rows(features, args.max_pixels_per_image, rng))
    domain_features = {domain: np.concatenate(by_domain[domain]) for domain in domains}
    transforms = fit_coral(domain_features, args.cov_reg)
    aligned_background = np.concatenate([transforms[weather].apply(domain_features[weather]) for weather in domains])
    model = fit_rx(capped_rows(aligned_background, args.max_background_pixels, rng), args.cov_reg)
    train_scores = model.score(aligned_background)
    threshold_99 = float(np.percentile(train_scores, 99))
    pooled_model: RXModel | None = None
    pooled_threshold: float | None = None
    if args.visualize_pooled_rx:
        pooled_background = capped_rows(np.concatenate([domain_features[domain] for domain in domains]), args.max_background_pixels, rng)
        pooled_model = fit_rx(pooled_background, args.cov_reg)
        pooled_threshold = float(np.percentile(pooled_model.score(pooled_background), 99))

    group_dir = output_dir / f"group_{group}"
    group_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "group": group,
        "train": [
            {"sample": sample.base, "condition": sample.condition, "domain": sample.domain_value(factor)}
            for sample in train
        ],
        "test": [
            {"sample": sample.base, "condition": sample.condition, "domain": sample.domain_value(factor)}
            for sample in test
        ],
    }
    (group_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    np.savez(group_dir / "rx_background.npz", mean=model.mean, inv_cov=model.inv_cov, train_score_p99=threshold_99)
    save_alignment(group_dir / "coral_alignment.npz", transforms, factor)
    if pooled_model is not None and pooled_threshold is not None:
        np.savez(
            group_dir / "pooled_rx_background.npz",
            mean=pooled_model.mean,
            inv_cov=pooled_model.inv_cov,
            train_score_p99=pooled_threshold,
        )
    rows: list[dict[str, object]] = []
    for sample in test:
        features, _ = road_features(sample)
        domain = sample.domain_value(factor)
        scores = model.score(transforms[domain].apply(features))
        score_path = group_dir / f"{sample.base}_scores.npy"
        save_score_map(score_path, scores, sample)
        row: dict[str, object] = {
            "group": group,
            "sample": sample.base,
            "condition": sample.condition,
            "domain_factor": factor,
            "domain": domain,
            "status": "ok",
            "road_pixels": len(scores),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_p95": float(np.percentile(scores, 95)),
            "score_p99": float(np.percentile(scores, 99)),
            "false_alarm_rate_at_train_p99": float(np.mean(scores > threshold_99)),
            "score_path": str(score_path),
        }
        if pooled_model is not None and pooled_threshold is not None:
            pooled_scores = pooled_model.score(features)
            pooled_score_path = group_dir / f"{sample.base}_pooled_scores.npy"
            save_score_map(pooled_score_path, pooled_scores, sample)
            row.update({
                "pooled_score_path": str(pooled_score_path),
                "pooled_score_mean": float(pooled_scores.mean()),
                "pooled_score_p99": float(np.percentile(pooled_scores, 99)),
                "pooled_false_alarm_rate_at_train_p99": float(np.mean(pooled_scores > pooled_threshold)),
            })
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("HSI_Drive"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--domain-factor", type=parse_domain_factor, default=None, metavar="FACTOR", help="Run one factor; omit to run all factors.")
    parser.add_argument("--all-domain-factors", action="store_true", help="Run season, weather, daytime, and roadtype experiments sequentially.")
    parser.add_argument("--domain-values", "--weather-domains", nargs="*", default=None, help="Optional domain IDs, e.g. 1 2 3 4.")
    parser.add_argument("--min-domains", type=int, default=2)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--cov-reg", type=float, default=1e-4)
    parser.add_argument("--max-pixels-per-image", type=int, default=20_000)
    parser.add_argument("--max-background-pixels", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument("--visualize", dest="visualize", action="store_true", help="Generate integrated visualizations after training (default).")
    visualization.add_argument("--no-visualize", dest="visualize", action="store_false", help="Skip automatic visualization.")
    parser.add_argument("--no-score-map-plots", dest="score_map_plots", action="store_false", help="Skip only per-image normal_score_maps PNGs.")
    parser.add_argument(
        "--visualize-pooled-rx",
        action="store_true",
        help="Also train the raw pooled RX baseline and save its per-image score maps and comparison plots.",
    )
    parser.set_defaults(visualize=True, score_map_plots=True)
    args = parser.parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be between 0 and 1")
    if args.min_domains < 2 or args.cov_reg <= 0:
        parser.error("--min-domains must be >= 2 and --cov-reg must be positive")
    return args


def run_visualization(args: argparse.Namespace) -> None:
    """Generate static score-map, distribution, and FAR plots in-process."""
    import matplotlib.pyplot as plt

    with (args.output_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    samples, _ = load_manifest(args.dataset_dir)
    samples_by_base = {sample.base: sample for sample in samples}

    def plot_pca(features: dict[str, np.ndarray], title: str, path: Path) -> None:
        domains = sorted(features)
        joined = np.vstack([features[domain] for domain in domains])
        centered = joined - joined.mean(axis=0, keepdims=True)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vectors[:2].T
        boundaries = np.cumsum([len(features[domain]) for domain in domains])
        fig, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
        start = 0
        for domain, end in zip(domains, boundaries):
            axis.scatter(projected[start:end, 0], projected[start:end, 1], s=5, alpha=0.30, label=f"{args.domain_factor} {domain}")
            start = end
        axis.set(title=title, xlabel="PC 1", ylabel="PC 2")
        axis.legend(markerscale=2)
        fig.savefig(path, dpi=180); plt.close(fig)

    def plot_covariance(before: dict[str, np.ndarray], after: dict[str, np.ndarray], path: Path) -> None:
        domains = sorted(before)
        matrices = []
        for values in (before, after):
            covariances = {domain: np.cov(values[domain], rowvar=False) for domain in domains}
            matrix = np.zeros((len(domains), len(domains)))
            for row, first in enumerate(domains):
                for col, second in enumerate(domains):
                    matrix[row, col] = np.linalg.norm(covariances[first] - covariances[second], ord="fro") / max(np.linalg.norm(covariances[first], ord="fro"), 1e-12)
            matrices.append(matrix)
        maximum = max(float(matrix.max()) for matrix in matrices)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        for axis, matrix, title in zip(axes, matrices, ("Before alignment", "After CORAL alignment")):
            image = axis.imshow(matrix, vmin=0, vmax=max(maximum, 1e-12), cmap="magma")
            axis.set(title=title, xticks=range(len(domains)), yticks=range(len(domains)))
            axis.set_xticklabels(domains); axis.set_yticklabels(domains)
            for row in range(len(domains)):
                for col in range(len(domains)):
                    axis.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=axes, label="relative covariance distance")
        fig.savefig(path, dpi=180); plt.close(fig)
    print("Generating integrated visualizations...")
    for group, group_rows in sorted(grouped.items()):
        visual_dir = args.output_dir / f"group_{group}" / "visualizations"
        maps_dir = visual_dir / "normal_score_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        threshold = float(np.load(args.output_dir / f"group_{group}" / "rx_background.npz")["train_score_p99"])
        saved_alignment = np.load(args.output_dir / f"group_{group}" / "coral_alignment.npz")
        transforms = {
            key.removeprefix("mean_domain_"): CoralTransform(saved_alignment[key], saved_alignment[f"whitening_domain_{key.removeprefix('mean_domain_')}"], saved_alignment["recoloring"], saved_alignment["target_mean"])
            for key in saved_alignment.files if key.startswith("mean_domain_")
        }
        scores_by_domain: dict[str, list[np.ndarray]] = defaultdict(list)
        pooled_scores_by_domain: dict[str, list[np.ndarray]] = defaultdict(list)
        before: dict[str, list[np.ndarray]] = defaultdict(list)
        after: dict[str, list[np.ndarray]] = defaultdict(list)
        rng = np.random.default_rng(args.seed)
        for row in group_rows:
            domain = row["domain"]
            score_map = np.load(Path(row["score_path"]))
            scores_by_domain[domain].append(score_map[np.isfinite(score_map)])
            if args.visualize_pooled_rx:
                pooled_score_path = row.get("pooled_score_path")
                if not pooled_score_path:
                    raise FileNotFoundError(
                        "Pooled RX results are missing. Re-run with --visualize-pooled-rx to create them."
                    )
                pooled_score_map = np.load(Path(pooled_score_path))
                pooled_scores_by_domain[domain].append(pooled_score_map[np.isfinite(pooled_score_map)])
            features, _ = road_features(samples_by_base[row["sample"]])
            features = capped_rows(features, 5_000, rng)
            before[domain].append(features)
            after[domain].append(transforms[domain].apply(features))
            if args.score_map_plots:
                cmap = plt.colormaps["turbo"].copy()
                cmap.set_bad("black")
                fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
                image = axis.imshow(np.ma.masked_invalid(score_map), cmap=cmap, vmin=np.nanpercentile(score_map, 1), vmax=np.nanpercentile(score_map, 99))
                axis.set(title=f"{row['sample']} | {args.domain_factor} {domain}")
                axis.axis("off")
                fig.colorbar(image, ax=axis, label="RX score")
                fig.savefig(maps_dir / f"{row['sample']}_score_map.png", dpi=180)
                plt.close(fig)
            if args.visualize_pooled_rx:
                pooled_maps_dir = visual_dir / "pooled_rx_score_maps"
                pooled_maps_dir.mkdir(parents=True, exist_ok=True)
                cmap = plt.colormaps["turbo"].copy()
                cmap.set_bad("black")
                fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
                image = axis.imshow(
                    np.ma.masked_invalid(pooled_score_map),
                    cmap=cmap,
                    vmin=np.nanpercentile(pooled_score_map, 1),
                    vmax=np.nanpercentile(pooled_score_map, 99),
                )
                axis.set(title=f"{row['sample']} | pooled RX | {args.domain_factor} {domain}")
                axis.axis("off")
                fig.colorbar(image, ax=axis, label="Pooled RX score")
                fig.savefig(pooled_maps_dir / f"{row['sample']}_pooled_score_map.png", dpi=180)
                plt.close(fig)
        domains = sorted(scores_by_domain)
        before_joined = {domain: np.concatenate(values) for domain, values in before.items()}
        after_joined = {domain: np.concatenate(values) for domain, values in after.items()}
        plot_pca(before_joined, f"Held-out normal Road features: before alignment ({args.domain_factor})", visual_dir / "feature_before_alignment.png")
        plot_pca(after_joined, f"Held-out normal Road features: after CORAL alignment ({args.domain_factor})", visual_dir / "feature_after_alignment.png")
        plot_covariance(before_joined, after_joined, visual_dir / "covariance_distance_before_after.png")
        values = [np.concatenate(scores_by_domain[domain]) for domain in domains]
        labels = [f"{args.domain_factor} {domain}" for domain in domains]
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        axis.boxplot(values, tick_labels=labels, showfliers=False)
        axis.axhline(threshold, color="tab:red", linestyle="--", label="RX train p99 threshold")
        axis.set(ylabel="Held-out normal Road RX score", title=f"RX score distribution by {args.domain_factor}")
        axis.legend()
        fig.savefig(visual_dir / "heldout_normal_rx_scores_by_domain.png", dpi=180)
        plt.close(fig)
        rates = [float(np.mean(value > threshold)) for value in values]
        fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        bars = axis.bar(labels, rates, color="tab:orange")
        axis.axhline(0.01, color="tab:red", linestyle="--", label="nominal 1% train p99")
        axis.set(ylabel="False-alarm rate", ylim=(0, max(0.02, max(rates) * 1.2)), title="Held-out normal Road false alarms")
        axis.legend()
        axis.bar_label(bars, labels=[f"{rate:.3f}" for rate in rates], padding=3)
        fig.savefig(visual_dir / "false_alarm_rate_by_domain.png", dpi=180)
        plt.close(fig)
        matrix = np.asarray([rates])
        fig, axis = plt.subplots(figsize=(max(5.5, len(domains) * 1.25), 4.5), constrained_layout=True)
        image = axis.imshow(matrix, vmin=0, vmax=max(0.05, float(matrix.max())), cmap="YlOrRd", aspect="auto")
        axis.set(
            title=f"Group {group}: integrated CORAL RX cross-{args.domain_factor} false alarms",
            xlabel=f"Test {args.domain_factor}",
            ylabel=f"Train {args.domain_factor}",
            xticks=range(len(domains)),
            yticks=[0],
        )
        axis.set_xticklabels(domains)
        axis.set_yticklabels(["all (CORAL)"])
        for col, rate in enumerate(rates):
            axis.text(col, 0, f"{rate:.1%}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=axis, label="False-alarm rate at train p99")
        fig.savefig(visual_dir / f"{args.domain_factor}_cross_fpr_heatmap.png", dpi=180)
        plt.close(fig)
        if args.visualize_pooled_rx:
            pooled_threshold = float(np.load(args.output_dir / f"group_{group}" / "pooled_rx_background.npz")["train_score_p99"])
            pooled_values = [np.concatenate(pooled_scores_by_domain[domain]) for domain in domains]
            fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
            axis.boxplot(pooled_values, tick_labels=labels, showfliers=False)
            axis.axhline(pooled_threshold, color="tab:red", linestyle="--", label="Pooled RX train p99 threshold")
            axis.set(ylabel="Held-out normal Road RX score", title=f"Pooled RX score distribution by {args.domain_factor}")
            axis.legend()
            fig.savefig(visual_dir / "pooled_rx_scores_by_domain.png", dpi=180)
            plt.close(fig)
            pooled_rates = [float(np.mean(value > pooled_threshold)) for value in pooled_values]
            fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
            positions = np.arange(len(domains))
            width = 0.38
            bars = axis.bar(positions - width / 2, rates, width, label="CORAL RX", color="tab:orange")
            pooled_bars = axis.bar(positions + width / 2, pooled_rates, width, label="Pooled RX", color="tab:blue")
            axis.axhline(0.01, color="tab:red", linestyle="--", label="nominal 1% train p99")
            axis.set(
                ylabel="False-alarm rate",
                ylim=(0, max(0.02, max(rates + pooled_rates) * 1.2)),
                title=f"CORAL vs pooled RX false alarms by {args.domain_factor}",
                xticks=positions,
            )
            axis.set_xticklabels(domains)
            axis.legend()
            axis.bar_label(bars, labels=[f"{rate:.3f}" for rate in rates], padding=3, fontsize=8)
            axis.bar_label(pooled_bars, labels=[f"{rate:.3f}" for rate in pooled_rates], padding=3, fontsize=8)
            fig.savefig(visual_dir / "coral_vs_pooled_false_alarm_rate_by_domain.png", dpi=180)
            plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    args.domain_values = set(args.domain_values) if args.domain_values else None
    if args.output_dir is None:
        args.output_dir = Path("domain_1_aligned_rx_results") / args.domain_factor
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, skipped = load_manifest(args.dataset_dir)
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.control_group(args.domain_factor)].append(sample)
    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "samples_loaded": len(samples),
        "domain_factor": args.domain_factor,
        "groups": {group: {domain: sum(s.domain_value(args.domain_factor) == domain for s in values) for domain in sorted({s.domain_value(args.domain_factor) for s in values})} for group, values in sorted(grouped.items())},
        "skipped_samples": skipped,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = [row for group, values in sorted(grouped.items()) for row in run_group(group, values, args, args.output_dir)]
    write_csv(args.output_dir / "results.csv", rows)
    completed = sum(row.get("status") == "ok" for row in rows)
    print(f"Loaded {len(samples)} normal nf samples into {len(grouped)} controlled groups.")
    print(f"Wrote {args.output_dir / 'manifest.json'} and {args.output_dir / 'results.csv'}.")
    print(f"Completed normal-Road score maps for {completed} held-out images.")
    if completed == 0:
        print("No valid group was evaluated. Inspect manifest.json/results.csv for domain or sample-count shortages.")
    elif args.visualize or args.visualize_pooled_rx:
        run_visualization(args)
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
    print("\n=== Saved domain-1 result folders ===")
    for output_dir in output_dirs:
        print(output_dir.resolve())


if __name__ == "__main__":
    main()
