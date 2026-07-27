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


@dataclass(frozen=True)
class Sample:
    base: str
    condition: str
    group: str
    weather: str
    cube_path: Path
    label_path: Path


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


def split_per_weather(samples: list[Sample], test_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Hold out normal images in every weather domain for normal-score checks."""
    by_weather: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_weather[sample.weather].append(sample)
    rng = np.random.default_rng(seed)
    train: list[Sample] = []
    test: list[Sample] = []
    for weather, values in by_weather.items():
        if len(values) < 2:
            return [], []
        values = [values[index] for index in rng.permutation(len(values))]
        n_test = min(max(1, round(len(values) * test_fraction)), len(values) - 1)
        test.extend(values[:n_test])
        train.extend(values[n_test:])
    return train, test


def save_score_map(path: Path, scores: np.ndarray, sample: Sample) -> None:
    label = load_label(sample.label_path)
    road = label == ROAD_LABEL
    score_map = np.full(label.shape, np.nan, dtype=np.float32)
    score_map[road] = scores.astype(np.float32)
    np.save(path, score_map)


def save_alignment(path: Path, transforms: dict[str, CoralTransform]) -> None:
    values: dict[str, np.ndarray] = {}
    for weather, transform in transforms.items():
        values[f"mean_weather_{weather}"] = transform.mean
        values[f"whitening_weather_{weather}"] = transform.whitening
    first = next(iter(transforms.values()))
    values["recoloring"] = first.recoloring
    values["target_mean"] = first.target_mean
    np.savez(path, **values)


def run_group(group: str, samples: list[Sample], args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
    selected = [sample for sample in samples if not args.weather_domains or sample.weather in args.weather_domains]
    domains = sorted({sample.weather for sample in selected})
    if args.weather_domains and not args.weather_domains.issubset(domains):
        missing = sorted(args.weather_domains - set(domains))
        return [{"group": group, "status": "skipped", "reason": f"missing requested weather domains: {missing}"}]
    if len(domains) < args.min_domains:
        return [{"group": group, "status": "skipped", "reason": f"only {len(domains)} weather domains: {domains}"}]

    train, test = split_per_weather(selected, args.test_fraction, args.seed)
    if not train:
        return [{"group": group, "status": "skipped", "reason": "each weather domain needs at least two normal nf samples"}]

    rng = np.random.default_rng(args.seed)
    by_domain: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample in train:
        features, _ = road_features(sample)
        by_domain[sample.weather].append(capped_rows(features, args.max_pixels_per_image, rng))
    domain_features = {weather: np.concatenate(by_domain[weather]) for weather in domains}
    transforms = fit_coral(domain_features, args.cov_reg)
    aligned_background = np.concatenate([transforms[weather].apply(domain_features[weather]) for weather in domains])
    model = fit_rx(capped_rows(aligned_background, args.max_background_pixels, rng), args.cov_reg)
    train_scores = model.score(aligned_background)
    threshold_99 = float(np.percentile(train_scores, 99))

    group_dir = output_dir / f"group_{group}"
    group_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "group": group,
        "train": [
            {"sample": sample.base, "condition": sample.condition, "weather": sample.weather}
            for sample in train
        ],
        "test": [
            {"sample": sample.base, "condition": sample.condition, "weather": sample.weather}
            for sample in test
        ],
    }
    (group_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    np.savez(group_dir / "rx_background.npz", mean=model.mean, inv_cov=model.inv_cov, train_score_p99=threshold_99)
    save_alignment(group_dir / "coral_alignment.npz", transforms)
    rows: list[dict[str, object]] = []
    for sample in test:
        features, _ = road_features(sample)
        scores = model.score(transforms[sample.weather].apply(features))
        score_path = group_dir / f"{sample.base}_scores.npy"
        save_score_map(score_path, scores, sample)
        rows.append({
            "group": group,
            "sample": sample.base,
            "condition": sample.condition,
            "weather": sample.weather,
            "status": "ok",
            "road_pixels": len(scores),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_p95": float(np.percentile(scores, 95)),
            "score_p99": float(np.percentile(scores, 99)),
            "false_alarm_rate_at_train_p99": float(np.mean(scores > threshold_99)),
            "score_path": str(score_path),
        })
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
    parser.add_argument("--output-dir", type=Path, default=Path("domain_aligned_rx_results"))
    parser.add_argument("--weather-domains", nargs="*", default=None, help="Optional IDs, e.g. 1 2 3 4.")
    parser.add_argument("--min-domains", type=int, default=2)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--cov-reg", type=float, default=1e-4)
    parser.add_argument("--max-pixels-per-image", type=int, default=20_000)
    parser.add_argument("--max-background-pixels", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be between 0 and 1")
    if args.min_domains < 2 or args.cov_reg <= 0:
        parser.error("--min-domains must be >= 2 and --cov-reg must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.weather_domains = set(args.weather_domains) if args.weather_domains else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, skipped = load_manifest(args.dataset_dir)
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group].append(sample)
    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "samples_loaded": len(samples),
        "groups": {group: {weather: sum(s.weather == weather for s in values) for weather in sorted({s.weather for s in values})} for group, values in sorted(grouped.items())},
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


if __name__ == "__main__":
    main()
