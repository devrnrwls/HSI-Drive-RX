#!/usr/bin/env python3
"""Create side-by-side single-weather and CORAL-aligned RX FPR heatmaps.

The output puts the domain-4 ``single_weather_cross_fpr_heatmap`` and the
domain-2 ``aligned_cross_weather_fpr_heatmap`` on one figure with a shared
color range, so absolute false-alarm rates can be compared directly.

실행:

  python domain_6_compare_cross_weather_heatmaps.py

  특정 그룹만 보려면:

  python domain_6_compare_cross_weather_heatmaps.py --groups 113
"""


from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def single_weather_matrix(rows: list[dict[str, str]], group: str) -> tuple[list[str], list[str], np.ndarray]:
    selected = [
        row for row in rows
        if row.get("status") == "ok" and row.get("group") == group and row.get("model") == "single_weather_rx"
    ]
    train_weather = sorted({row["train_weather"] for row in selected})
    test_weather = sorted({row["test_weather"] for row in selected})
    if not selected:
        raise ValueError(f"No completed single-weather RX rows for group {group}.")
    values = {(row["train_weather"], row["test_weather"]): float(row["false_alarm_rate"]) for row in selected}
    matrix = np.asarray([[values[(train, test)] for test in test_weather] for train in train_weather])
    return train_weather, test_weather, matrix


def aligned_weather_rates(results_dir: Path, group: str) -> dict[str, float]:
    """Aggregate domain-1 held-out score maps exactly as domain 2 does."""
    rows = [
        row for row in read_csv(results_dir / "results.csv")
        if row.get("status") == "ok" and row.get("group") == group
    ]
    if not rows:
        raise ValueError(f"No completed integrated RX rows for group {group}.")
    background = np.load(results_dir / f"group_{group}" / "rx_background.npz")
    threshold = float(background["train_score_p99"])
    by_weather: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        score_path = Path(row["score_path"])
        if not score_path.is_absolute():
            score_path = Path.cwd() / score_path
        score_map = np.load(score_path)
        by_weather[row["weather"]].append(score_map[np.isfinite(score_map)])
    return {
        weather: float(np.mean(np.concatenate(scores) > threshold))
        for weather, scores in by_weather.items()
    }


def draw_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    maximum: float,
) -> plt.AxesImage:
    image = axis.imshow(matrix, vmin=0, vmax=maximum, cmap="YlOrRd", aspect="auto")
    axis.set(title=title, xlabel="Test weather", ylabel="Train weather", xticks=range(len(x_labels)), yticks=range(len(y_labels)))
    axis.set_xticklabels(x_labels)
    axis.set_yticklabels(y_labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, f"{matrix[row, col]:.1%}", ha="center", va="center", fontsize=9)
    return image


def create_group_figure(single_rows: list[dict[str, str]], integrated_dir: Path, group: str, output_dir: Path) -> Path:
    train_weather, test_weather, single = single_weather_matrix(single_rows, group)
    aligned_rates = aligned_weather_rates(integrated_dir, group)
    missing = set(test_weather) - set(aligned_rates)
    if missing:
        raise ValueError(f"Group {group}: integrated RX lacks test weather {sorted(missing)}.")
    aligned = np.asarray([[aligned_rates[weather] for weather in test_weather]])
    combined = np.vstack([single, aligned])
    row_labels = [f"weather {weather}" for weather in train_weather] + ["all (CORAL)"]
    maximum = max(0.05, float(combined.max()))
    fig, axis = plt.subplots(
        figsize=(max(6.5, len(test_weather) * 1.55), max(4.5, len(row_labels) * 1.05)),
        constrained_layout=True,
    )
    image = draw_heatmap(
        axis,
        combined,
        test_weather,
        row_labels,
        f"Group {group}: single-weather RX + domain-integrated CORAL RX",
        maximum,
    )
    fig.colorbar(image, ax=axis, label="False-alarm rate at train p99")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"group_{group}_single_vs_aligned_cross_fpr_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated-results-dir", type=Path, default=Path("domain_aligned_rx_results"))
    parser.add_argument("--single-weather-results-dir", type=Path, default=Path("single_weather_rx_comparison_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("cross_weather_heatmap_comparisons"))
    parser.add_argument("--groups", nargs="*", help="Optional controlled group IDs, e.g. 113 111.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    single_rows = read_csv(args.single_weather_results_dir / "cross_weather_results.csv")
    groups = sorted({
        row["group"] for row in single_rows
        if row.get("status") == "ok" and row.get("model") == "single_weather_rx"
    })
    if args.groups:
        groups = [group for group in groups if group in args.groups]
    if not groups:
        raise ValueError("No completed groups found for the requested selection.")
    paths = [create_group_figure(single_rows, args.integrated_results_dir, group, args.output_dir) for group in groups]
    print(f"Wrote {len(paths)} combined heatmaps to {args.output_dir}.")


if __name__ == "__main__":
    main()
