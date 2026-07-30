#!/usr/bin/env python3
"""Run domain RX experiments for test fractions from 10% through 90%.

Each fraction is isolated under ``test_fraction_sweep/test_XXpct/``.  Domain
1 and domain 2 run every factor; domain 3 then writes one overall summary per
factor. Per-image score maps, including the optional pooled-RX baseline maps,
are disabled by default to keep the sweep manageable; alignment and aggregate
visualizations are generated.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FACTORS = ("season", "weather", "daytime", "roadtype")
FRACTIONS = tuple(index / 10 for index in range(1, 10))


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


def parse_fraction(value: str) -> float:
    """Accept 0.2, 20, or 20% as a test fraction."""
    text = value.strip()
    if text.endswith("%"):
        text = text[:-1]
    try:
        fraction = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid test fraction: {value}") from exc
    if fraction > 1:
        fraction /= 100
    if not 0 < fraction < 1:
        raise argparse.ArgumentTypeError("Test fraction must be between 0 and 1, or between 1 and 100 as a percentage.")
    return fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("HSI_Drive"))
    parser.add_argument("--output-root", type=Path, default=Path("test_fraction_sweep"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fractions", nargs="+", type=parse_fraction, metavar="FRACTION", help="Run only these fractions; accepts 0.2, 20, or 20%%.")
    parser.add_argument("--score-map-plots", action="store_true", help="Also generate per-image normal_score_maps PNGs.")
    parser.add_argument(
        "--visualize-pooled-rx",
        action="store_true",
        help="Also generate full-resolution pooled-RX score maps in domain 1 and domain 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).parent
    domain_1 = project_dir / "domain_1_aligned_rx.py"
    domain_2 = project_dir / "domain_2_compare_single_domain_rx.py"
    domain_3 = project_dir / "domain_3_overall_rx_comparison.py"
    completed: list[Path] = []

    fractions = args.test_fractions if args.test_fractions else FRACTIONS
    for fraction in fractions:
        label = f"test_{round(fraction * 100):02d}pct"
        trial_dir = args.output_root / label
        integrated_dir = trial_dir / "domain_1_aligned_rx_results"
        single_dir = trial_dir / "domain_2_single_domain_rx_results"
        overall_dir = trial_dir / "domain_3_overall_rx_comparisons"
        common = ["--dataset-dir", str(args.dataset_dir), "--test-fraction", str(fraction), "--seed", str(args.seed)]

        print(f"\n{'=' * 18} test fraction {fraction:.0%} {'=' * 18}")
        command_1 = [sys.executable, str(domain_1), *common, "--all-domain-factors", "--output-dir", str(integrated_dir)]
        if not args.score_map_plots:
            command_1.append("--no-score-map-plots")
        if args.visualize_pooled_rx:
            command_1.append("--visualize-pooled-rx")
        run(command_1)
        command_2 = [sys.executable, str(domain_2), *common, "--all-domain-factors", "--output-dir", str(single_dir)]
        if args.visualize_pooled_rx:
            command_2.append("--visualize-pooled-rx")
        run(command_2)

        for factor in FACTORS:
            run([
                sys.executable, str(domain_3),
                "--domain-factor", factor,
                "--results-dir", str(single_dir / factor),
                "--output-dir", str(overall_dir / factor),
            ])
        completed.append(trial_dir)

    print("\n=== Completed test-fraction sweep ===")
    for path in completed:
        print(path.resolve())


if __name__ == "__main__":
    main()
