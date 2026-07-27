#!/usr/bin/env python3
"""Interactive normal-Road false-alarm viewer for a trained aligned RX group.

Only held-out normal ``nf`` images listed as ``ok`` in ``results.csv`` are
shown.  This keeps the interactive inspection separate from images used to
estimate the RX background.
"""

from __future__ import annotations

import argparse
import csv
import json
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from domain_1_aligned_rx import CoralTransform, RXModel, ROAD_LABEL, cube_to_hwc, load_label, load_manifest


def completed_rows(results_path: Path) -> dict[str, list[dict[str, str]]]:
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    return dict(grouped)


def load_group_model(group_dir: Path) -> tuple[dict[str, CoralTransform], RXModel, float]:
    alignment = np.load(group_dir / "coral_alignment.npz")
    rx_data = np.load(group_dir / "rx_background.npz")
    transforms: dict[str, CoralTransform] = {}
    for key in alignment.files:
        if key.startswith("mean_weather_"):
            weather = key.removeprefix("mean_weather_")
            transforms[weather] = CoralTransform(
                mean=alignment[key],
                whitening=alignment[f"whitening_weather_{weather}"],
                recoloring=alignment["recoloring"],
                target_mean=alignment["target_mean"],
            )
    return transforms, RXModel(rx_data["mean"], rx_data["inv_cov"]), float(rx_data["train_score_p99"])


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        high = low + 1.0
    return np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    hsi = cube_to_hwc(cube)
    indices = np.linspace(0, hsi.shape[-1] - 1, 3, dtype=int)[::-1]
    return np.stack([normalize_to_uint8(hsi[..., index]) for index in indices], axis=-1)


def score_colormap(scores: np.ndarray, road: np.ndarray) -> np.ndarray:
    output = np.zeros((*road.shape, 3), dtype=np.uint8)
    values = scores[road]
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        high = low + 1.0
    normalized = np.clip((scores - low) / (high - low), 0.0, 1.0)
    # Compact blue -> cyan -> yellow -> red heatmap without OpenCV/matplotlib.
    output[..., 0] = (255 * np.clip(1.5 * normalized - 0.5, 0, 1)).astype(np.uint8)
    output[..., 1] = (255 * np.clip(1.5 - np.abs(2 * normalized - 1.0) * 1.5, 0, 1)).astype(np.uint8)
    output[..., 2] = (255 * np.clip(1.0 - 1.5 * normalized, 0, 1)).astype(np.uint8)
    output[~road] = 20
    return output


class NormalRoadRXViewer:
    def __init__(self, root: tk.Tk, dataset_dir: Path, results_dir: Path):
        self.root = root
        self.dataset_dir = dataset_dir
        self.results_dir = results_dir
        self.rows_by_group = completed_rows(results_dir / "results.csv")
        if not self.rows_by_group:
            raise ValueError("No completed normal-Road results found. Run domain_aligned_rx.py first.")
        samples, _ = load_manifest(dataset_dir)
        self.samples = {sample.base: sample for sample in samples}
        self.samples_by_group: dict[str, list[object]] = defaultdict(list)
        for sample in samples:
            self.samples_by_group[sample.group].append(sample)
        self.group_split: dict[str, list[dict[str, str]]] = {"train": [], "test": []}
        self.current_items: list[dict[str, str]] = []
        self.transforms: dict[str, CoralTransform] = {}
        self.model: RXModel | None = None
        self.train_p99 = 0.0
        self.rgb: np.ndarray | None = None
        self.road: np.ndarray | None = None
        self.scores: np.ndarray | None = None

        self.root.title("Domain-Aligned RX: Normal Road False-Alarm Viewer")
        self.root.geometry("1300x820")
        self.root.minsize(1050, 680)
        self._build_ui()
        first_group = sorted(self.rows_by_group)[0]
        self.group_var.set(first_group)
        self._load_group()

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="ns")

        ttk.Label(left, text="Controlled group").grid(row=0, column=0, sticky="w")
        self.group_var = tk.StringVar()
        group_box = ttk.Combobox(left, textvariable=self.group_var, values=sorted(self.rows_by_group), state="readonly", width=24)
        group_box.grid(row=1, column=0, sticky="ew", pady=(3, 10))
        group_box.bind("<<ComboboxSelected>>", lambda _event: self._load_group())

        ttk.Label(left, text="Held-out normal nf samples").grid(row=2, column=0, sticky="w")
        self.sample_list = tk.Listbox(left, width=34, height=24, exportselection=False)
        self.sample_list.grid(row=3, column=0, sticky="ns")
        self.sample_list.bind("<<ListboxSelect>>", lambda _event: self._load_sample())

        split_box = ttk.LabelFrame(left, text="Show images")
        split_box.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self.show_train_var = tk.BooleanVar(value=False)
        self.show_test_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(split_box, text="Train", variable=self.show_train_var, command=self._refresh_sample_list).grid(
            row=0, column=0, sticky="w", padx=8, pady=4
        )
        ttk.Checkbutton(split_box, text="Test", variable=self.show_test_var, command=self._refresh_sample_list).grid(
            row=0, column=1, sticky="w", padx=8, pady=4
        )

        threshold_box = ttk.LabelFrame(left, text="RX threshold")
        threshold_box.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        self.threshold_var = tk.DoubleVar()
        self.threshold_label = ttk.Label(threshold_box, text="")
        self.threshold_label.grid(row=0, column=0, sticky="w", padx=8, pady=(7, 2))
        self.threshold_scale = tk.Scale(
            threshold_box, from_=0, to=1, resolution=0.01, orient="horizontal", variable=self.threshold_var,
            command=lambda _value: self._update_threshold_view(), length=245,
        )
        self.threshold_scale.grid(row=1, column=0, padx=4, pady=(0, 7))
        ttk.Button(threshold_box, text="Reset to train p99", command=self._reset_threshold).grid(
            row=2, column=0, sticky="ew", padx=8, pady=(0, 7)
        )
        self.info_var = tk.StringVar(value="Select a held-out normal image.")
        ttk.Label(left, textvariable=self.info_var, wraplength=275).grid(row=6, column=0, sticky="w", pady=(12, 0))

        main = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure((0, 1), weight=1)
        main.rowconfigure((1, 3), weight=1)
        for row, text in ((0, "Normal test RGB"), (2, "Road ROI / false-alarm overlay")):
            ttk.Label(main, text=text).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 12, 3))
        ttk.Label(main, text="Aligned RX score map").grid(row=0, column=1, sticky="w", pady=(0, 3))
        ttk.Label(main, text="Thresholded anomaly decision").grid(row=2, column=1, sticky="w", pady=(12, 3))
        self.rgb_label = ttk.Label(main)
        self.score_label = ttk.Label(main)
        self.overlay_label = ttk.Label(main)
        self.mask_label = ttk.Label(main)
        self.rgb_label.grid(row=1, column=0, sticky="nw", padx=(0, 10))
        self.score_label.grid(row=1, column=1, sticky="nw")
        self.overlay_label.grid(row=3, column=0, sticky="nw", padx=(0, 10))
        self.mask_label.grid(row=3, column=1, sticky="nw")

    def _load_group(self) -> None:
        group = self.group_var.get()
        self.transforms, self.model, self.train_p99 = load_group_model(self.results_dir / f"group_{group}")
        split_path = self.results_dir / f"group_{group}" / "split.json"
        if split_path.is_file():
            self.group_split = json.loads(split_path.read_text(encoding="utf-8"))
        else:
            # Compatibility with results produced before split.json was added:
            # test rows are known, so remaining normal images in the same group
            # and tested weather domains are precisely the original train set.
            test_rows = self.rows_by_group[group]
            test_bases = {row["sample"] for row in test_rows}
            tested_weather = {row["weather"] for row in test_rows}
            train_rows = [
                {"sample": sample.base, "condition": sample.condition, "weather": sample.weather}
                for sample in self.samples_by_group[group]
                if sample.weather in tested_weather and sample.base not in test_bases
            ]
            self.group_split = {"train": train_rows, "test": test_rows}
        self._refresh_sample_list()

    def _refresh_sample_list(self) -> None:
        self.current_items = []
        if self.show_train_var.get():
            self.current_items.extend({**row, "split": "train"} for row in self.group_split.get("train", []))
        if self.show_test_var.get():
            self.current_items.extend({**row, "split": "test"} for row in self.group_split.get("test", []))
        self.current_items.sort(key=lambda row: (row["split"], row["sample"]))
        self.sample_list.delete(0, tk.END)
        for row in self.current_items:
            self.sample_list.insert(tk.END, f"[{row['split'].upper()}] {row['sample']}  | weather {row['weather']}")
        if self.current_items:
            self.sample_list.selection_set(0)
            self._load_sample()
        else:
            self.info_var.set("Select Train and/or Test to display images.")

    def _load_sample(self) -> None:
        selected = self.sample_list.curselection()
        if not selected or self.model is None:
            return
        row = self.current_items[selected[0]]
        sample = self.samples.get(row["sample"])
        if sample is None:
            messagebox.showerror("Missing input", f"Original sample not found: {row['sample']}")
            return
        cube = np.load(sample.cube_path)
        hsi = cube_to_hwc(cube).astype(np.float64, copy=False)
        label = load_label(sample.label_path)
        self.road = label == ROAD_LABEL
        features = hsi[self.road]
        self.scores = np.full(label.shape, np.nan, dtype=np.float64)
        self.scores[self.road] = self.model.score(self.transforms[sample.weather].apply(features))
        rgb_path = self.dataset_dir / "RGB" / f"{sample.base.removesuffix('_RC_TC')}_pseudocolor.png"
        if rgb_path.is_file():
            with Image.open(rgb_path) as image:
                self.rgb = np.asarray(image.convert("RGB"))
        else:
            self.rgb = pseudo_rgb(cube)
        upper = max(float(np.nanpercentile(self.scores, 99.9)) * 1.2, self.train_p99 * 1.5, 1.0)
        self.threshold_scale.configure(from_=0.0, to=upper, resolution=upper / 500.0)
        self._reset_threshold()
        self._show(self.rgb_label, self.rgb)
        self._show(self.score_label, score_colormap(self.scores, self.road))

    def _reset_threshold(self) -> None:
        self.threshold_var.set(self.train_p99)
        self._update_threshold_view()

    def _update_threshold_view(self) -> None:
        if self.rgb is None or self.road is None or self.scores is None:
            return
        threshold = self.threshold_var.get()
        detected = (self.scores > threshold) & self.road
        overlay = self.rgb.astype(np.float32).copy()
        overlay[~self.road] *= 0.30
        overlay[self.road] *= 0.72
        overlay[detected] = overlay[detected] * 0.20 + np.array([255, 0, 0]) * 0.80
        mask = np.zeros((*self.road.shape, 3), dtype=np.uint8)
        mask[self.road] = [42, 42, 42]
        mask[detected] = [255, 0, 0]
        count = int(detected.sum())
        road_count = int(self.road.sum())
        rate = count / road_count if road_count else 0.0
        self.threshold_label.config(text=f"Threshold: {threshold:.3f}  (train p99: {self.train_p99:.3f})")
        self.info_var.set(
            f"Split: {self.current_items[self.sample_list.curselection()[0]]['split'].upper()}\n"
            f"Detected normal Road pixels: {count:,}/{road_count:,}\nFalse-alarm rate: {rate:.3%}"
        )
        self._show(self.overlay_label, np.clip(overlay, 0, 255).astype(np.uint8))
        self._show(self.mask_label, mask)

    @staticmethod
    def _show(label: ttk.Label, rgb: np.ndarray) -> None:
        image = Image.fromarray(rgb)
        image.thumbnail((470, 310), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("HSI_Drive"))
    parser.add_argument("--results-dir", type=Path, default=Path("domain_aligned_rx_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    try:
        NormalRoadRXViewer(root, args.dataset_dir, args.results_dir)
    except Exception as exc:
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
