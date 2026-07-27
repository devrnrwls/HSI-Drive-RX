#!/usr/bin/env python3
"""Show integrated CORAL RX and every single-weather RX result at once."""

from __future__ import annotations

import argparse
import csv
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from domain_1_aligned_rx import ROAD_LABEL, RXModel, cube_to_hwc, load_label, load_manifest


def read_integrated_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    return dict(grouped)


def single_weather_models(results_dir: Path) -> dict[str, list[str]]:
    models: dict[str, list[str]] = {}
    for group_dir in results_dir.glob("group_*"):
        weather = sorted(
            path.name.removeprefix("single_weather_").removesuffix("_rx_background.npz")
            for path in group_dir.glob("single_weather_*_rx_background.npz")
        )
        if weather:
            models[group_dir.name.removeprefix("group_")] = weather
    return models


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    return np.clip((image - low) * 255.0 / max(high - low, 1e-12), 0, 255).astype(np.uint8)


def pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    hsi = cube_to_hwc(cube)
    indices = np.linspace(0, hsi.shape[-1] - 1, 3, dtype=int)[::-1]
    return np.stack([normalize_to_uint8(hsi[..., index]) for index in indices], axis=-1)


def score_colormap(scores: np.ndarray, road: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((scores - low) / max(high - low, 1e-12), 0.0, 1.0)
    output = np.zeros((*road.shape, 3), dtype=np.uint8)
    output[..., 0] = (255 * np.clip(1.5 * normalized - 0.5, 0, 1)).astype(np.uint8)
    output[..., 1] = (255 * np.clip(1.5 - np.abs(2 * normalized - 1.0) * 1.5, 0, 1)).astype(np.uint8)
    output[..., 2] = (255 * np.clip(1.0 - 1.5 * normalized, 0, 1)).astype(np.uint8)
    output[~road] = 20
    return output


def false_alarm_overlay(rgb: np.ndarray, road: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    detected = (scores > threshold) & road
    overlay = rgb.astype(np.float32).copy()
    overlay[~road] *= 0.30
    overlay[road] *= 0.72
    overlay[detected] = overlay[detected] * 0.20 + np.array([255, 0, 0]) * 0.80
    return np.clip(overlay, 0, 255).astype(np.uint8), float(detected.sum() / max(int(road.sum()), 1))


class RXComparisonViewer:
    def __init__(self, root: tk.Tk, dataset_dir: Path, integrated_dir: Path, single_dir: Path):
        self.root, self.dataset_dir = root, dataset_dir
        self.integrated_dir, self.single_dir = integrated_dir, single_dir
        integrated = read_integrated_rows(integrated_dir / "results.csv")
        single = single_weather_models(single_dir)
        self.rows_by_group = {group: rows for group, rows in integrated.items() if group in single}
        self.models_by_group = {group: single[group] for group in self.rows_by_group}
        if not self.rows_by_group:
            raise ValueError("No common completed groups found in domain-1 and domain-4 results.")
        samples, _ = load_manifest(dataset_dir)
        self.samples = {sample.base: sample for sample in samples}
        self.current_rows: list[dict[str, str]] = []

        root.title("RX Comparison: Integrated CORAL and All Training Weathers")
        root.geometry("1500x900")
        root.minsize(1150, 700)
        self._build_ui()
        self.group_var.set(sorted(self.rows_by_group)[0])
        self._load_group()

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        ttk.Label(left, text="Controlled group").grid(row=0, column=0, sticky="w")
        self.group_var = tk.StringVar()
        group_box = ttk.Combobox(left, textvariable=self.group_var, values=sorted(self.rows_by_group), state="readonly", width=28)
        group_box.grid(row=1, column=0, sticky="ew", pady=(3, 10))
        group_box.bind("<<ComboboxSelected>>", lambda _event: self._load_group())
        ttk.Label(left, text="Shared held-out test samples").grid(row=2, column=0, sticky="w")
        self.sample_list = tk.Listbox(left, width=40, height=21, exportselection=False)
        self.sample_list.grid(row=3, column=0, sticky="ns")
        self.sample_list.bind("<<ListboxSelect>>", lambda _event: self._load_selected_sample())
        ttk.Label(left, text="Test RGB").grid(row=4, column=0, sticky="w", pady=(12, 3))
        self.rgb_label = ttk.Label(left)
        self.rgb_label.grid(row=5, column=0, sticky="nw")
        self.info_var = tk.StringVar(value="Select a shared held-out image.")
        ttk.Label(left, textvariable=self.info_var, wraplength=300).grid(row=6, column=0, sticky="w", pady=(12, 0))

        container = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        container.grid(row=0, column=1, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.view = ttk.Frame(self.canvas)
        self.view.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.view, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.canvas_window, width=event.width))
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _load_group(self) -> None:
        group = self.group_var.get()
        self.current_rows = sorted(self.rows_by_group[group], key=lambda row: (row["weather"], row["sample"]))
        self.sample_list.delete(0, tk.END)
        for row in self.current_rows:
            self.sample_list.insert(tk.END, f"{row['sample']}  | test weather {row['weather']}")
        if self.current_rows:
            self.sample_list.selection_set(0)
            self._load_selected_sample()

    def _load_selected_sample(self) -> None:
        selected = self.sample_list.curselection()
        if not selected:
            return
        row = self.current_rows[selected[0]]
        sample = self.samples.get(row["sample"])
        if sample is None:
            messagebox.showerror("Missing input", f"Original sample not found: {row['sample']}")
            return
        score_path = Path(row["score_path"])
        if not score_path.is_absolute():
            score_path = Path.cwd() / score_path
        integrated_scores = np.load(score_path)
        integrated_data = np.load(self.integrated_dir / f"group_{row['group']}" / "rx_background.npz")
        cube = np.load(sample.cube_path)
        hsi = cube_to_hwc(cube).astype(np.float64, copy=False)
        label = load_label(sample.label_path)
        road = label == ROAD_LABEL
        rgb_path = self.dataset_dir / "RGB" / f"{sample.base.removesuffix('_RC_TC')}_pseudocolor.png"
        if rgb_path.is_file():
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"))
        else:
            rgb = pseudo_rgb(cube)
        comparisons: list[tuple[str, np.ndarray, float]] = [
            ("Integrated CORAL RX (train: all weather)", integrated_scores, float(integrated_data["train_score_p99"]))
        ]
        for weather in self.models_by_group[row["group"]]:
            saved = np.load(self.single_dir / f"group_{row['group']}" / f"single_weather_{weather}_rx_background.npz")
            model = RXModel(saved["mean"], saved["inv_cov"])
            scores = np.full(label.shape, np.nan, dtype=np.float64)
            scores[road] = model.score(hsi[road])
            comparisons.append((f"Single-weather RX (train: weather {weather})", scores, float(saved["train_score_p99"])))
        self._show(self.rgb_label, rgb, (300, 210))
        self.info_var.set(
            f"Test weather: {row['weather']}\n"
            f"Showing integrated RX plus {len(comparisons) - 1} single-weather models.\n"
            "All score maps share one 1st–99th percentile color range; red overlays use each model's train p99."
        )
        self._render_comparisons(comparisons, rgb, road)

    def _render_comparisons(self, comparisons: list[tuple[str, np.ndarray, float]], rgb: np.ndarray, road: np.ndarray) -> None:
        for child in self.view.winfo_children():
            child.destroy()
        self.view.columnconfigure(1, weight=1)
        self.view.columnconfigure(2, weight=1)
        for col, title in enumerate(("Detector", "Score map (shared scale)", "False alarms at train p99", "Image FAR")):
            ttk.Label(self.view, text=title).grid(row=0, column=col, sticky="w", padx=8, pady=(0, 5))
        values = np.concatenate([scores[road] for _, scores, _ in comparisons])
        low, high = np.percentile(values, (1, 99))
        for index, (name, scores, threshold) in enumerate(comparisons, start=1):
            score_label, overlay_label = ttk.Label(self.view), ttk.Label(self.view)
            overlay, far = false_alarm_overlay(rgb, road, scores, threshold)
            ttk.Label(self.view, text=name, wraplength=180).grid(row=index, column=0, sticky="nw", padx=8, pady=8)
            score_label.grid(row=index, column=1, sticky="nw", padx=8, pady=8)
            overlay_label.grid(row=index, column=2, sticky="nw", padx=8, pady=8)
            ttk.Label(self.view, text=f"p99: {threshold:.3f}\nFAR: {far:.3%}").grid(row=index, column=3, sticky="nw", padx=8, pady=8)
            self._show(score_label, score_colormap(scores, road, low, high), (330, 230))
            self._show(overlay_label, overlay, (330, 230))
        self.canvas.yview_moveto(0)

    @staticmethod
    def _show(label: ttk.Label, rgb: np.ndarray, size: tuple[int, int]) -> None:
        image = Image.fromarray(rgb)
        image.thumbnail(size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("HSI_Drive"))
    parser.add_argument("--integrated-results-dir", type=Path, default=Path("domain_aligned_rx_results"))
    parser.add_argument("--single-weather-results-dir", type=Path, default=Path("single_weather_rx_comparison_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    try:
        RXComparisonViewer(root, args.dataset_dir, args.integrated_results_dir, args.single_weather_results_dir)
    except Exception:
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
