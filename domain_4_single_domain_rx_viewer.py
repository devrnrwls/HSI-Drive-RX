#!/usr/bin/env python3
"""Inspect domain-4 single-domain RX scores on held-out normal Road images.

Choose a controlled group and the weather domain used to train the RX
background.  The viewer then applies that exact saved model to every held-out
test image, making weather-domain shift visible at pixel level.
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from domain_1_aligned_rx import RXModel, ROAD_LABEL, cube_to_hwc, load_label, load_manifest, parse_domain_factor
from domain_4_aligned_rx_viewer import pseudo_rgb, score_colormap


def available_groups(results_dir: Path, factor: str) -> dict[str, list[str]]:
    """Return group IDs and their saved single-domain training domains."""
    groups: dict[str, list[str]] = {}
    prefix = "single_weather" if factor == "weather" else f"single_{factor}"
    for group_dir in sorted(results_dir.glob("group_*")):
        weather = sorted(
            path.name.removeprefix(f"{prefix}_").removesuffix("_rx_background.npz")
            for path in group_dir.glob(f"{prefix}_*_rx_background.npz")
        )
        if weather and (group_dir / "split.json").is_file():
            groups[group_dir.name.removeprefix("group_")] = weather
    return groups


class SingleWeatherRXViewer:
    def __init__(self, root: tk.Tk, dataset_dir: Path, results_root: Path, domain_factor: str):
        self.root = root
        self.dataset_dir = dataset_dir
        self.results_root = results_root
        self.results_dir = results_root / domain_factor
        self.domain_factor = domain_factor
        samples, _ = load_manifest(dataset_dir)
        self.samples = {sample.base: sample for sample in samples}
        self.test_rows: list[dict[str, str]] = []
        self.model: RXModel | None = None
        self.train_p99 = 0.0
        self.rgb: np.ndarray | None = None
        self.road: np.ndarray | None = None
        self.scores: np.ndarray | None = None

        self.root.title(f"Domain 2: Single-Domain RX Viewer | Factor: {domain_factor}")
        self.root.geometry("1300x820")
        self.root.minsize(1050, 680)
        self._build_ui()
        self._switch_factor()

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        ttk.Label(left, text="Domain factor").grid(row=0, column=1, sticky="w")
        self.factor_var = tk.StringVar(value=self.domain_factor)
        factor_box = ttk.Combobox(left, textvariable=self.factor_var, values=("season", "weather", "daytime", "roadtype"), state="readonly", width=12)
        factor_box.grid(row=1, column=1, sticky="ew", pady=(3, 10))
        factor_box.bind("<<ComboboxSelected>>", lambda _event: self._switch_factor())
        self.selected_factor_label = ttk.Label(left, text=f"Selected factor: {self.domain_factor}", font=("TkDefaultFont", 10, "bold"))
        self.selected_factor_label.grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        ttk.Label(left, text="Controlled group").grid(row=1, column=0, sticky="w")
        self.group_var = tk.StringVar()
        self.group_box = ttk.Combobox(left, textvariable=self.group_var, values=(), state="readonly", width=24)
        self.group_box.grid(row=2, column=0, sticky="ew", pady=(3, 10))
        self.group_box.bind("<<ComboboxSelected>>", lambda _event: self._load_group())

        self.train_domain_label = ttk.Label(left, text=f"RX training {self.domain_factor}")
        self.train_domain_label.grid(row=3, column=0, sticky="w")
        self.train_weather_var = tk.StringVar()
        self.train_weather_box = ttk.Combobox(left, textvariable=self.train_weather_var, state="readonly", width=24)
        self.train_weather_box.grid(row=4, column=0, sticky="ew", pady=(3, 10))
        self.train_weather_box.bind("<<ComboboxSelected>>", lambda _event: self._load_model())

        ttk.Label(left, text="Held-out normal nf samples").grid(row=5, column=0, sticky="w")
        self.sample_list = tk.Listbox(left, width=34, height=22, exportselection=False)
        self.sample_list.grid(row=6, column=0, sticky="ns")
        self.sample_list.bind("<<ListboxSelect>>", lambda _event: self._load_sample())

        threshold_box = ttk.LabelFrame(left, text="RX threshold")
        threshold_box.grid(row=7, column=0, sticky="ew", pady=(12, 0))
        self.threshold_var = tk.DoubleVar()
        self.threshold_label = ttk.Label(threshold_box, text="")
        self.threshold_label.grid(row=0, column=0, sticky="w", padx=8, pady=(7, 2))
        self.threshold_scale = tk.Scale(
            threshold_box, from_=0, to=1, resolution=0.01, orient="horizontal", variable=self.threshold_var,
            command=lambda _value: self._update_threshold_view(), length=245,
        )
        self.threshold_scale.grid(row=1, column=0, padx=4, pady=(0, 7))
        ttk.Button(threshold_box, text="Reset to source train p99", command=self._reset_threshold).grid(
            row=2, column=0, sticky="ew", padx=8, pady=(0, 7)
        )
        self.info_var = tk.StringVar(value="Select a held-out normal image.")
        ttk.Label(left, textvariable=self.info_var, wraplength=275).grid(row=8, column=0, sticky="w", pady=(12, 0))

        main = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure((0, 1), weight=1)
        main.rowconfigure((1, 3), weight=1)
        for row, text in ((0, "Normal test RGB"), (2, "Road ROI / false-alarm overlay")):
            ttk.Label(main, text=text).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 12, 3))
        self.score_map_heading = ttk.Label(main, text=f"Single-{self.domain_factor} RX score map")
        self.score_map_heading.grid(row=0, column=1, sticky="w", pady=(0, 3))
        ttk.Label(main, text="Thresholded anomaly decision").grid(row=2, column=1, sticky="w", pady=(12, 3))
        self.rgb_label, self.score_label = ttk.Label(main), ttk.Label(main)
        self.overlay_label, self.mask_label = ttk.Label(main), ttk.Label(main)
        self.rgb_label.grid(row=1, column=0, sticky="nw", padx=(0, 10))
        self.score_label.grid(row=1, column=1, sticky="nw")
        self.overlay_label.grid(row=3, column=0, sticky="nw", padx=(0, 10))
        self.mask_label.grid(row=3, column=1, sticky="nw")

    def _load_group(self) -> None:
        group = self.group_var.get()
        weather = self.groups[group]
        self.train_weather_box.configure(values=weather)
        self.train_weather_var.set(weather[0])
        split = json.loads((self.results_dir / f"group_{group}" / "split.json").read_text(encoding="utf-8"))
        self.test_rows = split["test"]
        self.sample_list.delete(0, tk.END)
        for row in self.test_rows:
            self.sample_list.insert(tk.END, f"{row['sample']}  | test {self.domain_factor} {row['domain']}")
        self._load_model()
        if self.test_rows:
            self.sample_list.selection_set(0)
            self._load_sample()

    def _switch_factor(self) -> None:
        self.domain_factor = self.factor_var.get()
        self.root.title(f"Domain 3: Single-Domain RX Viewer | Factor: {self.domain_factor}")
        self.selected_factor_label.config(text=f"Selected factor: {self.domain_factor}")
        self.train_domain_label.config(text=f"RX training {self.domain_factor}")
        self.score_map_heading.config(text=f"Single-{self.domain_factor} RX score map")
        self.results_dir = self.results_root / self.domain_factor
        self.groups = available_groups(self.results_dir, self.domain_factor)
        if not self.groups:
            self.group_var.set("")
            self.sample_list.delete(0, tk.END)
            self.info_var.set(f"No completed {self.domain_factor} results found.")
            return
        self.group_box.configure(values=sorted(self.groups))
        self.group_var.set(sorted(self.groups)[0])
        self._load_group()

    def _load_model(self) -> None:
        group = self.group_var.get()
        weather = self.train_weather_var.get()
        prefix = "single_weather" if self.domain_factor == "weather" else f"single_{self.domain_factor}"
        path = self.results_dir / f"group_{group}" / f"{prefix}_{weather}_rx_background.npz"
        saved = np.load(path)
        self.model = RXModel(saved["mean"], saved["inv_cov"])
        self.train_p99 = float(saved["train_score_p99"])
        self._load_sample()

    def _load_sample(self) -> None:
        selected = self.sample_list.curselection()
        if not selected or self.model is None:
            return
        row = self.test_rows[selected[0]]
        sample = self.samples.get(row["sample"])
        if sample is None:
            messagebox.showerror("Missing input", f"Original sample not found: {row['sample']}")
            return
        cube = np.load(sample.cube_path)
        hsi = cube_to_hwc(cube).astype(np.float64, copy=False)
        label = load_label(sample.label_path)
        self.road = label == ROAD_LABEL
        self.scores = np.full(label.shape, np.nan, dtype=np.float64)
        self.scores[self.road] = self.model.score(hsi[self.road])
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
        count, road_count = int(detected.sum()), int(self.road.sum())
        rate = count / road_count if road_count else 0.0
        self.threshold_label.config(text=f"Threshold: {threshold:.3f}  (source train p99: {self.train_p99:.3f})")
        self.info_var.set(
            f"Train {self.domain_factor}: {self.train_weather_var.get()} | Test {self.domain_factor}: {self.test_rows[self.sample_list.curselection()[0]]['domain']}\n"
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
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--domain-factor", type=parse_domain_factor, default="weather", metavar="FACTOR")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path("domain_2_single_domain_rx_results") if args.results_dir is None else args.results_dir.parent
    root = tk.Tk()
    try:
        SingleWeatherRXViewer(root, args.dataset_dir, results_root, args.domain_factor)
    except Exception:
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
