#!/usr/bin/env python3
import argparse
import sys
import types
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np


if "utils.simulation_core" not in sys.modules:
    utils_module = types.ModuleType("utils")
    simulation_core_module = types.ModuleType("utils.simulation_core")

    class Detector:
        pass

    simulation_core_module.Detector = Detector
    utils_module.simulation_core = simulation_core_module
    sys.modules.setdefault("utils", utils_module)
    sys.modules["utils.simulation_core"] = simulation_core_module

from detectors import cdlss_ad, erx, erx_ablation, lbl_ad, rt_ck_rxd, rx_baseline, rx_bil


ROAD_LABEL = 1
DEFAULT_DATASET_DIR = "Generated_Anomaly_Dataset"
DEFAULT_BUFFER_LEN = 99

DETECTOR_CHOICES = {
    "rx_baseline": "RX Baseline",
    "rt_ck_rxd": "RT-CK-RXD",
    "rx_bil": "RX-BIL",
    "cdlss_ad": "CDLSS-AD",
    "lbl_ad": "Lbl-AD",
    "erx": "ERX",
    "erx_ablation": "ERX Ablation",
    "erx_r": "ERX-R",
}

DETECTOR_DEFAULTS = {
    "rx_baseline": {"threshold": 1.5},
    "rt_ck_rxd": {"threshold": 1.5},
    "rx_bil": {"threshold": 1.5, "pixel_dropout": 0.5},
    "cdlss_ad": {"threshold": 1.5},
    "lbl_ad": {"threshold": 1.5, "pca_dims": 3},
    "erx": {"threshold": 1.5, "projected_dimensions": 5, "momentum": 0.1},
    "erx_ablation": {"threshold": 1.5, "projected_dimensions": 10, "momentum": 0.01},
    "erx_r": {"threshold": 1.5, "momentum": 0.01},
}


class GeneratedAnomalyRXDetector:
    def __init__(self, root, dataset_dir):
        self.root = root
        self.dataset_dir = Path(dataset_dir)
        self.cube_dir = self.dataset_dir / "Cubes_Scaling"
        self.rgb_dir = self.dataset_dir / "RGB"
        self.target_label_dir = self.dataset_dir / "Target_Labels"
        self.anomaly_label_dir = self.dataset_dir / "Labels"
        self.result_dir = self.dataset_dir / "Detection_Results"

        self.cube = None
        self.rgb = None
        self.target_label = None
        self.anomaly_label = None
        self.current_base = None

        self.files = self._find_cube_files()
        self.displayed_files = list(self.files)

        self.root.title("Generated Anomaly Detector")
        self.root.geometry("1280x820")
        self.root.minsize(1120, 720)
        self._build_ui()
        if self.files:
            first_idx = self._first_loadable_file_index()
            self.file_list.selection_set(first_idx)
            self.file_list.event_generate("<<ListboxSelect>>")

    def _find_cube_files(self):
        if not self.cube_dir.is_dir():
            raise FileNotFoundError(f"Cube directory not found: {self.cube_dir}")
        files = sorted(self.cube_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files found in {self.cube_dir}")
        return files

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.rowconfigure(2, weight=1)

        ttk.Label(left, text="Generated Cubes_Scaling files").grid(row=0, column=0, sticky="w")
        self.search_var = tk.StringVar()
        search = ttk.Entry(left, textvariable=self.search_var, width=40)
        search.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        search.bind("<KeyRelease>", self._filter_files)

        list_frame = ttk.Frame(left)
        list_frame.grid(row=2, column=0, sticky="ns")
        self.file_list = tk.Listbox(list_frame, width=42, height=30, exportselection=False)
        self.file_list.grid(row=0, column=0, sticky="ns")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_selected)
        self._refresh_listbox()

        ttk.Button(left, text="파일 리스트 업데이트", command=self._reload_file_list).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        detect_box = ttk.LabelFrame(left, text="Anomaly Detection")
        detect_box.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self.threshold_var = tk.StringVar(value="1.5")
        self.normalise_var = tk.BooleanVar(value=True)
        self.buffer_len_var = tk.StringVar(value=str(DEFAULT_BUFFER_LEN))
        self.detector_vars = {
            name: tk.BooleanVar(value=(name == "rx_baseline"))
            for name in DETECTOR_CHOICES
        }
        ttk.Label(detect_box, text="Algorithms").grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 2))
        for row, (name, label) in enumerate(DETECTOR_CHOICES.items(), start=1):
            ttk.Checkbutton(
                detect_box,
                text=label,
                variable=self.detector_vars[name],
                command=lambda selected=name: self._select_detector(selected),
            ).grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=1)
        param_row = len(DETECTOR_CHOICES) + 1
        ttk.Label(detect_box, text="Threshold").grid(row=param_row, column=0, sticky="w", padx=8, pady=(6, 2))
        ttk.Entry(detect_box, textvariable=self.threshold_var, width=10).grid(
            row=param_row, column=1, sticky="w", pady=(6, 2)
        )
        ttk.Label(detect_box, text="Buffer lines").grid(row=param_row + 1, column=0, sticky="w", padx=8, pady=2)
        ttk.Entry(detect_box, textvariable=self.buffer_len_var, width=10).grid(
            row=param_row + 1, column=1, sticky="w", pady=2
        )
        ttk.Checkbutton(
            detect_box,
            text="Normalise line scores",
            variable=self.normalise_var,
        ).grid(row=param_row + 2, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        ttk.Button(detect_box, text="Anomaly 검출", command=self._detect).grid(
            row=param_row + 3, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 6)
        )

        self.status_var = tk.StringVar(value=f"{len(self.files)} files")
        ttk.Label(left, textvariable=self.status_var, wraplength=310).grid(row=5, column=0, sticky="w", pady=(10, 0))

        main = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)
        main.rowconfigure(3, weight=1)

        ttk.Label(main, text="Generated RGB").grid(row=0, column=0, sticky="w")
        ttk.Label(main, text="Target road ROI").grid(row=0, column=1, sticky="w")
        self.rgb_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.roi_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.rgb_canvas.grid(row=1, column=0, sticky="nw", padx=(0, 12), pady=(6, 10))
        self.roi_canvas.grid(row=1, column=1, sticky="nw", pady=(6, 10))

        ttk.Label(main, text="Score map").grid(row=2, column=0, sticky="w")
        ttk.Label(main, text="Detection overlay").grid(row=2, column=1, sticky="w")
        self.score_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.detect_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.score_canvas.grid(row=3, column=0, sticky="nw", padx=(0, 12), pady=(6, 10))
        self.detect_canvas.grid(row=3, column=1, sticky="nw", pady=(6, 10))

        self.info_var = tk.StringVar(value="Select a generated cube file, then run Anomaly detection.")
        ttk.Label(main, textvariable=self.info_var, wraplength=900).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 6),
        )

    def _refresh_listbox(self):
        self.file_list.delete(0, tk.END)
        for path in self.displayed_files:
            self.file_list.insert(tk.END, path.name)

    def _filter_files(self, _event=None):
        query = self.search_var.get().strip().lower()
        self.displayed_files = [p for p in self.files if query in p.name.lower()] if query else list(self.files)
        self._refresh_listbox()
        self.status_var.set(f"{len(self.displayed_files)} / {len(self.files)} files")

    def _reload_file_list(self):
        try:
            self.files = self._find_cube_files()
            query = self.search_var.get().strip().lower()
            self.displayed_files = [p for p in self.files if query in p.name.lower()] if query else list(self.files)
            self._refresh_listbox()
            if self.displayed_files:
                first_idx = self._first_loadable_file_index()
                self.file_list.selection_clear(0, tk.END)
                self.file_list.selection_set(first_idx)
                self.file_list.see(first_idx)
                self._load_sample(self.displayed_files[first_idx])
            self.status_var.set(f"Updated file list: {len(self.displayed_files)} / {len(self.files)} files")
        except Exception as exc:
            messagebox.showerror("Update error", str(exc))

    def _base_from_cube(self, cube_path):
        name = cube_path.name
        if name.endswith("_RC_TC.npy"):
            return name.removesuffix("_RC_TC.npy")
        return cube_path.stem

    def _first_loadable_file_index(self):
        for idx, path in enumerate(self.displayed_files):
            base = self._base_from_cube(path)
            if (self.target_label_dir / f"{base}.png").is_file():
                return idx
        return 0

    def _single_channel_label(self, label):
        if label.ndim == 3:
            return label[:, :, 0]
        return label

    def _on_file_selected(self, _event=None):
        selected = self.file_list.curselection()
        if not selected:
            return
        try:
            self._load_sample(self.displayed_files[selected[0]])
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _load_sample(self, cube_path):
        base = self._base_from_cube(cube_path)
        rgb_path = self.rgb_dir / f"{base}_pseudocolor.png"
        target_label_path = self.target_label_dir / f"{base}.png"
        anomaly_label_path = self.anomaly_label_dir / f"{base}.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        if not target_label_path.is_file():
            raise FileNotFoundError(f"Target label file not found: {target_label_path}")

        cube = np.load(cube_path)
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        target_label = cv2.imread(str(target_label_path), cv2.IMREAD_UNCHANGED)
        anomaly_label = cv2.imread(str(anomaly_label_path), cv2.IMREAD_UNCHANGED) if anomaly_label_path.is_file() else None

        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube, got {cube.shape}: {cube_path}")
        if rgb_bgr is None:
            raise ValueError(f"Failed to read RGB file: {rgb_path}")
        if target_label is None:
            raise ValueError(f"Failed to read target label file: {target_label_path}")
        target_label = self._single_channel_label(target_label)
        if anomaly_label is not None:
            anomaly_label = self._single_channel_label(anomaly_label)

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        height, width = self._cube_hw(cube)
        if rgb.shape[:2] != (height, width) or target_label.shape[:2] != (height, width):
            raise ValueError(f"Shape mismatch: cube={cube.shape}, RGB={rgb.shape}, target_label={target_label.shape}")
        if anomaly_label is not None and anomaly_label.shape[:2] != (height, width):
            raise ValueError(f"Anomaly label shape mismatch: {anomaly_label.shape}, expected {(height, width)}")

        self.current_base = base
        self.cube = cube
        self.rgb = rgb
        self.target_label = target_label
        self.anomaly_label = anomaly_label

        roi_mask = self.target_label == ROAD_LABEL
        self._show_image(self.rgb_canvas, rgb)
        self._show_image(self.roi_canvas, self._roi_overlay(rgb, roi_mask))
        self.score_canvas.delete("all")
        self.detect_canvas.delete("all")

        roi_pixels = int(np.count_nonzero(roi_mask))
        anomaly_pixels = int(np.count_nonzero(anomaly_label == 1)) if anomaly_label is not None else 0
        self.info_var.set(
            f"{base} | cube {cube.shape} {cube.dtype} | road ROI pixels={roi_pixels} | "
            f"ground-truth anomaly pixels={anomaly_pixels}"
        )
        self.status_var.set(f"Loaded {base}")

    def _detect(self):
        if self.cube is None:
            messagebox.showinfo("Load required", "Select a generated cube file first.")
            return
        try:
            threshold = float(self.threshold_var.get())
            buffer_len = int(self.buffer_len_var.get())
            if buffer_len <= 0:
                raise ValueError("Buffer lines must be a positive integer.")
            selected_detectors = self._selected_detectors()
            if not selected_detectors:
                raise ValueError("Select at least one algorithm.")

            roi_mask = self.target_label == ROAD_LABEL
            results = {}
            for detector_name in selected_detectors:
                md_map = self._run_detector(detector_name, self.cube, roi_mask, buffer_len, self.normalise_var.get())
                detection_mask = (md_map > threshold) & roi_mask
                score_rgb = self._score_map_rgb(md_map, roi_mask)
                detection_rgb = self._detection_overlay(self.rgb, roi_mask, detection_mask, self.anomaly_label)
                saved_paths = self._save_results(detector_name, md_map, detection_mask, score_rgb, detection_rgb)
                results[detector_name] = {
                    "md_map": md_map,
                    "detection_mask": detection_mask,
                    "score_rgb": score_rgb,
                    "detection_rgb": detection_rgb,
                    "saved_paths": saved_paths,
                }
        except Exception as exc:
            messagebox.showerror("Detection error", str(exc))
            return

        display_name = selected_detectors[-1]
        display_result = results[display_name]
        md_map = display_result["md_map"]
        detection_mask = display_result["detection_mask"]
        self._show_image(self.score_canvas, display_result["score_rgb"])
        self._show_image(self.detect_canvas, display_result["detection_rgb"])

        detected_pixels = int(np.count_nonzero(detection_mask))
        roi_pixels = int(np.count_nonzero(roi_mask))
        max_score = float(np.nanmax(md_map[roi_mask])) if roi_pixels else 0.0
        detector_labels = ", ".join(DETECTOR_CHOICES[name] for name in selected_detectors)
        self.info_var.set(
            f"{self.current_base} | algorithms={detector_labels} | displayed={DETECTOR_CHOICES[display_name]} | "
            f"threshold={threshold:.3f} | detected={detected_pixels}/{roi_pixels} | "
            f"max score={max_score:.3f} | saved to {display_result['saved_paths']['dir']}"
        )
        self.status_var.set(f"Detection complete: {self.current_base}")
        messagebox.showinfo(
            "Detection complete",
            f"{DETECTOR_CHOICES[display_name]} Detection complete!\n"
            f"Detected pixels: {detected_pixels}/{roi_pixels}\n"
            f"Saved to: {display_result['saved_paths']['dir']}",
        )

    def _selected_detectors(self):
        return [name for name, var in self.detector_vars.items() if var.get()]

    def _select_detector(self, selected_name):
        selected_var = self.detector_vars[selected_name]
        if not selected_var.get():
            selected_var.set(True)
            return
        for name, var in self.detector_vars.items():
            if name != selected_name:
                var.set(False)

    def _make_detector(self, detector_name, n_bands, n_pixels, buffer_len, normalise):
        defaults = DETECTOR_DEFAULTS[detector_name]
        if detector_name == "rx_baseline":
            return rx_baseline.RXBaseline(
                n_bands=n_bands,
                n_pixels=n_pixels,
                buffer_len=buffer_len,
                line_offset=0,
                normalise_md=normalise,
            )
        if detector_name == "rt_ck_rxd":
            return rt_ck_rxd.RT_CK_RXD(
                n_bands=n_bands,
                n_pixels=n_pixels,
                buffer_len=buffer_len,
                line_offset=0,
                normalise_md=normalise,
            )
        if detector_name == "rx_bil":
            return rx_bil.RX_BIL(
                n_bands=n_bands,
                n_pixels=n_pixels,
                buffer_len=buffer_len,
                pixel_dropout=defaults["pixel_dropout"],
                line_offset=0,
                normalise_md=normalise,
            )
        if detector_name == "cdlss_ad":
            return cdlss_ad.CDLSS_AD(
                n_bands=n_bands,
                n_pixels=n_pixels,
                buffer_len=buffer_len,
                line_offset=0,
                normalise_md=normalise,
            )
        if detector_name == "lbl_ad":
            return lbl_ad.LblAD(
                n_bands=n_bands,
                n_pixels=n_pixels,
                buffer_len=buffer_len,
                pca_dims=defaults["pca_dims"],
                line_offset=0,
                normalise_md=normalise,
            )
        if detector_name == "erx":
            return erx.ERX(
                n_bands=n_bands,
                n_pixels=n_pixels,
                buffer_len=buffer_len,
                n_projdims=defaults["projected_dimensions"],
                momentum=defaults["momentum"],
                normalise_md=normalise,
            )
        if detector_name == "erx_ablation":
            return erx_ablation.ERXwAblation(
                n_bands=n_bands,
                n_pixels=n_pixels,
                buffer_len=buffer_len,
                n_projdims=defaults["projected_dimensions"],
                momentum=defaults["momentum"],
                line_offset=0,
                normalise_md=normalise,
            )
        if detector_name == "erx_r":
            return erx_ablation.ERX_R(
                n_bands=n_bands,
                n_pixels=n_pixels,
                buffer_len=buffer_len,
                momentum=defaults["momentum"],
                line_offset=0,
                normalise_md=normalise,
            )
        raise ValueError(f"Unknown detector: {detector_name}")

    def _run_detector(self, detector_name, cube, roi_mask, buffer_len, normalise):
        hsi = self._cube_to_hwc(cube).astype(np.float64, copy=False)
        height, width, n_bands = hsi.shape
        if buffer_len > height:
            raise ValueError(
                f"Buffer lines ({buffer_len}) cannot exceed cube height ({height}) for line-scan detectors."
            )

        detector = self._make_detector(detector_name, n_bands, width, buffer_len, normalise)
        md_map = np.full((height, width), np.nan, dtype=np.float32)
        for row_idx in range(height):
            md_line = detector.forward(np.nan_to_num(hsi[row_idx], copy=False))
            if md_line is not None:
                md_line = np.asarray(md_line, dtype=np.float32)
                if md_line.shape[0] != width:
                    raise ValueError(
                        f"{detector_name} returned {md_line.shape[0]} scores for line width {width}."
                    )
                md_map[row_idx] = md_line

        md_map[~roi_mask] = np.nan
        if not np.isfinite(md_map[roi_mask]).any():
            raise ValueError(f"{detector_name} did not produce finite scores inside the road ROI.")
        return md_map

    def _cube_to_hwc(self, cube):
        if cube.shape[0] == 25:
            return np.moveaxis(cube, 0, -1)
        return cube

    # def _rx_baseline_roi(self, cube, roi_mask, normalise):
    #     spectra = self._spectra_by_mask(cube, roi_mask).astype(np.float64)
    #     if spectra.shape[0] <= spectra.shape[1]:
    #         raise ValueError(
    #             f"Not enough ROI pixels for RX covariance: pixels={spectra.shape[0]}, bands={spectra.shape[1]}"
    #         )
    #     spectra = np.nan_to_num(spectra, copy=False)
    #     mean = spectra.mean(axis=0)
    #     centered = spectra - mean
    #     cov = np.cov(centered, rowvar=False)
    #     scale = float(np.trace(cov)) / max(cov.shape[0], 1)
    #     cov = cov + np.eye(cov.shape[0], dtype=np.float64) * max(scale, 1.0) * 1e-6
    #     inv_cov = np.linalg.pinv(cov)
    #     md = np.sqrt(np.einsum("ij,jk,ik->i", centered, inv_cov, centered))
    #     if normalise:
    #         std = float(md.std())
    #         md = (md - float(md.mean())) / std if std > 0.0 else md * 0.0
    #
    #     height, width = roi_mask.shape[:2]
    #     md_map = np.full((height, width), np.nan, dtype=np.float32)
    #     md_map[roi_mask] = md.astype(np.float32)
    #     return md_map

    def _spectra_by_mask(self, cube, mask):
        if cube.shape[0] == 25:
            return np.moveaxis(cube, 0, -1)[mask]
        return cube[mask]

    def _cube_hw(self, cube):
        if cube.shape[0] == 25:
            return cube.shape[1], cube.shape[2]
        return cube.shape[0], cube.shape[1]

    def _roi_overlay(self, rgb, roi_mask):
        out = rgb.copy().astype(np.float32)
        out[roi_mask] = out[roi_mask] * 0.45 + np.array((30, 170, 255), dtype=np.float32) * 0.55
        out[~roi_mask] *= 0.35
        return np.clip(out, 0, 255).astype(np.uint8)

    def _score_map_rgb(self, md_map, roi_mask):
        values = md_map[roi_mask]
        out = np.zeros((*md_map.shape, 3), dtype=np.uint8)
        if values.size == 0:
            return out
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return out
        v_min = float(np.percentile(finite, 1))
        v_max = float(np.percentile(finite, 99))
        if v_max <= v_min:
            v_max = v_min + 1.0
        norm = np.zeros(md_map.shape, dtype=np.uint8)
        scaled = np.clip((np.nan_to_num(md_map, nan=v_min) - v_min) / (v_max - v_min), 0.0, 1.0)
        norm[roi_mask] = (scaled[roi_mask] * 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        out[roi_mask] = heat_rgb[roi_mask]
        out[~roi_mask] = 20
        return out

    def _detection_overlay(self, rgb, roi_mask, detection_mask, anomaly_label):
        out = rgb.copy().astype(np.float32)
        out[~roi_mask] *= 0.30
        out[roi_mask] *= 0.72
        if anomaly_label is not None:
            gt_mask = (anomaly_label == 1) & roi_mask
            out[gt_mask] = out[gt_mask] * 0.35 + np.array((255, 210, 0), dtype=np.float32) * 0.65
        out[detection_mask] = out[detection_mask] * 0.25 + np.array((255, 0, 0), dtype=np.float32) * 0.75
        return np.clip(out, 0, 255).astype(np.uint8)

    def _save_results(self, detector_name, md_map, detection_mask, score_rgb, detection_rgb):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        base = self.current_base
        md_path = self.result_dir / f"{base}_{detector_name}_scores.npy"
        binary_path = self.result_dir / f"{base}_{detector_name}_binary.png"
        score_path = self.result_dir / f"{base}_{detector_name}_score_map.png"
        overlay_path = self.result_dir / f"{base}_{detector_name}_overlay.png"
        np.save(md_path, md_map)
        cv2.imwrite(str(binary_path), (detection_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(score_path), cv2.cvtColor(score_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(overlay_path), cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2BGR))
        return {
            "dir": self.result_dir,
            "scores": md_path,
            "binary": binary_path,
            "score_map": score_path,
            "overlay": overlay_path,
        }

    def _show_image(self, canvas, rgb):
        height, width = rgb.shape[:2]
        canvas.config(width=width, height=height)
        image = self._photo_image(rgb)
        canvas.image = image
        canvas.delete("all")
        canvas.create_image(0, 0, image=image, anchor="nw")

    def _photo_image(self, rgb):
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        height, width = rgb.shape[:2]
        header = f"P6 {width} {height} 255\n".encode("ascii")
        data = header + np.ascontiguousarray(rgb).tobytes()
        return tk.PhotoImage(data=data, format="PPM")


def parse_args():
    parser = argparse.ArgumentParser(description="Run selectable anomaly detectors on generated anomaly samples.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Generated anomaly dataset directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    try:
        GeneratedAnomalyRXDetector(root, args.dataset_dir)
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
