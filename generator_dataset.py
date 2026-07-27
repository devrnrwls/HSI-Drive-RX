#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np


ROAD_LABEL = 1
DEFAULT_OUTPUT_DIR = "Generated_Anomaly_Dataset"


class AnomalyDatasetGenerator:
    def __init__(self, root, data_dir, output_dir):
        self.root = root
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.cube_dir = self.data_dir / "Cubes_Scaling"
        self.rgb_dir = self.data_dir / "RGB"
        self.label_dir = self.data_dir / "Labels"

        self.files = self._find_cube_files()
        self.displayed_files = list(self.files)
        self.items = [None, None]
        self.rectangles = [None, None]
        self.selections = [None, None]

        self.root.title("HSI-Drive Anomaly Dataset Generator")
        self.root.geometry("1280x900")
        self.root.minsize(1120, 820)
        self._build_ui()

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

        ttk.Label(left, text="Cubes_Scaling files").grid(row=0, column=0, sticky="w")
        self.search_var = tk.StringVar()
        search = ttk.Entry(left, textvariable=self.search_var, width=40)
        search.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        search.bind("<KeyRelease>", self._filter_files)

        list_frame = ttk.Frame(left)
        list_frame.grid(row=2, column=0, sticky="ns")
        self.file_list = tk.Listbox(
            list_frame,
            width=42,
            height=28,
            selectmode=tk.EXTENDED,
            exportselection=False,
        )
        self.file_list.grid(row=0, column=0, sticky="ns")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self._refresh_listbox()

        load_box = ttk.LabelFrame(left, text="Load")
        load_box.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(load_box, text="Load as source", command=lambda: self._load_selected(0)).grid(
            row=0, column=0, sticky="ew", padx=8, pady=(6, 3)
        )
        ttk.Button(load_box, text="Load as target", command=lambda: self._load_selected(1)).grid(
            row=1, column=0, sticky="ew", padx=8, pady=3
        )
        ttk.Button(load_box, text="Load selected source&target", command=self._load_two).grid(
            row=2, column=0, sticky="ew", padx=8, pady=(3, 6)
        )

        patch_box = ttk.LabelFrame(left, text="Patch")
        patch_box.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self.patch_w_var = tk.StringVar(value="32")
        self.patch_h_var = tk.StringVar(value="32")
        self.min_road_ratio_var = tk.StringVar(value="0.90")
        self.mask_shape_var = tk.StringVar(value="irregular")
        self.feather_var = tk.StringVar(value="5")
        ttk.Label(patch_box, text="Width").grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))
        ttk.Entry(patch_box, textvariable=self.patch_w_var, width=8).grid(row=0, column=1, sticky="w", pady=(6, 2))
        ttk.Label(patch_box, text="Height").grid(row=1, column=0, sticky="w", padx=8, pady=2)
        ttk.Entry(patch_box, textvariable=self.patch_h_var, width=8).grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(patch_box, text="Min Road ratio").grid(row=2, column=0, sticky="w", padx=8, pady=2)
        ttk.Entry(patch_box, textvariable=self.min_road_ratio_var, width=8).grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(patch_box, text="Mask shape").grid(row=3, column=0, sticky="w", padx=8, pady=2)
        ttk.Combobox(
            patch_box,
            textvariable=self.mask_shape_var,
            values=("rectangle", "ellipse", "irregular"),
            state="readonly",
            width=10,
        ).grid(row=3, column=1, sticky="w", pady=2)
        ttk.Label(patch_box, text="Feather px").grid(row=4, column=0, sticky="w", padx=8, pady=2)
        ttk.Entry(patch_box, textvariable=self.feather_var, width=8).grid(row=4, column=1, sticky="w", pady=2)
        ttk.Button(patch_box, text="Random source Road patch", command=lambda: self._random_patch(0)).grid(
            row=5, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 3)
        )
        ttk.Button(patch_box, text="Random target Road patch", command=lambda: self._random_patch(1)).grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=8, pady=3
        )
        ttk.Button(patch_box, text="Random both", command=self._random_both).grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=8, pady=(3, 6)
        )

        output_box = ttk.LabelFrame(left, text="Output")
        output_box.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        self.output_var = tk.StringVar(value=str(self.output_dir))
        ttk.Entry(output_box, textvariable=self.output_var, width=34).grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        ttk.Button(output_box, text="Generate anomaly sample", command=self._generate).grid(
            row=1, column=0, sticky="ew", padx=8, pady=(3, 6)
        )

        self.status_var = tk.StringVar(value=f"{len(self.files)} files")
        ttk.Label(left, textvariable=self.status_var, wraplength=300).grid(row=6, column=0, sticky="w", pady=(8, 0))

        main = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(2, weight=1)
        main.rowconfigure(5, weight=1)

        self.title_vars = [tk.StringVar(value="Source patch image"), tk.StringVar(value="Target paste image")]
        self.info_vars = [
            tk.StringVar(value="Load source, then click road area."),
            tk.StringVar(value="Load target, then click road area."),
        ]
        self.canvases = []
        for idx in range(2):
            ttk.Label(main, textvariable=self.title_vars[idx]).grid(row=0, column=idx, sticky="w")
            canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
            canvas.grid(row=1, column=idx, sticky="nw", padx=(0, 12) if idx == 0 else 0, pady=(6, 6))
            canvas.bind("<Button-1>", lambda event, image_idx=idx: self._on_canvas_click(event, image_idx))
            self.canvases.append(canvas)
            ttk.Label(main, textvariable=self.info_vars[idx], wraplength=480).grid(
                row=2,
                column=idx,
                sticky="nw",
                padx=(0, 12) if idx == 0 else 0,
            )

        ttk.Label(main, text="Generated RGB").grid(row=3, column=0, sticky="w", pady=(16, 0))
        ttk.Label(main, text="Generated anomaly label overlay").grid(row=3, column=1, sticky="w", pady=(16, 0))
        self.generated_rgb_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.generated_label_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.generated_rgb_canvas.grid(row=4, column=0, sticky="nw", padx=(0, 12), pady=(6, 6))
        self.generated_label_canvas.grid(row=4, column=1, sticky="nw", pady=(6, 6))
        self.generated_info_var = tk.StringVar(value="Generated sample preview will appear here.")
        ttk.Label(main, textvariable=self.generated_info_var, wraplength=980).grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="nw",
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

    def _load_selected(self, idx):
        selected = self.file_list.curselection()
        if len(selected) != 1:
            messagebox.showinfo("Selection required", "Select exactly one cube file.")
            return
        try:
            self._load_path_into_slot(self.displayed_files[selected[0]], idx)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _load_two(self):
        selected = self.file_list.curselection()
        if len(selected) != 2:
            messagebox.showinfo("Selection required", "Select exactly two cube files.")
            return
        try:
            self._load_path_into_slot(self.displayed_files[selected[0]], 0)
            self._load_path_into_slot(self.displayed_files[selected[1]], 1)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _base_from_cube(self, cube_path):
        name = cube_path.name
        if name.endswith("_RC_TC.npy"):
            return name.removesuffix("_RC_TC.npy")
        return cube_path.stem

    def _load_path_into_slot(self, cube_path, idx):
        item = self._load_item(cube_path)
        self.items[idx] = item
        self.selections[idx] = None
        self.rectangles[idx] = None
        self.title_vars[idx].set(("Source: " if idx == 0 else "Target: ") + item["base"])
        self.info_vars[idx].set(f"cube {item['cube'].shape} {item['cube'].dtype}; click Road(1) area")
        self._show_image(idx)

    def _load_item(self, cube_path):
        base = self._base_from_cube(cube_path)
        rgb_path = self.rgb_dir / f"{base}_pseudocolor.png"
        label_path = self.label_dir / f"{base}.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        cube = np.load(cube_path)
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube, got {cube.shape}: {cube_path}")
        if rgb_bgr is None:
            raise ValueError(f"Failed to read RGB file: {rgb_path}")
        if label is None:
            raise ValueError(f"Failed to read label file: {label_path}")

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        height, width = self._cube_hw(cube)
        if rgb.shape[:2] != (height, width) or label.shape[:2] != (height, width):
            raise ValueError(f"Shape mismatch: cube={cube.shape}, RGB={rgb.shape}, label={label.shape}")
        return {"base": base, "cube": cube, "rgb": rgb, "label": label, "cube_path": cube_path}

    def _show_image(self, idx):
        item = self.items[idx]
        canvas = self.canvases[idx]
        self._show_rgb_on_canvas(canvas, item["rgb"])

    def _show_rgb_on_canvas(self, canvas, rgb):
        image = self._photo_image(rgb)
        canvas.config(width=image.width(), height=image.height())
        canvas.delete("all")
        canvas.create_image(0, 0, image=image, anchor="nw")
        canvas.image = image

    def _anomaly_overlay(self, rgb, label):
        out = rgb.copy().astype(np.float32)
        mask = label == 1
        out[mask] = out[mask] * 0.35 + np.array((255, 0, 0), dtype=np.float32) * 0.65
        out[~mask] *= 0.65
        return np.clip(out, 0, 255).astype(np.uint8)

    def _photo_image(self, rgb):
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        height, width = rgb.shape[:2]
        header = f"P6 {width} {height} 255\n".encode("ascii")
        data = header + np.ascontiguousarray(rgb).tobytes()
        return tk.PhotoImage(data=data, format="PPM")

    def _parse_patch_settings(self):
        try:
            patch_w = int(self.patch_w_var.get())
            patch_h = int(self.patch_h_var.get())
            min_road_ratio = float(self.min_road_ratio_var.get())
            feather = int(self.feather_var.get())
        except ValueError as exc:
            raise ValueError("Patch width/height, feather, and min road ratio must be numeric.") from exc
        if patch_w <= 0 or patch_h <= 0:
            raise ValueError("Patch width and height must be positive.")
        if feather < 0:
            raise ValueError("Feather pixels must be 0 or greater.")
        if min_road_ratio < 0.0 or min_road_ratio > 1.0:
            raise ValueError("Min Road ratio must be between 0 and 1.")
        if self.mask_shape_var.get() not in {"rectangle", "ellipse", "irregular"}:
            raise ValueError("Mask shape must be rectangle, ellipse, or irregular.")
        return patch_w, patch_h, min_road_ratio

    def _cube_hw(self, cube):
        if cube.shape[0] == 25:
            return cube.shape[1], cube.shape[2]
        return cube.shape[0], cube.shape[1]

    def _clamp_patch(self, x, y, width, height, patch_w, patch_h):
        x = max(0, min(int(x), width - patch_w))
        y = max(0, min(int(y), height - patch_h))
        return x, y

    def _on_canvas_click(self, event, idx):
        item = self.items[idx]
        if item is None:
            return
        try:
            patch_w, patch_h, min_road_ratio = self._parse_patch_settings()
            height, width = item["label"].shape[:2]
            if patch_w > width or patch_h > height:
                raise ValueError(f"Patch {patch_w}x{patch_h} is larger than image {width}x{height}.")
            x, y = self._clamp_patch(event.x - patch_w // 2, event.y - patch_h // 2, width, height, patch_w, patch_h)
            self._set_selection(idx, x, y, patch_w, patch_h, min_road_ratio)
        except Exception as exc:
            messagebox.showerror("Patch selection error", str(exc))

    def _set_selection(self, idx, x, y, patch_w, patch_h, min_road_ratio):
        item = self.items[idx]
        ratio = self._road_ratio(item["label"], x, y, patch_w, patch_h)
        self.selections[idx] = {
            "x": x,
            "y": y,
            "w": patch_w,
            "h": patch_h,
            "mask_shape": self.mask_shape_var.get(),
            "road_ratio": ratio,
        }
        self._draw_selection(idx)
        role = "source" if idx == 0 else "target"
        status = "OK" if ratio >= min_road_ratio else "LOW ROAD RATIO"
        self.info_vars[idx].set(
            f"{role} patch x={x}, y={y}, size={patch_w}x{patch_h}, Road ratio={ratio:.3f} ({status})"
        )

    def _road_ratio(self, label, x, y, patch_w, patch_h):
        patch = label[y:y + patch_h, x:x + patch_w]
        mask = self._binary_mask(patch_w, patch_h)
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels == 0:
            return 0.0
        return float(np.count_nonzero((patch == ROAD_LABEL) & mask)) / float(mask_pixels)

    def _draw_selection(self, idx):
        if self.rectangles[idx] is not None:
            self.canvases[idx].delete(self.rectangles[idx])
        selection = self.selections[idx]
        color = "#1f77b4" if idx == 0 else "#d62728"
        self.rectangles[idx] = self.canvases[idx].create_rectangle(
            selection["x"],
            selection["y"],
            selection["x"] + selection["w"],
            selection["y"] + selection["h"],
            outline=color,
            width=2,
        )

    def _random_patch(self, idx):
        item = self.items[idx]
        if item is None:
            messagebox.showinfo("Load required", "Load an image first.")
            return
        try:
            patch_w, patch_h, min_road_ratio = self._parse_patch_settings()
            x, y, ratio = self._find_random_road_patch(item["label"], patch_w, patch_h, min_road_ratio)
            self._set_selection(idx, x, y, patch_w, patch_h, min_road_ratio)
        except Exception as exc:
            messagebox.showerror("Random patch error", str(exc))

    def _random_both(self):
        self._random_patch(0)
        self._random_patch(1)

    def _find_random_road_patch(self, label, patch_w, patch_h, min_road_ratio, attempts=2000):
        height, width = label.shape[:2]
        if patch_w > width or patch_h > height:
            raise ValueError(f"Patch {patch_w}x{patch_h} is larger than image {width}x{height}.")
        road_y, road_x = np.where(label == ROAD_LABEL)
        if road_x.size == 0:
            raise ValueError("No Road(1) pixels found.")
        best = None
        for _ in range(attempts):
            center_idx = np.random.randint(0, road_x.size)
            x = int(road_x[center_idx]) - patch_w // 2
            y = int(road_y[center_idx]) - patch_h // 2
            x, y = self._clamp_patch(x, y, width, height, patch_w, patch_h)
            ratio = self._road_ratio(label, x, y, patch_w, patch_h)
            if best is None or ratio > best[2]:
                best = (x, y, ratio)
            if ratio >= min_road_ratio:
                return x, y, ratio
        if best is None:
            raise ValueError("Could not sample a Road patch.")
        raise ValueError(
            f"Could not find patch with Road ratio >= {min_road_ratio:.3f}. Best ratio={best[2]:.3f}."
        )

    def _generate(self):
        try:
            patch_w, patch_h, min_road_ratio = self._parse_patch_settings()
            if self.items[0] is None or self.items[1] is None:
                raise ValueError("Load both source and target images.")
            if self.selections[0] is None or self.selections[1] is None:
                raise ValueError("Select both source and target patches.")
            for idx in (0, 1):
                sel = self.selections[idx]
                if sel["w"] != patch_w or sel["h"] != patch_h:
                    raise ValueError("Patch size changed after selection. Select patches again.")
                if sel["mask_shape"] != self.mask_shape_var.get():
                    raise ValueError("Mask shape changed after selection. Select patches again.")
                if sel["road_ratio"] < min_road_ratio:
                    raise ValueError(
                        f"{'Source' if idx == 0 else 'Target'} Road ratio "
                        f"{sel['road_ratio']:.3f} is below {min_road_ratio:.3f}."
                    )
            saved = self._save_sample()
        except Exception as exc:
            messagebox.showerror("Generate error", str(exc))
            return
        self._show_generated_sample(saved)
        self.status_var.set(f"Saved {saved['name']} to {saved['output_dir']}")
        messagebox.showinfo("Generated", f"Saved sample: {saved['name']}")

    def _show_generated_sample(self, saved):
        self._show_rgb_on_canvas(self.generated_rgb_canvas, saved["rgb"])
        self._show_rgb_on_canvas(self.generated_label_canvas, self._anomaly_overlay(saved["rgb"], saved["label"]))
        anomaly_pixels = int(np.count_nonzero(saved["label"] == 1))
        self.generated_info_var.set(
            f"{saved['name']} | anomaly pixels={anomaly_pixels} | "
            f"RGB={saved['rgb_path']} | Label={saved['label_path']}"
        )

    def _save_sample(self):
        source = self.items[0]
        target = self.items[1]
        src_sel = self.selections[0]
        tgt_sel = self.selections[1]
        if source["cube"].shape != target["cube"].shape:
            raise ValueError(f"Source and target cube shapes differ: {source['cube'].shape} vs {target['cube'].shape}")
        out_dir = Path(self.output_var.get()).expanduser()
        cube_out_dir = out_dir / "Cubes_Scaling"
        rgb_out_dir = out_dir / "RGB"
        label_out_dir = out_dir / "Labels"
        target_label_out_dir = out_dir / "Target_Labels"
        cube_out_dir.mkdir(parents=True, exist_ok=True)
        rgb_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)
        target_label_out_dir.mkdir(parents=True, exist_ok=True)

        sample_id = self._next_sample_id(cube_out_dir)
        name = f"anomaly_{sample_id:06d}"
        cube = target["cube"].copy()
        rgb = target["rgb"].copy()
        anomaly_label = np.zeros(target["label"].shape[:2], dtype=np.uint8)
        binary_mask = self._binary_mask(src_sel["w"], src_sel["h"])
        alpha_mask = self._alpha_mask(binary_mask)

        self._paste_cube_patch(cube, source["cube"], src_sel, tgt_sel, alpha_mask)
        self._paste_rgb_patch(rgb, source["rgb"], src_sel, tgt_sel, alpha_mask)
        label_patch = anomaly_label[
            tgt_sel["y"]:tgt_sel["y"] + tgt_sel["h"],
            tgt_sel["x"]:tgt_sel["x"] + tgt_sel["w"],
        ]
        label_patch[binary_mask] = 1

        cube_path = cube_out_dir / f"{name}_RC_TC.npy"
        rgb_path = rgb_out_dir / f"{name}_pseudocolor.png"
        label_path = label_out_dir / f"{name}.png"
        target_label_path = target_label_out_dir / f"{name}.png"
        np.save(cube_path, cube)
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(label_path), anomaly_label)
        cv2.imwrite(str(target_label_path), target["label"])
        self._append_metadata(out_dir / "metadata.csv", name, source, target, src_sel, tgt_sel)
        return {
            "name": name,
            "output_dir": out_dir,
            "rgb": rgb,
            "label": anomaly_label,
            "rgb_path": rgb_path,
            "label_path": label_path,
            "target_label_path": target_label_path,
        }

    def _binary_mask(self, patch_w, patch_h):
        shape = self.mask_shape_var.get()
        if shape == "rectangle":
            return np.ones((patch_h, patch_w), dtype=bool)

        yy, xx = np.mgrid[0:patch_h, 0:patch_w]
        cx = (patch_w - 1) / 2.0
        cy = (patch_h - 1) / 2.0
        rx = max(patch_w * 0.45, 1.0)
        ry = max(patch_h * 0.45, 1.0)

        if shape == "ellipse":
            return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0

        angles = np.arctan2(yy - cy, xx - cx)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        radial = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        seed = (patch_w * 73856093) ^ (patch_h * 19349663)
        rng = np.random.default_rng(seed)
        control_angles = np.linspace(0, 2 * np.pi, 17)
        control_radii = rng.uniform(0.72, 1.08, size=17)
        control_radii[-1] = control_radii[0]
        boundary = np.interp(angles.ravel(), control_angles, control_radii).reshape((patch_h, patch_w))
        mask = radial <= boundary
        return self._smooth_binary_mask(mask)

    def _smooth_binary_mask(self, mask):
        kernel = np.ones((3, 3), dtype=np.uint8)
        smoothed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=1)
        return smoothed.astype(bool)

    def _alpha_mask(self, binary_mask):
        feather = int(self.feather_var.get())
        if feather <= 0:
            return binary_mask.astype(np.float32)
        mask_u8 = binary_mask.astype(np.uint8)
        dist_inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        alpha = np.clip(dist_inside / float(feather), 0.0, 1.0)
        alpha[~binary_mask] = 0.0
        return alpha.astype(np.float32)

    def _paste_rgb_patch(self, target_rgb, source_rgb, src_sel, tgt_sel, alpha_mask):
        sy = slice(src_sel["y"], src_sel["y"] + src_sel["h"])
        sx = slice(src_sel["x"], src_sel["x"] + src_sel["w"])
        ty = slice(tgt_sel["y"], tgt_sel["y"] + tgt_sel["h"])
        tx = slice(tgt_sel["x"], tgt_sel["x"] + tgt_sel["w"])
        src_patch = source_rgb[sy, sx].astype(np.float32)
        tgt_patch = target_rgb[ty, tx].astype(np.float32)
        alpha = alpha_mask[..., None]
        blended = src_patch * alpha + tgt_patch * (1.0 - alpha)
        target_rgb[ty, tx] = np.clip(blended, 0, 255).astype(np.uint8)

    def _paste_cube_patch(self, target_cube, source_cube, src_sel, tgt_sel, alpha_mask):
        sy = slice(src_sel["y"], src_sel["y"] + src_sel["h"])
        sx = slice(src_sel["x"], src_sel["x"] + src_sel["w"])
        ty = slice(tgt_sel["y"], tgt_sel["y"] + tgt_sel["h"])
        tx = slice(tgt_sel["x"], tgt_sel["x"] + tgt_sel["w"])
        if target_cube.shape[0] == 25:
            alpha = alpha_mask[None, :, :]
            target_cube[:, ty, tx] = source_cube[:, sy, sx] * alpha + target_cube[:, ty, tx] * (1.0 - alpha)
        else:
            alpha = alpha_mask[:, :, None]
            target_cube[ty, tx, :] = source_cube[sy, sx, :] * alpha + target_cube[ty, tx, :] * (1.0 - alpha)

    def _next_sample_id(self, cube_out_dir):
        max_id = 0
        for path in cube_out_dir.glob("anomaly_*_RC_TC.npy"):
            parts = path.name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                max_id = max(max_id, int(parts[1]))
        return max_id + 1

    def _append_metadata(self, metadata_path, name, source, target, src_sel, tgt_sel):
        new_file = not metadata_path.exists()
        fields = [
            "name", "created_at", "source_base", "target_base",
            "source_x", "source_y", "target_x", "target_y",
            "patch_w", "patch_h", "mask_shape", "feather",
            "source_road_ratio", "target_road_ratio",
        ]
        with metadata_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow({
                "name": name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_base": source["base"],
                "target_base": target["base"],
                "source_x": src_sel["x"],
                "source_y": src_sel["y"],
                "target_x": tgt_sel["x"],
                "target_y": tgt_sel["y"],
                "patch_w": src_sel["w"],
                "patch_h": src_sel["h"],
                "mask_shape": src_sel["mask_shape"],
                "feather": self.feather_var.get(),
                "source_road_ratio": f"{src_sel['road_ratio']:.6f}",
                "target_road_ratio": f"{tgt_sel['road_ratio']:.6f}",
            })


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HSI-Drive-style road anomaly samples.")
    parser.add_argument("--data-dir", default="HSI_Drive", help="Original HSI_Drive dataset directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Generated dataset output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    try:
        AnomalyDatasetGenerator(root, args.data_dir, args.output_dir)
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
