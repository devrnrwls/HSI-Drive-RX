#!/usr/bin/env python3
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np


DEFAULT_DATASET_DIR = "data_backup/Generated_Anomaly_Dataset2"


class AnomalyDatasetVisualizer:
    def __init__(self, root, dataset_dir):
        self.root = root
        self.dataset_dir = Path(dataset_dir)
        self.cube_dir = self.dataset_dir / "Cubes_Scaling"
        self.rgb_dir = self.dataset_dir / "RGB"
        self.label_dir = self.dataset_dir / "Labels"

        self.cube = None
        self.rgb = None
        self.label = None
        self.current_base = None
        self.rgb_markers = []
        self.overlay_markers = []
        self.selected_pixels = []

        self.files = self._find_cube_files()
        self.displayed_files = list(self.files)

        self.root.title("Generated Anomaly Dataset Visualizer")
        self.root.geometry("1180x760")
        self.root.minsize(1000, 680)
        self._build_ui()
        if self.files:
            self.file_list.selection_set(0)
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
        search = ttk.Entry(left, textvariable=self.search_var, width=38)
        search.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        search.bind("<KeyRelease>", self._filter_files)

        list_frame = ttk.Frame(left)
        list_frame.grid(row=2, column=0, sticky="ns")
        self.file_list = tk.Listbox(list_frame, width=40, height=30, exportselection=False)
        self.file_list.grid(row=0, column=0, sticky="ns")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_selected)
        self._refresh_listbox()

        self.status_var = tk.StringVar(value=f"{len(self.files)} files")
        ttk.Label(left, textvariable=self.status_var, wraplength=300).grid(row=3, column=0, sticky="w", pady=(8, 0))

        scale_box = ttk.LabelFrame(left, text="Y-axis")
        scale_box.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        self.y_scale_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(
            scale_box,
            text="Auto min/max",
            value="auto",
            variable=self.y_scale_mode,
            command=self._redraw_spectrum,
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))
        ttk.Radiobutton(
            scale_box,
            text="Fixed 0-1",
            value="fixed",
            variable=self.y_scale_mode,
            command=self._redraw_spectrum,
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(2, 6))

        selection_box = ttk.LabelFrame(left, text="Selections")
        selection_box.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(selection_box, text="Clear selected pixels", command=self._clear_selected_pixels).grid(
            row=0, column=0, sticky="ew", padx=8, pady=(6, 6)
        )

        main = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(4, weight=1)

        ttk.Label(main, text="Generated RGB - click a pixel").grid(row=0, column=0, sticky="w")
        ttk.Label(main, text="Anomaly label overlay - click a pixel").grid(row=0, column=1, sticky="w")
        self.rgb_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.rgb_canvas.grid(row=1, column=0, sticky="nw", padx=(0, 12), pady=(6, 8))
        self.rgb_canvas.bind("<Button-1>", self._on_image_click)
        self.overlay_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.overlay_canvas.grid(row=1, column=1, sticky="nw", pady=(6, 8))
        self.overlay_canvas.bind("<Button-1>", self._on_image_click)

        self.info_var = tk.StringVar(value="Select a generated cube file.")
        ttk.Label(main, textvariable=self.info_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

        plot_frame = ttk.Frame(main)
        plot_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.plot_canvas = tk.Canvas(plot_frame, height=360, bg="white", highlightthickness=1)
        self.plot_canvas.grid(row=0, column=0, sticky="nsew")

    def _refresh_listbox(self):
        self.file_list.delete(0, tk.END)
        for path in self.displayed_files:
            self.file_list.insert(tk.END, path.name)

    def _filter_files(self, _event=None):
        query = self.search_var.get().strip().lower()
        self.displayed_files = [p for p in self.files if query in p.name.lower()] if query else list(self.files)
        self._refresh_listbox()
        self.status_var.set(f"{len(self.displayed_files)} / {len(self.files)} files")

    def _base_from_cube(self, cube_path):
        name = cube_path.name
        if name.endswith("_RC_TC.npy"):
            return name.removesuffix("_RC_TC.npy")
        return cube_path.stem

    def _on_file_selected(self, _event=None):
        selected = self.file_list.curselection()
        if not selected:
            return
        try:
            self._load_triplet(self.displayed_files[selected[0]])
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _load_triplet(self, cube_path):
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

        self.current_base = base
        self.cube = cube
        self.rgb = rgb
        self.label = label
        self.rgb_markers = []
        self.overlay_markers = []
        self.selected_pixels = []
        self._show_image(self.rgb_canvas, rgb)
        self._show_image(self.overlay_canvas, self._overlay(rgb, label))
        anomaly_pixels = int(np.count_nonzero(label == 1))
        self.info_var.set(f"{base} | cube {cube.shape} {cube.dtype} | anomaly pixels={anomaly_pixels}")
        self.plot_canvas.delete("all")

    def _cube_hw(self, cube):
        if cube.shape[0] == 25:
            return cube.shape[1], cube.shape[2]
        return cube.shape[0], cube.shape[1]

    def _spectrum_at(self, cube, x, y):
        if cube.shape[0] == 25:
            return cube[:, y, x]
        return cube[y, x, :]

    def _overlay(self, rgb, label):
        out = rgb.copy().astype(np.float32)
        mask = label == 1
        out[mask] = out[mask] * 0.35 + np.array((255, 0, 0), dtype=np.float32) * 0.65
        out[~mask] *= 0.65
        return np.clip(out, 0, 255).astype(np.uint8)

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

    def _on_image_click(self, event):
        if self.cube is None:
            return
        x = int(event.x)
        y = int(event.y)
        height, width = self._cube_hw(self.cube)
        if x < 0 or y < 0 or x >= width or y >= height:
            return
        spectrum = np.asarray(self._spectrum_at(self.cube, x, y), dtype=float).copy()
        label_value = int(self.label[y, x])
        color = self._selection_color(len(self.selected_pixels))
        number = len(self.selected_pixels) + 1
        self.selected_pixels.append({
            "x": x,
            "y": y,
            "label": label_value,
            "spectrum": spectrum,
            "color": color,
            "number": number,
        })
        self.rgb_markers.append(self._draw_marker(self.rgb_canvas, x, y, color, number))
        self.overlay_markers.append(self._draw_marker(self.overlay_canvas, x, y, color, number))
        self.info_var.set(
            f"{self.current_base} | selected #{number} x={x}, y={y} | anomaly_label={label_value} | "
            f"band min={float(np.min(spectrum)):.4f}, max={float(np.max(spectrum)):.4f} | "
            f"total selected={len(self.selected_pixels)}"
        )
        self._redraw_spectrum()

    def _selection_color(self, idx):
        colors = (
            "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
            "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
        )
        return colors[idx % len(colors)]

    def _draw_marker(self, canvas, x, y, color, number):
        radius = 4
        return (
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=2),
            canvas.create_line(x - 8, y, x + 8, y, fill=color, width=1),
            canvas.create_line(x, y - 8, x, y + 8, fill=color, width=1),
            canvas.create_text(x + 10, y - 10, text=str(number), fill=color, anchor="w", font=("TkDefaultFont", 9, "bold")),
        )

    def _clear_selected_pixels(self):
        for marker_group in self.rgb_markers:
            for item_id in marker_group:
                self.rgb_canvas.delete(item_id)
        for marker_group in self.overlay_markers:
            for item_id in marker_group:
                self.overlay_canvas.delete(item_id)
        self.rgb_markers = []
        self.overlay_markers = []
        self.selected_pixels = []
        if self.current_base is not None and self.cube is not None:
            anomaly_pixels = int(np.count_nonzero(self.label == 1))
            self.info_var.set(f"{self.current_base} | cube {self.cube.shape} {self.cube.dtype} | anomaly pixels={anomaly_pixels}")
        self.plot_canvas.delete("all")

    def _redraw_spectrum(self):
        if not self.selected_pixels:
            return
        self.plot_canvas.delete("all")
        width = max(self.plot_canvas.winfo_width(), 760)
        height = max(self.plot_canvas.winfo_height(), 320)
        margin_left = 62
        margin_right = 24
        margin_top = 42
        margin_bottom = 48
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        x0 = margin_left
        y0 = margin_top + plot_h
        spectra = [item["spectrum"] for item in self.selected_pixels]
        if self.y_scale_mode.get() == "fixed":
            v_min = 0.0
            v_max = 1.0
            scale_label = "fixed 0-1"
        else:
            all_values = np.concatenate(spectra)
            v_min = float(np.nanmin(all_values))
            v_max = float(np.nanmax(all_values))
            scale_label = "auto min/max"
        if v_max == v_min:
            v_max = v_min + 1.0
        self.plot_canvas.create_text(
            margin_left,
            16,
            anchor="w",
            text=f"Selected pixel spectra comparison - {len(self.selected_pixels)} pixels - {scale_label}",
            font=("TkDefaultFont", 11, "bold"),
        )
        self.plot_canvas.create_line(x0, margin_top, x0, y0, fill="#444")
        self.plot_canvas.create_line(x0, y0, x0 + plot_w, y0, fill="#444")
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = y0 - frac * plot_h
            value = v_min + frac * (v_max - v_min)
            self.plot_canvas.create_line(x0 - 4, yy, x0 + plot_w, yy, fill="#e5e5e5")
            self.plot_canvas.create_text(x0 - 8, yy, anchor="e", text=f"{value:.3f}")
        n = max(len(values) for values in spectra)
        for item_idx, item in enumerate(self.selected_pixels):
            values = item["spectrum"]
            points = []
            for band_idx, value in enumerate(values):
                px = x0 + (band_idx / (len(values) - 1)) * plot_w if len(values) > 1 else x0
                py = y0 - ((float(value) - v_min) / (v_max - v_min)) * plot_h
                points.append((px, py))
            for point_idx in range(len(points) - 1):
                self.plot_canvas.create_line(
                    *points[point_idx],
                    *points[point_idx + 1],
                    fill=item["color"],
                    width=2,
                )
            for px, py in points:
                self.plot_canvas.create_oval(px - 2.2, py - 2.2, px + 2.2, py + 2.2, fill=item["color"], outline="")
            legend_y = 38 + item_idx * 18
            self.plot_canvas.create_line(x0 + 12, legend_y, x0 + 42, legend_y, fill=item["color"], width=3)
            self.plot_canvas.create_text(
                x0 + 50,
                legend_y,
                anchor="w",
                text=f"#{item['number']} ({item['x']}, {item['y']}) label={item['label']}",
                fill="#222",
            )

        for idx in range(n):
            if idx % 2 == 0:
                px = x0 + (idx / (n - 1)) * plot_w if n > 1 else x0
                self.plot_canvas.create_text(px, y0 + 14, text=str(idx + 1), anchor="n")
        self.plot_canvas.create_text(x0 + plot_w / 2, height - 12, text="Band index", anchor="s")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize generated HSI anomaly dataset samples.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Generated anomaly dataset directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    try:
        AnomalyDatasetVisualizer(root, args.dataset_dir)
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
