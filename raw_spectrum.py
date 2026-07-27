#!/usr/bin/env python3
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np


ROAD_LABEL = 1
ROAD_COLOR = (128, 64, 128)


class RoadRepresentativeSpectrum:
    def __init__(self, root, data_dir):
        self.root = root
        self.data_dir = Path(data_dir)
        self.cube_dir = self.data_dir / "Cubes_Scaling"
        self.rgb_dir = self.data_dir / "RGB"
        self.label_dir = self.data_dir / "Labels"

        self.current_base = None
        self.cube = None
        self.rgb = None
        self.label = None
        self.road_mask = None
        self.mean_spectrum = None
        self.std_spectrum = None
        self.median_spectrum = None
        self.road_pixel_count = 0
        self.reference_spectrum = None
        self.reference_base = None
        self.reference_pixel_count = 0
        self.clicked_spectrum = None
        self.clicked_x = None
        self.clicked_y = None
        self.rgb_marker = None

        self.files = self._find_cube_files()
        self.displayed_files = list(self.files)

        self.root.title("HSI-Drive Road Representative Spectrum")
        self.root.geometry("1220x760")
        self.root.minsize(1040, 680)
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

        ttk.Label(left, text="Cubes_Scaling files").grid(row=0, column=0, sticky="w")
        self.search_var = tk.StringVar()
        search = ttk.Entry(left, textvariable=self.search_var, width=38)
        search.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        search.bind("<KeyRelease>", self._filter_files)

        list_frame = ttk.Frame(left)
        list_frame.grid(row=2, column=0, sticky="ns")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.file_list = tk.Listbox(list_frame, width=40, height=30, exportselection=False)
        self.file_list.grid(row=0, column=0, sticky="ns")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_selected)
        self._refresh_listbox()

        self.status_var = tk.StringVar(value=f"{len(self.files)} files")
        ttk.Label(left, textvariable=self.status_var).grid(row=3, column=0, sticky="w", pady=(8, 0))

        scale_box = ttk.LabelFrame(left, text="Y-axis")
        scale_box.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        self.y_scale_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(
            scale_box,
            text="Auto mean +/- std",
            value="auto",
            variable=self.y_scale_mode,
            command=self._draw_plot,
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))
        ttk.Radiobutton(
            scale_box,
            text="Fixed 0-1",
            value="fixed",
            variable=self.y_scale_mode,
            command=self._draw_plot,
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(2, 6))

        display_box = ttk.LabelFrame(left, text="Plot")
        display_box.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        self.show_std_var = tk.BooleanVar(value=True)
        self.show_median_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            display_box,
            text="Show std range",
            variable=self.show_std_var,
            command=self._draw_plot,
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))
        ttk.Checkbutton(
            display_box,
            text="Show median",
            variable=self.show_median_var,
            command=self._draw_plot,
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(2, 6))

        reference_box = ttk.LabelFrame(left, text="Reference")
        reference_box.grid(row=6, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(
            reference_box,
            text="Set current Road mean as reference",
            command=self._set_reference_spectrum,
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 4))
        ttk.Button(
            reference_box,
            text="Clear clicked pixel",
            command=self._clear_clicked_pixel,
        ).grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 4))
        self.reference_info_var = tk.StringVar(value="Reference: none")
        ttk.Label(reference_box, textvariable=self.reference_info_var, wraplength=280).grid(
            row=2, column=0, sticky="w", padx=8, pady=(2, 6)
        )

        main = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(4, weight=1)

        ttk.Label(main, text="RGB image - click a pixel to compare").grid(row=0, column=0, sticky="w")
        ttk.Label(main, text="Road mask overlay - label 1 only").grid(row=0, column=1, sticky="w")

        self.rgb_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.rgb_canvas.grid(row=1, column=0, sticky="nw", padx=(0, 12), pady=(6, 8))
        self.rgb_canvas.bind("<Button-1>", self._on_rgb_click)

        self.mask_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.mask_canvas.grid(row=1, column=1, sticky="nw", pady=(6, 8))

        self.info_var = tk.StringVar(value="Select a cube file.")
        ttk.Label(main, textvariable=self.info_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

        plot_frame = ttk.Frame(main)
        plot_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.plot_canvas = tk.Canvas(plot_frame, height=370, bg="white", highlightthickness=1)
        self.plot_canvas.grid(row=0, column=0, sticky="nsew")
        self.plot_canvas.bind("<Configure>", lambda _event: self._draw_plot())

    def _refresh_listbox(self):
        self.file_list.delete(0, tk.END)
        for path in self.displayed_files:
            self.file_list.insert(tk.END, path.name)

    def _filter_files(self, _event=None):
        query = self.search_var.get().strip().lower()
        if query:
            self.displayed_files = [path for path in self.files if query in path.name.lower()]
        else:
            self.displayed_files = list(self.files)
        self._refresh_listbox()
        self.status_var.set(f"{len(self.displayed_files)} / {len(self.files)} files")

    def _on_file_selected(self, _event=None):
        selection = self.file_list.curselection()
        if not selection:
            return
        try:
            self._load_triplet(self.displayed_files[selection[0]])
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _base_from_cube(self, cube_path):
        name = cube_path.name
        if name.endswith("_RC_TC.npy"):
            return name.removesuffix("_RC_TC.npy")
        return cube_path.stem

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
            raise ValueError(
                f"Shape mismatch: cube={cube.shape}, RGB={rgb.shape}, label={label.shape}"
            )

        road_mask = label == ROAD_LABEL
        road_pixel_count = int(np.count_nonzero(road_mask))
        if road_pixel_count == 0:
            raise ValueError(f"No Road label pixels found in {label_path}")

        spectra = self._masked_spectra(cube, road_mask)
        self.current_base = base
        self.cube = cube
        self.rgb = rgb
        self.label = label
        self.road_mask = road_mask
        self.road_pixel_count = road_pixel_count
        self.mean_spectrum = np.mean(spectra, axis=0)
        self.std_spectrum = np.std(spectra, axis=0)
        self.median_spectrum = np.median(spectra, axis=0)
        self.clicked_spectrum = None
        self.clicked_x = None
        self.clicked_y = None
        self.rgb_marker = None

        total_pixels = height * width
        road_ratio = (road_pixel_count / total_pixels) * 100.0
        self._show_image(self.rgb_canvas, rgb)
        self._show_image(self.mask_canvas, self._road_overlay(rgb, road_mask))
        self.info_var.set(
            f"{base} | cube {cube.shape} {cube.dtype} | Road pixels={road_pixel_count} "
            f"of {total_pixels} ({road_ratio:.2f}%)"
        )
        self._draw_plot()

    def _set_reference_spectrum(self):
        if self.mean_spectrum is None:
            messagebox.showinfo("Reference required", "Select a cube file before setting a reference.")
            return
        self.reference_spectrum = np.asarray(self.mean_spectrum, dtype=float).copy()
        self.reference_base = self.current_base
        self.reference_pixel_count = self.road_pixel_count
        self.reference_info_var.set(
            f"Reference: {self.reference_base} ({self.reference_pixel_count} Road pixels)"
        )
        self._draw_plot()

    def _clear_clicked_pixel(self):
        self.clicked_spectrum = None
        self.clicked_x = None
        self.clicked_y = None
        if self.rgb_marker is not None:
            for item_id in self.rgb_marker:
                self.rgb_canvas.delete(item_id)
            self.rgb_marker = None
        self._draw_plot()

    def _on_rgb_click(self, event):
        if self.cube is None:
            return
        x = int(event.x)
        y = int(event.y)
        height, width = self._cube_hw(self.cube)
        if x < 0 or y < 0 or x >= width or y >= height:
            return

        self.clicked_spectrum = np.asarray(self._spectrum_at(self.cube, x, y), dtype=float).copy()
        self.clicked_x = x
        self.clicked_y = y
        self.rgb_marker = self._draw_marker(self.rgb_canvas, self.rgb_marker, x, y)
        self.info_var.set(
            f"{self.current_base} | clicked x={x}, y={y} | pixel band min="
            f"{float(np.min(self.clicked_spectrum)):.4f}, max={float(np.max(self.clicked_spectrum)):.4f} | "
            f"Road pixels={self.road_pixel_count}"
        )
        self._draw_plot()

    def _cube_hw(self, cube):
        if cube.shape[0] == 25:
            return cube.shape[1], cube.shape[2]
        return cube.shape[0], cube.shape[1]

    def _masked_spectra(self, cube, mask):
        if cube.shape[0] == 25:
            return np.moveaxis(cube, 0, -1)[mask]
        return cube[mask]

    def _spectrum_at(self, cube, x, y):
        if cube.shape[0] == 25:
            return cube[:, y, x]
        return cube[y, x, :]

    def _road_overlay(self, rgb, road_mask):
        overlay = rgb.copy().astype(np.float32)
        color = np.array(ROAD_COLOR, dtype=np.float32)
        overlay[road_mask] = overlay[road_mask] * 0.45 + color * 0.55
        overlay[~road_mask] *= 0.35
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def _show_image(self, canvas, rgb):
        height, width = rgb.shape[:2]
        canvas.config(width=width, height=height)
        ppm = self._rgb_to_ppm(rgb)
        image = tk.PhotoImage(data=ppm, format="PPM")
        canvas.image = image
        canvas.delete("all")
        canvas.create_image(0, 0, image=image, anchor="nw")

    def _draw_marker(self, canvas, previous_marker, x, y):
        if previous_marker is not None:
            for item_id in previous_marker:
                canvas.delete(item_id)
        radius = 4
        return (
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline="#d62728", width=2),
            canvas.create_line(x - 8, y, x + 8, y, fill="#d62728", width=1),
            canvas.create_line(x, y - 8, x, y + 8, fill="#d62728", width=1),
        )

    def _rgb_to_ppm(self, rgb):
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        height, width = rgb.shape[:2]
        header = f"P6 {width} {height} 255\n".encode("ascii")
        return header + np.ascontiguousarray(rgb).tobytes()

    def _draw_plot(self):
        if not hasattr(self, "plot_canvas"):
            return
        self.plot_canvas.delete("all")
        width = max(self.plot_canvas.winfo_width(), 760)
        height = max(self.plot_canvas.winfo_height(), 330)
        margin_left = 64
        margin_right = 24
        margin_top = 44
        margin_bottom = 50
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        x0 = margin_left
        y0 = margin_top + plot_h

        self.plot_canvas.create_line(x0, margin_top, x0, y0, fill="#444")
        self.plot_canvas.create_line(x0, y0, x0 + plot_w, y0, fill="#444")

        plotted_values = []
        if self.mean_spectrum is not None:
            plotted_values.append(np.asarray(self.mean_spectrum, dtype=float))
        if self.reference_spectrum is not None:
            plotted_values.append(np.asarray(self.reference_spectrum, dtype=float))
        if self.clicked_spectrum is not None:
            plotted_values.append(np.asarray(self.clicked_spectrum, dtype=float))

        if not plotted_values:
            self.plot_canvas.create_text(
                x0 + plot_w / 2,
                margin_top + plot_h / 2,
                text="Select a cube file to compute Road(1) representative spectrum.",
                fill="#666",
            )
            return

        mean = np.asarray(self.mean_spectrum, dtype=float) if self.mean_spectrum is not None else None
        std = np.asarray(self.std_spectrum, dtype=float) if self.std_spectrum is not None else None
        median = np.asarray(self.median_spectrum, dtype=float) if self.median_spectrum is not None else None
        reference = (
            np.asarray(self.reference_spectrum, dtype=float)
            if self.reference_spectrum is not None
            else None
        )
        clicked = (
            np.asarray(self.clicked_spectrum, dtype=float)
            if self.clicked_spectrum is not None
            else None
        )

        if self.y_scale_mode.get() == "fixed":
            v_min = 0.0
            v_max = 1.0
            scale_label = "fixed 0-1"
        else:
            lower_parts = list(plotted_values)
            upper_parts = list(plotted_values)
            if mean is not None and std is not None and self.show_std_var.get():
                lower_parts.append(mean - std)
                upper_parts.append(mean + std)
            if median is not None and self.show_median_var.get():
                lower_parts.append(median)
                upper_parts.append(median)
            lower = np.concatenate(lower_parts)
            upper = np.concatenate(upper_parts)
            v_min = float(np.nanmin(lower))
            v_max = float(np.nanmax(upper))
            scale_label = "auto"
        if v_max == v_min:
            v_max = v_min + 1.0

        title = f"Road(1) representative spectrum and pixel comparison - {scale_label}"
        self.plot_canvas.create_text(
            margin_left,
            16,
            anchor="w",
            text=title,
            font=("TkDefaultFont", 11, "bold"),
        )

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = y0 - frac * plot_h
            value = v_min + frac * (v_max - v_min)
            self.plot_canvas.create_line(x0 - 4, yy, x0 + plot_w, yy, fill="#e5e5e5")
            self.plot_canvas.create_text(x0 - 8, yy, anchor="e", text=f"{value:.3f}")

        if mean is not None and std is not None and self.show_std_var.get():
            lower = mean - std
            upper = mean + std
            band_polygon = []
            upper_points = self._line_points(upper, x0, y0, plot_w, plot_h, v_min, v_max)
            lower_points = self._line_points(lower, x0, y0, plot_w, plot_h, v_min, v_max)
            for px, py in upper_points:
                band_polygon.extend((px, py))
            for px, py in reversed(lower_points):
                band_polygon.extend((px, py))
            self.plot_canvas.create_polygon(*band_polygon, fill="#b7d4f0", outline="")

        legend_x = x0 + 12
        legend_y = 38
        if mean is not None:
            self._draw_line(mean, "#1f77b4", x0, y0, plot_w, plot_h, v_min, v_max, width=3)
            self._draw_legend_line(
                legend_x,
                legend_y,
                "#1f77b4",
                f"Current Road mean: {self.current_base} ({self.road_pixel_count})",
            )
            legend_y += 20

        if median is not None and self.show_median_var.get():
            self._draw_line(median, "#d62728", x0, y0, plot_w, plot_h, v_min, v_max, width=2)
            self._draw_legend_line(legend_x, legend_y, "#d62728", "Current Road median")
            legend_y += 20

        if reference is not None:
            self._draw_line(reference, "#2ca02c", x0, y0, plot_w, plot_h, v_min, v_max, width=3)
            self._draw_legend_line(
                legend_x,
                legend_y,
                "#2ca02c",
                f"Reference Road mean: {self.reference_base} ({self.reference_pixel_count})",
            )
            legend_y += 20

        if clicked is not None:
            self._draw_line(clicked, "#ff7f0e", x0, y0, plot_w, plot_h, v_min, v_max, width=2)
            self._draw_legend_line(
                legend_x,
                legend_y,
                "#ff7f0e",
                f"Clicked pixel: {self.current_base} ({self.clicked_x}, {self.clicked_y})",
            )

        n = max(len(values) for values in plotted_values)
        for idx in range(n):
            if idx % 2 == 0:
                px = x0 + (idx / (n - 1)) * plot_w if n > 1 else x0
                self.plot_canvas.create_text(px, y0 + 14, text=str(idx + 1), anchor="n")

        self.plot_canvas.create_text(x0 + plot_w / 2, height - 12, text="Band index", anchor="s")

    def _line_points(self, values, x0, y0, plot_w, plot_h, v_min, v_max):
        values = np.asarray(values, dtype=float)
        n = len(values)
        points = []
        for idx, value in enumerate(values):
            px = x0 + (idx / (n - 1)) * plot_w if n > 1 else x0
            py = y0 - ((float(value) - v_min) / (v_max - v_min)) * plot_h
            points.append((px, py))
        return points

    def _draw_line(self, values, color, x0, y0, plot_w, plot_h, v_min, v_max, width):
        points = self._line_points(values, x0, y0, plot_w, plot_h, v_min, v_max)
        for idx in range(len(points) - 1):
            self.plot_canvas.create_line(*points[idx], *points[idx + 1], fill=color, width=width)
        for px, py in points:
            self.plot_canvas.create_oval(px - 2.5, py - 2.5, px + 2.5, py + 2.5, fill=color, outline="")

    def _draw_legend_line(self, x, y, color, text):
        self.plot_canvas.create_line(x, y, x + 32, y, fill=color, width=3)
        self.plot_canvas.create_text(x + 40, y, anchor="w", text=text, fill="#222")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the representative spectrum of Road(1) pixels using HSI-Drive labels."
    )
    parser.add_argument(
        "--data-dir",
        default="HSI_Drive",
        help="Directory containing Cubes_Scaling, RGB, and Labels folders.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    try:
        RoadRepresentativeSpectrum(root, args.data_dir)
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
