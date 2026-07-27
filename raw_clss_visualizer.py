#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np


LABEL_COLORS = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (244, 35, 232),
    3: (70, 70, 70),
    4: (102, 102, 156),
    5: (190, 153, 153),
    6: (153, 153, 153),
    7: (250, 170, 30),
    8: (220, 220, 0),
    9: (107, 142, 35),
    10: (70, 130, 180),
}

CLASS_DEFINITIONS = {
    0: ("Unlabeled", "No class / ignore"),
    1: ("Road", "Tarmac"),
    2: ("Road marks", "Lane markings and road markings"),
    3: ("Vegetation", "Any vegetation, including wood"),
    4: ("Painted metal", "Road signs, traffic light posts, vehicle bodies, etc."),
    5: ("Sky", "Sky"),
    6: ("Concrete/stone/brick", "Sidewalks, walls, facades, etc."),
    7: ("Pedestrian/cyclist", "Pedestrians and cyclists"),
    8: ("Water", "Water courses, puddles, etc."),
    9: ("Unpainted metal", "Back of road signs, posts, streetlights, crash barriers, etc."),
    10: ("Glass/transparent plastic", "Vehicle windscreens, lights, windows, etc."),
}


class HSIVisualizer:
    def __init__(self, root, data_dir):
        self.root = root
        self.data_dir = Path(data_dir)
        self.cube_dir = self.data_dir / "Cubes_Scaling" #"Cubes_Scaling" Cubes_NoScaling
        self.rgb_dir = self.data_dir / "RGB"
        self.label_dir = self.data_dir / "Labels"

        self.cube = None
        self.rgb = None
        self.label = None
        self.current_base = None
        self.rgb_marker = None
        self.label_marker = None
        self.last_spectrum = None
        self.last_label_value = None
        self.last_label_name = None
        self.last_x = None
        self.last_y = None
        self.class_filter_var = None

        self.root.title("HSI Drive Cube Visualizer")
        self.root.geometry("1180x760")
        self.root.minsize(1000, 680)

        self.files = self._find_cube_files()
        self._build_ui()
        if self.files:
            self.file_list.selection_set(0)
            self.file_list.event_generate("<<ListboxSelect>>")

    def _find_cube_files(self):
        if not self.cube_dir.is_dir():
            raise FileNotFoundError(f"Cube directory not found: {self.cube_dir}")
        return sorted(self.cube_dir.glob("*.npy"))

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.rowconfigure(2, weight=1)

        ttk.Label(left, text="Cubes_Scaling files").grid(row=0, column=0, sticky="w")
        self.search_var = tk.StringVar()
        search = ttk.Entry(left, textvariable=self.search_var, width=34)
        search.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        search.bind("<KeyRelease>", self._filter_files)

        list_frame = ttk.Frame(left)
        list_frame.grid(row=2, column=0, sticky="ns")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.file_list = tk.Listbox(list_frame, width=38, height=28, exportselection=False)
        self.file_list.grid(row=0, column=0, sticky="ns")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_selected)
        self.displayed_files = list(self.files)
        self._refresh_listbox()

        self.status_var = tk.StringVar(value=f"{len(self.files)} files")
        ttk.Label(left, textvariable=self.status_var).grid(row=3, column=0, sticky="w", pady=(8, 0))

        ttk.Label(left, text="Label class filter").grid(row=4, column=0, sticky="w", pady=(14, 4))
        self.class_filter_var = tk.StringVar(value="all: All classes")
        class_filter = ttk.Combobox(
            left,
            textvariable=self.class_filter_var,
            values=self._class_filter_values(),
            state="readonly",
            width=36,
        )
        class_filter.grid(row=5, column=0, sticky="ew")
        class_filter.bind("<<ComboboxSelected>>", self._on_class_filter_changed)

        ttk.Label(left, text="Class definitions").grid(row=6, column=0, sticky="w", pady=(14, 4))
        legend = tk.Text(left, width=38, height=15, wrap="word", relief="solid", borderwidth=1)
        legend.grid(row=7, column=0, sticky="ew")
        for value, (name, description) in CLASS_DEFINITIONS.items():
            legend.insert(tk.END, f"{value}: {name}\n")
            legend.insert(tk.END, f"   {description}\n")
        legend.configure(state="disabled")

        main = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=0)
        main.rowconfigure(3, weight=1)

        ttk.Label(main, text="RGB image - click a pixel").grid(row=0, column=0, sticky="w")
        ttk.Label(main, text="Label image - click a pixel").grid(row=0, column=1, sticky="w")

        self.rgb_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.rgb_canvas.grid(row=1, column=0, sticky="nw", padx=(0, 12), pady=(6, 10))
        self.rgb_canvas.bind("<Button-1>", self._on_image_click)

        self.label_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.label_canvas.grid(row=1, column=1, sticky="nw", pady=(6, 10))
        self.label_canvas.bind("<Button-1>", self._on_image_click)

        self.info_var = tk.StringVar(value="Select a cube file, then click on RGB or label image.")
        ttk.Label(main, textvariable=self.info_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 6))

        controls = ttk.Frame(main)
        controls.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(controls, text="Y-axis scale:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.y_scale_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(
            controls,
            text="Auto min/max",
            value="auto",
            variable=self.y_scale_mode,
            command=self._redraw_last_spectrum,
        ).grid(row=0, column=1, sticky="w", padx=(0, 10))
        ttk.Radiobutton(
            controls,
            text="Fixed 0-1",
            value="fixed",
            variable=self.y_scale_mode,
            command=self._redraw_last_spectrum,
        ).grid(row=0, column=2, sticky="w")

        plot_frame = ttk.Frame(main)
        plot_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.plot_canvas = tk.Canvas(plot_frame, height=310, bg="white", highlightthickness=1)
        self.plot_canvas.grid(row=0, column=0, sticky="nsew")

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

    def _class_filter_values(self):
        values = ["all: All classes"]
        values.extend(f"{value}: {name}" for value, (name, _description) in CLASS_DEFINITIONS.items())
        return values

    def _selected_class_filter(self):
        selected = self.class_filter_var.get()
        if selected.startswith("all:"):
            return None
        try:
            return int(selected.split(":", 1)[0])
        except ValueError:
            return None

    def _on_class_filter_changed(self, _event=None):
        if self.label is None:
            return
        self._show_image(self.label_canvas, self._label_to_rgb(self.label))
        if self.last_x is not None and self.last_y is not None:
            self.label_marker = self._draw_marker(self.label_canvas, None, self.last_x, self.last_y)

    def _on_file_selected(self, _event=None):
        selection = self.file_list.curselection()
        if not selection:
            return
        cube_path = self.displayed_files[selection[0]]
        try:
            self._load_triplet(cube_path)
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

        if rgb_bgr is None:
            raise ValueError(f"Failed to read RGB file: {rgb_path}")
        if label is None:
            raise ValueError(f"Failed to read label file: {label_path}")
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube, got shape {cube.shape}")

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        if cube.shape[0] == 25:
            expected_hw = cube.shape[1], cube.shape[2]
        else:
            expected_hw = cube.shape[0], cube.shape[1]
        if label.shape[:2] != expected_hw or rgb.shape[:2] != expected_hw:
            raise ValueError(
                f"Shape mismatch: cube={cube.shape}, RGB={rgb.shape}, label={label.shape}"
            )

        self.current_base = base
        self.cube = cube
        self.rgb = rgb
        self.label = label

        self._show_image(self.rgb_canvas, rgb)
        self._show_image(self.label_canvas, self._label_to_rgb(label))
        self.plot_canvas.delete("all")
        self.rgb_marker = None
        self.label_marker = None
        self.last_spectrum = None
        self.last_label_value = None
        self.last_label_name = None
        self.last_x = None
        self.last_y = None
        self.info_var.set(
            f"{base} | cube {cube.shape} {cube.dtype} | RGB {rgb.shape} | label {label.shape}"
        )

    def _show_image(self, canvas, rgb):
        height, width = rgb.shape[:2]
        canvas.config(width=width, height=height)
        ppm = self._rgb_to_ppm(rgb)
        image = tk.PhotoImage(data=ppm, format="PPM")
        canvas.image = image
        canvas.delete("all")
        canvas.create_image(0, 0, image=image, anchor="nw")

    def _rgb_to_ppm(self, rgb):
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        height, width = rgb.shape[:2]
        header = f"P6 {width} {height} 255\n".encode("ascii")
        return header + np.ascontiguousarray(rgb).tobytes()

    def _label_to_rgb(self, label):
        out = np.zeros((*label.shape[:2], 3), dtype=np.uint8)
        selected_class = self._selected_class_filter()
        if selected_class is not None:
            out[label == selected_class] = LABEL_COLORS.get(selected_class, (255, 255, 255))
            return out
        for value in np.unique(label):
            out[label == value] = LABEL_COLORS.get(int(value), (255, 255, 255))
        return out

    def _on_image_click(self, event):
        if self.cube is None:
            return
        x = int(event.x)
        y = int(event.y)
        height, width = self.label.shape[:2]
        if x < 0 or y < 0 or x >= width or y >= height:
            return

        if self.cube.shape[0] == 25:
            spectrum = self.cube[:, y, x]
        else:
            spectrum = self.cube[y, x, :]

        label_value = int(self.label[y, x])
        label_name, label_description = CLASS_DEFINITIONS.get(label_value, ("Unknown", "Unknown label value"))
        rgb_value = tuple(int(v) for v in self.rgb[y, x])
        self.info_var.set(
            f"{self.current_base} | x={x}, y={y} | label={label_value}: {label_name} "
            f"({label_description}) | "
            f"RGB={rgb_value} | band min={float(np.min(spectrum)):.4f}, "
            f"max={float(np.max(spectrum)):.4f}"
        )
        self.rgb_marker = self._draw_marker(self.rgb_canvas, self.rgb_marker, x, y)
        self.label_marker = self._draw_marker(self.label_canvas, self.label_marker, x, y)
        self.last_spectrum = np.asarray(spectrum, dtype=float).copy()
        self.last_label_value = label_value
        self.last_label_name = label_name
        self.last_x = x
        self.last_y = y
        self._draw_spectrum(spectrum, label_value, label_name, x, y)

    def _redraw_last_spectrum(self):
        if self.last_spectrum is None:
            return
        self._draw_spectrum(
            self.last_spectrum,
            self.last_label_value,
            self.last_label_name,
            self.last_x,
            self.last_y,
        )

    def _draw_marker(self, canvas, previous_marker, x, y):
        if previous_marker is not None:
            for item in previous_marker:
                canvas.delete(item)
        radius = 4
        return (
            canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline="yellow", width=2
            ),
            canvas.create_line(x - 8, y, x + 8, y, fill="yellow", width=1),
            canvas.create_line(x, y - 8, x, y + 8, fill="yellow", width=1),
        )

    def _draw_spectrum(self, spectrum, label_value, label_name, x, y):
        self.plot_canvas.delete("all")
        width = max(self.plot_canvas.winfo_width(), 700)
        height = max(self.plot_canvas.winfo_height(), 300)
        margin_left = 56
        margin_right = 22
        margin_top = 34
        margin_bottom = 44
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        values = np.asarray(spectrum, dtype=float)
        if self.y_scale_mode.get() == "fixed":
            v_min = 0.0
            v_max = 1.0
            scale_label = "fixed 0-1"
        else:
            v_min = float(np.min(values))
            v_max = float(np.max(values))
            scale_label = "auto min/max"
        if v_max == v_min:
            v_max = v_min + 1.0

        self.plot_canvas.create_text(
            margin_left,
            14,
            anchor="w",
            text=f"Band values at ({x}, {y}) - label {label_value}: {label_name} - {scale_label}",
            font=("TkDefaultFont", 11, "bold"),
        )

        x0, y0 = margin_left, margin_top + plot_h
        self.plot_canvas.create_line(x0, margin_top, x0, y0, fill="#444")
        self.plot_canvas.create_line(x0, y0, x0 + plot_w, y0, fill="#444")

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = y0 - frac * plot_h
            value = v_min + frac * (v_max - v_min)
            self.plot_canvas.create_line(x0 - 4, yy, x0 + plot_w, yy, fill="#e5e5e5")
            self.plot_canvas.create_text(x0 - 8, yy, anchor="e", text=f"{value:.3f}")

        points = []
        n = len(values)
        for i, value in enumerate(values):
            px = x0 + (i / (n - 1)) * plot_w if n > 1 else x0
            py = y0 - ((value - v_min) / (v_max - v_min)) * plot_h
            points.append((px, py))

        for idx in range(len(points) - 1):
            self.plot_canvas.create_line(*points[idx], *points[idx + 1], fill="#1f77b4", width=2)

        for idx, (px, py) in enumerate(points):
            self.plot_canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#1f77b4", outline="")
            if idx % 2 == 0:
                self.plot_canvas.create_text(px, y0 + 14, text=str(idx + 1), anchor="n")

        self.plot_canvas.create_text(x0 + plot_w / 2, height - 12, text="Band index", anchor="s")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HSI-Drive cubes with RGB, labels, and pixel spectra.")
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
        HSIVisualizer(root, args.data_dir)
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
