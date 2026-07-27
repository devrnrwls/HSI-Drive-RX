#!/usr/bin/env python3
import argparse
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


class RawVisualizer:
    def __init__(self, root, data_dir, raw_height, raw_width):
        self.root = root
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "RAW"
        self.rgb_dir = self.data_dir / "RGB"
        self.label_dir = self.data_dir / "Labels"
        self.cube_noscaling_dir = self.data_dir / "Cubes_NoScaling"
        self.cube_scaling_dir = self.data_dir / "Cubes_Scaling"
        self.raw_height = raw_height
        self.raw_width = raw_width

        self.raw = None
        self.raw_display = None
        self.rgb = None
        self.label = None
        self.cube_noscaling = None
        self.cube_scaling = None
        self.current_base = None
        self.raw_marker = None
        self.rgb_marker = None
        self.label_marker = None
        self.last_click = None
        self.last_noscaling_spectrum = None
        self.last_scaling_spectrum = None
        self.selected_noscaling_index = None
        self.selected_scaling_index = None

        self.root.title("HSI Drive RAW Visualizer")
        self.root.geometry("1280x980")
        self.root.minsize(1100, 860)

        self.files = self._find_raw_files()
        self._build_ui()
        if self.files:
            self.file_list.selection_set(0)
            self.file_list.event_generate("<<ListboxSelect>>")

    def _find_raw_files(self):
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"RAW directory not found: {self.raw_dir}")
        return sorted(self.raw_dir.glob("*.bin"))

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        left.rowconfigure(2, weight=1)

        ttk.Label(left, text="RAW bin files").grid(row=0, column=0, sticky="w")
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

        ttk.Label(left, text="Class definitions").grid(row=4, column=0, sticky="w", pady=(14, 4))
        legend = tk.Text(left, width=38, height=15, wrap="word", relief="solid", borderwidth=1)
        legend.grid(row=5, column=0, sticky="ew")
        for value, (name, description) in CLASS_DEFINITIONS.items():
            legend.insert(tk.END, f"{value}: {name}\n")
            legend.insert(tk.END, f"   {description}\n")
        legend.configure(state="disabled")

        main = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(6, weight=1)

        ttk.Label(main, text="RAW image - click a pixel").grid(row=0, column=0, sticky="w")
        ttk.Label(main, text="RGB image").grid(row=0, column=1, sticky="w")
        self.raw_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.raw_canvas.grid(row=1, column=0, sticky="nw", padx=(0, 12), pady=(6, 10))
        self.raw_canvas.bind("<Button-1>", self._on_display_click)

        self.rgb_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.rgb_canvas.grid(row=1, column=1, sticky="nw", pady=(6, 10))
        self.rgb_canvas.bind("<Button-1>", self._on_display_click)

        ttk.Label(main, text="Label image").grid(row=2, column=0, sticky="w")
        ttk.Label(main, text="RAW local 5x5 mosaic values").grid(row=2, column=1, sticky="w")
        self.label_canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
        self.label_canvas.grid(row=3, column=0, sticky="nw", padx=(0, 12), pady=(6, 10))
        self.label_canvas.bind("<Button-1>", self._on_display_click)

        self.patch_canvas = tk.Canvas(main, width=409, height=216, bg="white", highlightthickness=1)
        self.patch_canvas.grid(row=3, column=1, sticky="nw", pady=(6, 10))

        ttk.Label(main, text="Cubes_NoScaling spectrum").grid(row=4, column=0, sticky="w")
        ttk.Label(main, text="Cubes_Scaling spectrum").grid(row=4, column=1, sticky="w")
        self.cube_noscaling_canvas = tk.Canvas(main, width=409, height=216, bg="white", highlightthickness=1)
        self.cube_noscaling_canvas.grid(row=5, column=0, sticky="nw", padx=(0, 12), pady=(6, 10))
        self.cube_noscaling_canvas.bind("<Button-1>", lambda event: self._on_spectrum_click(event, "noscaling"))

        self.cube_scaling_canvas = tk.Canvas(main, width=409, height=216, bg="white", highlightthickness=1)
        self.cube_scaling_canvas.grid(row=5, column=1, sticky="nw", pady=(6, 10))
        self.cube_scaling_canvas.bind("<Button-1>", lambda event: self._on_spectrum_click(event, "scaling"))

        self.info_var = tk.StringVar(value="Select a RAW file, then click RAW/RGB/Label image.")
        ttk.Label(main, textvariable=self.info_var).grid(row=6, column=0, columnspan=2, sticky="nw")

        controls = ttk.Frame(main)
        controls.grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(controls, text="RAW display scale:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.raw_scale_mode = tk.StringVar(value="fixed")
        ttk.Radiobutton(
            controls,
            text="Fixed 0-4095",
            value="fixed",
            variable=self.raw_scale_mode,
            command=self._redraw_raw_image,
        ).grid(row=0, column=1, sticky="w", padx=(0, 10))
        ttk.Radiobutton(
            controls,
            text="Auto min/max",
            value="auto",
            variable=self.raw_scale_mode,
            command=self._redraw_raw_image,
        ).grid(row=0, column=2, sticky="w")

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
        raw_path = self.displayed_files[selection[0]]
        try:
            self._load_triplet(raw_path)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def _load_triplet(self, raw_path):
        base = raw_path.stem
        rgb_path = self.rgb_dir / f"{base}_pseudocolor.png"
        label_path = self.label_dir / f"{base}.png"
        cube_noscaling_path = self.cube_noscaling_dir / f"{base}_RC_TC.npy"
        cube_scaling_path = self.cube_scaling_dir / f"{base}_RC_TC.npy"

        if not rgb_path.is_file():
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        if not cube_noscaling_path.is_file():
            raise FileNotFoundError(f"Cubes_NoScaling file not found: {cube_noscaling_path}")
        if not cube_scaling_path.is_file():
            raise FileNotFoundError(f"Cubes_Scaling file not found: {cube_scaling_path}")

        raw_flat = np.fromfile(raw_path, dtype=np.uint16)
        expected_size = self.raw_height * self.raw_width
        if raw_flat.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} uint16 values for {self.raw_height}x{self.raw_width}, "
                f"got {raw_flat.size}: {raw_path}"
            )
        raw = raw_flat.reshape((self.raw_height, self.raw_width))

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None:
            raise ValueError(f"Failed to read RGB file: {rgb_path}")
        if label is None:
            raise ValueError(f"Failed to read label file: {label_path}")
        cube_noscaling = np.load(cube_noscaling_path)
        cube_scaling = np.load(cube_scaling_path)
        self._validate_cube(cube_noscaling, cube_noscaling_path, label.shape[:2])
        self._validate_cube(cube_scaling, cube_scaling_path, label.shape[:2])

        self.current_base = base
        self.raw = raw
        self.rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        self.label = label
        self.cube_noscaling = cube_noscaling
        self.cube_scaling = cube_scaling
        self.raw_display = None
        self.raw_marker = None
        self.rgb_marker = None
        self.label_marker = None
        self.last_click = None
        self.last_noscaling_spectrum = None
        self.last_scaling_spectrum = None
        self.selected_noscaling_index = None
        self.selected_scaling_index = None

        self._redraw_raw_image()
        self._show_image(self.rgb_canvas, self.rgb)
        self._show_image(self.label_canvas, self._label_to_rgb(label))
        self.patch_canvas.delete("all")
        self.cube_noscaling_canvas.delete("all")
        self.cube_scaling_canvas.delete("all")
        self._draw_empty_spectrum(self.cube_noscaling_canvas, "Click an image pixel to show NoScaling spectrum")
        self._draw_empty_spectrum(self.cube_scaling_canvas, "Click an image pixel to show Scaling spectrum")
        self.info_var.set(
            f"{base} | RAW {raw.shape} {raw.dtype} min={int(raw.min())} max={int(raw.max())} | "
            f"RGB {self.rgb.shape} | label {label.shape} | "
            f"NoScaling {cube_noscaling.shape} | Scaling {cube_scaling.shape}"
        )

    def _validate_cube(self, cube, path, label_shape):
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube, got {cube.shape}: {path}")
        if cube.shape[0] == 25:
            height, width = cube.shape[1], cube.shape[2]
        else:
            height, width = cube.shape[0], cube.shape[1]
        if label_shape != (height, width):
            raise ValueError(f"Cube/label shape mismatch: cube={cube.shape}, label={label_shape}: {path}")

    def _redraw_raw_image(self):
        if self.raw is None:
            return
        if self.raw_scale_mode.get() == "auto":
            raw_min = float(self.raw.min())
            raw_max = float(self.raw.max())
        else:
            raw_min = 0.0
            raw_max = 4095.0
        if raw_max == raw_min:
            raw_max = raw_min + 1.0
        normalized = (self.raw.astype(np.float32) - raw_min) / (raw_max - raw_min)
        gray = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        display_h, display_w = self.label.shape[:2] if self.label is not None else (216, 409)
        gray_small = cv2.resize(gray, (display_w, display_h), interpolation=cv2.INTER_AREA)
        self.raw_display = np.stack([gray_small, gray_small, gray_small], axis=-1)
        self._show_image(self.raw_canvas, self.raw_display)
        self.raw_marker = None
        if self.last_click is not None:
            display_x, display_y, _, _ = self.last_click
            self.raw_marker = self._draw_marker(self.raw_canvas, self.raw_marker, display_x, display_y)

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
        for value in np.unique(label):
            out[label == value] = LABEL_COLORS.get(int(value), (255, 255, 255))
        return out

    def _on_display_click(self, event):
        if self.raw is None:
            return
        display_h, display_w = self.label.shape[:2]
        display_x = int(event.x)
        display_y = int(event.y)
        if display_x < 0 or display_y < 0 or display_x >= display_w or display_y >= display_h:
            return

        raw_x = min(int(round(display_x * (self.raw_width - 1) / max(display_w - 1, 1))), self.raw_width - 1)
        raw_y = min(int(round(display_y * (self.raw_height - 1) / max(display_h - 1, 1))), self.raw_height - 1)
        label_value = int(self.label[display_y, display_x])
        label_name, label_description = CLASS_DEFINITIONS.get(label_value, ("Unknown", "Unknown label value"))
        rgb_value = tuple(int(v) for v in self.rgb[display_y, display_x])
        raw_value = int(self.raw[raw_y, raw_x])

        self.last_click = (display_x, display_y, raw_x, raw_y)
        self.raw_marker = self._draw_marker(self.raw_canvas, self.raw_marker, display_x, display_y)
        self.rgb_marker = self._draw_marker(self.rgb_canvas, self.rgb_marker, display_x, display_y)
        self.label_marker = self._draw_marker(self.label_canvas, self.label_marker, display_x, display_y)
        self._draw_patch(raw_x, raw_y)

        noscaling_spectrum = self._spectrum_at(self.cube_noscaling, display_x, display_y)
        scaling_spectrum = self._spectrum_at(self.cube_scaling, display_x, display_y)
        self.last_noscaling_spectrum = np.asarray(noscaling_spectrum, dtype=float).copy()
        self.last_scaling_spectrum = np.asarray(scaling_spectrum, dtype=float).copy()
        self.selected_noscaling_index = None
        self.selected_scaling_index = None
        self._draw_spectrum(
            self.cube_noscaling_canvas,
            self.last_noscaling_spectrum,
            f"Cubes_NoScaling spectrum at ({display_x}, {display_y})",
            fixed_range=None,
            line_color="#1f77b4",
            selected_index=self.selected_noscaling_index,
        )
        self._draw_spectrum(
            self.cube_scaling_canvas,
            self.last_scaling_spectrum,
            f"Cubes_Scaling spectrum at ({display_x}, {display_y})",
            fixed_range=None,
            line_color="#2ca02c",
            selected_index=self.selected_scaling_index,
        )

        self.info_var.set(
            f"{self.current_base} | display x={display_x}, y={display_y} | "
            f"RAW x={raw_x}, y={raw_y}, value={raw_value} | "
            f"label={label_value}: {label_name} ({label_description}) | RGB={rgb_value} | "
            f"NoScaling band min/max={float(noscaling_spectrum.min()):.4f}/"
            f"{float(noscaling_spectrum.max()):.4f} | "
            f"Scaling band min/max={float(scaling_spectrum.min()):.4f}/"
            f"{float(scaling_spectrum.max()):.4f}"
        )

    def _spectrum_at(self, cube, x, y):
        if cube.shape[0] == 25:
            return cube[:, y, x]
        return cube[y, x, :]

    def _draw_empty_spectrum(self, canvas, message):
        canvas.config(width=409, height=216)
        canvas.delete("all")
        canvas.create_text(204, 108, text=message, fill="#555", anchor="center")

    def _on_spectrum_click(self, event, kind):
        if kind == "noscaling":
            values = self.last_noscaling_spectrum
            title = self._spectrum_title("Cubes_NoScaling")
            canvas = self.cube_noscaling_canvas
            line_color = "#1f77b4"
        else:
            values = self.last_scaling_spectrum
            title = self._spectrum_title("Cubes_Scaling")
            canvas = self.cube_scaling_canvas
            line_color = "#2ca02c"
        if values is None:
            return

        index = self._band_index_from_canvas_x(event.x, len(values))
        if kind == "noscaling":
            self.selected_noscaling_index = index
            selected_index = self.selected_noscaling_index
        else:
            self.selected_scaling_index = index
            selected_index = self.selected_scaling_index

        self._draw_spectrum(
            canvas,
            values,
            title,
            fixed_range=None,
            line_color=line_color,
            selected_index=selected_index,
        )

    def _spectrum_title(self, cube_name):
        if self.last_click is None:
            return f"{cube_name} spectrum"
        display_x, display_y, _, _ = self.last_click
        return f"{cube_name} spectrum at ({display_x}, {display_y})"

    def _band_index_from_canvas_x(self, x, num_bands):
        margin_left = 52
        margin_right = 16
        width = 409
        plot_w = width - margin_left - margin_right
        if num_bands <= 1:
            return 0
        ratio = (x - margin_left) / plot_w
        index = int(round(ratio * (num_bands - 1)))
        return max(0, min(num_bands - 1, index))

    def _draw_spectrum(self, canvas, spectrum, title, fixed_range=None, line_color="#1f77b4", selected_index=None):
        canvas.config(width=409, height=216)
        canvas.delete("all")

        width = 409
        height = 216
        margin_left = 52
        margin_right = 16
        margin_top = 30
        margin_bottom = 34
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        values = np.asarray(spectrum, dtype=float)
        if fixed_range is None:
            v_min = float(np.nanmin(values))
            v_max = float(np.nanmax(values))
            scale_label = "auto"
        else:
            v_min, v_max = fixed_range
            scale_label = f"{v_min:g}-{v_max:g}"
        if v_max == v_min:
            v_max = v_min + 1.0

        canvas.create_text(
            margin_left,
            10,
            anchor="w",
            text=f"{title} | y={scale_label}",
            font=("TkDefaultFont", 9, "bold"),
        )

        x0 = margin_left
        y0 = margin_top + plot_h
        canvas.create_line(x0, margin_top, x0, y0, fill="#444")
        canvas.create_line(x0, y0, x0 + plot_w, y0, fill="#444")

        for frac in (0.0, 0.5, 1.0):
            yy = y0 - frac * plot_h
            value = v_min + frac * (v_max - v_min)
            canvas.create_line(x0 - 4, yy, x0 + plot_w, yy, fill="#e6e6e6")
            canvas.create_text(x0 - 8, yy, anchor="e", text=f"{value:.3f}", font=("TkDefaultFont", 8))

        points = []
        n = len(values)
        for index, value in enumerate(values):
            px = x0 + (index / (n - 1)) * plot_w if n > 1 else x0
            py = y0 - ((value - v_min) / (v_max - v_min)) * plot_h
            py = max(margin_top, min(y0, py))
            points.append((px, py))

        for index in range(len(points) - 1):
            canvas.create_line(*points[index], *points[index + 1], fill=line_color, width=2)

        for index, (px, py) in enumerate(points):
            canvas.create_oval(px - 2, py - 2, px + 2, py + 2, fill=line_color, outline="")
            if index % 4 == 0:
                canvas.create_text(px, y0 + 10, text=str(index + 1), anchor="n", font=("TkDefaultFont", 8))

        if selected_index is not None:
            selected_index = max(0, min(len(values) - 1, int(selected_index)))
            px, py = points[selected_index]
            value = float(values[selected_index])
            canvas.create_line(px, margin_top, px, y0, fill="#d62728", dash=(3, 2))
            canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#d62728", outline="")
            label_x = min(max(px + 8, margin_left + 70), width - 92)
            label_y = max(py - 18, margin_top + 14)
            canvas.create_text(
                label_x,
                label_y,
                anchor="w",
                text=f"band {selected_index + 1}: {value:.6f}",
                fill="#d62728",
                font=("TkDefaultFont", 8, "bold"),
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

    def _draw_patch(self, raw_x, raw_y):
        self.patch_canvas.delete("all")
        radius = 2
        y0 = max(raw_y - radius, 0)
        y1 = min(raw_y + radius + 1, self.raw_height)
        x0 = max(raw_x - radius, 0)
        x1 = min(raw_x + radius + 1, self.raw_width)
        patch = self.raw[y0:y1, x0:x1]

        canvas_w = 409
        canvas_h = 216
        self.patch_canvas.config(width=canvas_w, height=canvas_h)
        self.patch_canvas.create_text(
            12,
            12,
            anchor="nw",
            text=f"RAW 5x5 around ({raw_x}, {raw_y})",
            font=("TkDefaultFont", 11, "bold"),
        )

        cell = 34
        start_x = 24
        start_y = 46
        p_min = float(patch.min())
        p_max = float(patch.max())
        if p_max == p_min:
            p_max = p_min + 1.0

        for row in range(patch.shape[0]):
            for col in range(patch.shape[1]):
                value = int(patch[row, col])
                shade = int(255 * (value - p_min) / (p_max - p_min))
                color = f"#{shade:02x}{shade:02x}{shade:02x}"
                x = start_x + col * cell
                y = start_y + row * cell
                outline = "red" if (y0 + row == raw_y and x0 + col == raw_x) else "#777"
                width = 2 if outline == "red" else 1
                self.patch_canvas.create_rectangle(x, y, x + cell, y + cell, fill=color, outline=outline, width=width)
                text_color = "white" if shade < 120 else "black"
                self.patch_canvas.create_text(x + cell / 2, y + cell / 2, text=str(value), fill=text_color)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HSI-Drive RAW bin files with RGB and labels.")
    parser.add_argument(
        "--data-dir",
        default="HSI_Drive",
        help="Directory containing RAW, RGB, and Labels folders.",
    )
    parser.add_argument("--raw-height", type=int, default=1088, help="RAW image height.")
    parser.add_argument("--raw-width", type=int, default=2048, help="RAW image width.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    try:
        RawVisualizer(root, args.data_dir, args.raw_height, args.raw_width)
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
