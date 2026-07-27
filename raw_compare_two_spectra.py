#!/usr/bin/env python3
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np


COLORS = ("#1f77b4", "#d62728")


class TwoSpectrumComparer:
    def __init__(self, root, data_dir):
        self.root = root
        self.data_dir = Path(data_dir)
        self.cube_dir = self.data_dir / "Cubes_Scaling"
        self.rgb_dir = self.data_dir / "RGB"

        self.files = self._find_cube_files()
        self.displayed_files = list(self.files)
        self.items = [None, None]
        self.markers = [None, None]

        self.root.title("HSI-Drive Two Spectrum Comparer")
        self.root.geometry("1260x760")
        self.root.minsize(1080, 680)
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
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.file_list = tk.Listbox(
            list_frame,
            width=42,
            height=30,
            selectmode=tk.EXTENDED,
            exportselection=False,
        )
        self.file_list.grid(row=0, column=0, sticky="ns")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self._refresh_listbox()

        controls = ttk.Frame(left)
        controls.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(controls, text="Load to image 1", command=lambda: self._load_single(0)).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(controls, text="Load to image 2", command=lambda: self._load_single(1)).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Button(controls, text="Load selected 2", command=self._load_selected).grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Button(controls, text="Clear points", command=self._clear_points).grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
        )

        self.status_var = tk.StringVar(value=f"{len(self.files)} files")
        ttk.Label(left, textvariable=self.status_var).grid(row=4, column=0, sticky="w", pady=(8, 0))

        scale_box = ttk.LabelFrame(left, text="Y-axis")
        scale_box.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        self.y_scale_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(
            scale_box,
            text="Auto combined min/max",
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

        main = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(3, weight=1)

        self.title_vars = [tk.StringVar(value="Image 1"), tk.StringVar(value="Image 2")]
        self.info_vars = [
            tk.StringVar(value="Load two files, then click a pixel."),
            tk.StringVar(value="Load two files, then click a pixel."),
        ]
        self.canvases = []
        for idx in range(2):
            ttk.Label(main, textvariable=self.title_vars[idx]).grid(row=0, column=idx, sticky="w")
            canvas = tk.Canvas(main, width=409, height=216, bg="#202020", highlightthickness=1)
            canvas.grid(row=1, column=idx, sticky="nw", padx=(0, 12) if idx == 0 else 0, pady=(6, 6))
            canvas.bind("<Button-1>", lambda event, image_idx=idx: self._on_image_click(event, image_idx))
            self.canvases.append(canvas)
            ttk.Label(main, textvariable=self.info_vars[idx]).grid(
                row=2,
                column=idx,
                sticky="w",
                padx=(0, 12) if idx == 0 else 0,
                pady=(0, 8),
            )

        plot_frame = ttk.Frame(main)
        plot_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.plot_canvas = tk.Canvas(plot_frame, height=360, bg="white", highlightthickness=1)
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

    def _load_selected(self):
        selected = self.file_list.curselection()
        if len(selected) != 2:
            messagebox.showinfo("Selection required", "Select exactly two Cubes_Scaling .npy files.")
            return

        try:
            for idx, list_idx in enumerate(selected):
                self._load_path_into_slot(self.displayed_files[list_idx], idx)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        self._draw_plot()

    def _load_single(self, idx):
        selected = self.file_list.curselection()
        if len(selected) != 1:
            messagebox.showinfo("Selection required", "Select exactly one Cubes_Scaling .npy file.")
            return

        try:
            self._load_path_into_slot(self.displayed_files[selected[0]], idx)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        self._draw_plot()

    def _load_path_into_slot(self, cube_path, idx):
        self.items[idx] = self._load_item(cube_path)
        self._show_image(idx)
        self.markers[idx] = None
        self.title_vars[idx].set(f"Image {idx + 1}: {self.items[idx]['base']}")
        self.info_vars[idx].set(
            f"cube {self.items[idx]['cube'].shape} {self.items[idx]['cube'].dtype}; click a pixel"
        )

    def _base_from_cube(self, cube_path):
        name = cube_path.name
        if name.endswith("_RC_TC.npy"):
            return name.removesuffix("_RC_TC.npy")
        return cube_path.stem

    def _load_item(self, cube_path):
        base = self._base_from_cube(cube_path)
        rgb_path = self.rgb_dir / f"{base}_pseudocolor.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")

        cube = np.load(cube_path)
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube, got {cube.shape}: {cube_path}")

        image = tk.PhotoImage(file=str(rgb_path))
        height, width = self._cube_hw(cube)
        if image.width() != width or image.height() != height:
            raise ValueError(
                f"Shape mismatch for {base}: cube={cube.shape}, RGB={image.width()}x{image.height()}"
            )

        return {
            "base": base,
            "cube_path": cube_path,
            "rgb_path": rgb_path,
            "cube": cube,
            "image": image,
            "x": None,
            "y": None,
            "spectrum": None,
        }

    def _cube_hw(self, cube):
        if cube.shape[0] == 25:
            return cube.shape[1], cube.shape[2]
        return cube.shape[0], cube.shape[1]

    def _spectrum_at(self, cube, x, y):
        if cube.shape[0] == 25:
            return cube[:, y, x]
        return cube[y, x, :]

    def _show_image(self, idx):
        item = self.items[idx]
        canvas = self.canvases[idx]
        image = item["image"]
        canvas.config(width=image.width(), height=image.height())
        canvas.delete("all")
        canvas.create_image(0, 0, image=image, anchor="nw")
        canvas.image = image

    def _on_image_click(self, event, idx):
        item = self.items[idx]
        if item is None:
            return

        x = int(event.x)
        y = int(event.y)
        height, width = self._cube_hw(item["cube"])
        if x < 0 or y < 0 or x >= width or y >= height:
            return

        spectrum = np.asarray(self._spectrum_at(item["cube"], x, y), dtype=float)
        item["x"] = x
        item["y"] = y
        item["spectrum"] = spectrum.copy()
        self.markers[idx] = self._draw_marker(self.canvases[idx], self.markers[idx], x, y, COLORS[idx])
        self.info_vars[idx].set(
            f"x={x}, y={y}; band min={float(np.min(spectrum)):.4f}, "
            f"max={float(np.max(spectrum)):.4f}"
        )
        self._draw_plot()

    def _draw_marker(self, canvas, previous_marker, x, y, color):
        if previous_marker is not None:
            for item_id in previous_marker:
                canvas.delete(item_id)
        radius = 4
        return (
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=2),
            canvas.create_line(x - 8, y, x + 8, y, fill=color, width=1),
            canvas.create_line(x, y - 8, x, y + 8, fill=color, width=1),
        )

    def _clear_points(self):
        for idx, item in enumerate(self.items):
            if item is not None:
                item["x"] = None
                item["y"] = None
                item["spectrum"] = None
                self.info_vars[idx].set(
                    f"cube {item['cube'].shape} {item['cube'].dtype}; click a pixel"
                )
            if self.markers[idx] is not None:
                for item_id in self.markers[idx]:
                    self.canvases[idx].delete(item_id)
                self.markers[idx] = None
        self._draw_plot()

    def _draw_plot(self):
        if not hasattr(self, "plot_canvas"):
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

        spectra = [item["spectrum"] for item in self.items if item is not None and item["spectrum"] is not None]
        if self.y_scale_mode.get() == "fixed":
            v_min = 0.0
            v_max = 1.0
            scale_label = "fixed 0-1"
        elif spectra:
            all_values = np.concatenate(spectra)
            v_min = float(np.nanmin(all_values))
            v_max = float(np.nanmax(all_values))
            scale_label = "auto combined min/max"
        else:
            v_min = 0.0
            v_max = 1.0
            scale_label = "auto combined min/max"
        if v_max == v_min:
            v_max = v_min + 1.0

        self.plot_canvas.create_text(
            margin_left,
            16,
            anchor="w",
            text=f"Selected pixel spectra - {scale_label}",
            font=("TkDefaultFont", 11, "bold"),
        )
        self.plot_canvas.create_line(x0, margin_top, x0, y0, fill="#444")
        self.plot_canvas.create_line(x0, y0, x0 + plot_w, y0, fill="#444")

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = y0 - frac * plot_h
            value = v_min + frac * (v_max - v_min)
            self.plot_canvas.create_line(x0 - 4, yy, x0 + plot_w, yy, fill="#e5e5e5")
            self.plot_canvas.create_text(x0 - 8, yy, anchor="e", text=f"{value:.3f}")

        if not spectra:
            self.plot_canvas.create_text(
                x0 + plot_w / 2,
                margin_top + plot_h / 2,
                text="Click one pixel in each loaded image.",
                fill="#666",
            )
            self.plot_canvas.create_text(x0 + plot_w / 2, height - 12, text="Band index", anchor="s")
            return

        max_bands = max(len(values) for values in spectra)
        for idx in range(max_bands):
            if idx % 2 == 0:
                px = x0 + (idx / (max_bands - 1)) * plot_w if max_bands > 1 else x0
                self.plot_canvas.create_text(px, y0 + 14, text=str(idx + 1), anchor="n")

        for idx, item in enumerate(self.items):
            if item is None or item["spectrum"] is None:
                continue
            self._draw_spectrum_line(item, idx, x0, y0, plot_w, plot_h, v_min, v_max)

        self.plot_canvas.create_text(x0 + plot_w / 2, height - 12, text="Band index", anchor="s")

    def _draw_spectrum_line(self, item, idx, x0, y0, plot_w, plot_h, v_min, v_max):
        values = item["spectrum"]
        color = COLORS[idx]
        n = len(values)
        points = []
        for band_idx, value in enumerate(values):
            px = x0 + (band_idx / (n - 1)) * plot_w if n > 1 else x0
            py = y0 - ((float(value) - v_min) / (v_max - v_min)) * plot_h
            points.append((px, py))

        for point_idx in range(len(points) - 1):
            self.plot_canvas.create_line(*points[point_idx], *points[point_idx + 1], fill=color, width=2)
        for px, py in points:
            self.plot_canvas.create_oval(px - 2.5, py - 2.5, px + 2.5, py + 2.5, fill=color, outline="")

        label = f"{idx + 1}: {item['base']} ({item['x']}, {item['y']})"
        self.plot_canvas.create_line(x0 + 12, 38 + idx * 20, x0 + 42, 38 + idx * 20, fill=color, width=3)
        self.plot_canvas.create_text(x0 + 50, 38 + idx * 20, anchor="w", text=label, fill="#222")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select two Cubes_Scaling images and compare clicked-pixel spectra in one graph."
    )
    parser.add_argument(
        "--data-dir",
        default="HSI_Drive",
        help="Directory containing Cubes_Scaling and RGB folders.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    try:
        TwoSpectrumComparer(root, args.data_dir)
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
