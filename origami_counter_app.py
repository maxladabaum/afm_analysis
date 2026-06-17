from __future__ import annotations

import csv
import io
import json
import math
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, Button, Canvas, Entry, Frame, Label, Listbox, Menu, Scrollbar, StringVar, Tk, filedialog, messagebox, ttk

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk
from skimage import color, exposure, filters, measure, morphology, transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


APP_TITLE = "DNA Origami AFM Counter"
OUTPUT_DIR = "analysis_output"
LABELS_FILE = "labels.json"
MODEL_FILE = "origami_state_classifier.joblib"
COUNTS_FILE = "origami_counts.csv"
CURRENT_RESULTS_DIR = "classified_images"
PLOTS_DIR = "plots"
THUMB_MAX = 980
MODEL_FORMAT_VERSION = 2
UNLABELED_COLOR = "#45a3ff"
STATE_COLORS = [
    "#ff5a5f",
    "#2ec4b6",
    "#ffbe0b",
    "#8338ec",
    "#3a86ff",
    "#fb5607",
    "#06d6a0",
    "#ef476f",
    "#118ab2",
    "#9b5de5",
    "#00bbf9",
    "#f15bb5",
]


@dataclass
class OrigamiObject:
    object_id: int
    bbox: tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    centroid: tuple[float, float]
    area: float
    features: list[float]
    label: str | None = None
    prediction: str | None = None
    confidence: float | None = None


@dataclass
class ScaleInfo:
    pixels_per_um: float
    bar_pixels: float
    bar_um: float
    detected: bool
    source: str = "auto"


def image_key(path: Path) -> str:
    return str(path.resolve())


def discover_images(root: Path) -> list[Path]:
    images = []
    for path in root.rglob("origami*.png"):
        if path.is_file():
            images.append(path)
    return sorted(images, key=lambda p: (p.parent.name, p.name))


def state_color(state: str, states: list[str]) -> str:
    if state in states:
        return STATE_COLORS[states.index(state) % len(STATE_COLORS)]
    return STATE_COLORS[abs(hash(state)) % len(STATE_COLORS)]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    arr = color.rgb2gray(rgb).astype(np.float32)
    return exposure.rescale_intensity(arr, in_range="image", out_range=(0.0, 1.0))


def paired_spm_path(png_path: Path) -> Path:
    if png_path.name.lower().endswith(".spm.png"):
        candidate = Path(str(png_path)[:-4])
        if candidate.exists():
            return candidate
        name = png_path.name
        if name.lower().endswith("_1.spm.png"):
            return png_path.with_name(name[:-10] + ".spm")
        return candidate
    return png_path.with_suffix(".spm")


def parse_spm_scan_size_um(spm_path: Path) -> float | None:
    if not spm_path.exists():
        return None
    pattern = re.compile(r"\\Scan Size:\s*([0-9.]+)\s*(nm|um|µm|m)?", re.IGNORECASE)
    try:
        with spm_path.open("r", encoding="latin-1", errors="ignore") as f:
            for line in f:
                match = pattern.search(line)
                if not match:
                    continue
                value = float(match.group(1))
                unit = (match.group(2) or "nm").lower()
                if unit == "nm":
                    return value / 1000.0
                if unit in {"um", "µm"}:
                    return value
                if unit == "m":
                    return value * 1_000_000.0
    except OSError:
        return None
    return None


def format_um(value: float) -> str:
    if value >= 1:
        text = f"{value:.3f}"
    else:
        text = f"{value:.4f}"
    return text.rstrip("0").rstrip(".")


def detect_scale_bar(rgb: np.ndarray, bar_um: float = 1.0) -> ScaleInfo:
    """Detect the black horizontal scale bar and convert pixels to microns."""
    h, w = rgb.shape[:2]
    scan_minr, scan_minc, scan_maxr, scan_maxc = scan_bbox(rgb)
    dark = (rgb[:, :, 0] < 35) & (rgb[:, :, 1] < 35) & (rgb[:, :, 2] < 35)
    search = np.zeros_like(dark)
    row_start = max(0, scan_maxr)
    row_end = min(h, scan_maxr + 35)
    col_start = max(0, scan_minc - 5)
    col_end = min(w, scan_maxc + 5)
    search[row_start:row_end, col_start:col_end] = dark[row_start:row_end, col_start:col_end]
    labels = measure.label(search)
    candidates = []
    for prop in measure.regionprops(labels):
        minr, minc, maxr, maxc = prop.bbox
        height = maxr - minr
        width = maxc - minc
        if width >= 25 and 2 <= height <= 12 and width / max(height, 1) >= 8:
            candidates.append((width, prop.area, prop.bbox))
    if candidates:
        width, _area, _bbox = max(candidates, key=lambda item: item[0])
        return ScaleInfo(pixels_per_um=float(width) / max(bar_um, 1e-9), bar_pixels=float(width), bar_um=bar_um, detected=True, source="bar")

    fallback_px_per_um = float(scan_maxc - scan_minc) / 10.0
    return ScaleInfo(pixels_per_um=fallback_px_per_um, bar_pixels=fallback_px_per_um * bar_um, bar_um=bar_um, detected=False, source="fallback")


def scale_info_from_spm(rgb: np.ndarray, png_path: Path) -> ScaleInfo | None:
    scan_size_um = parse_spm_scan_size_um(paired_spm_path(png_path))
    if scan_size_um is None or scan_size_um <= 0:
        return None
    scan_minr, scan_minc, scan_maxr, scan_maxc = scan_bbox(rgb)
    scan_width_px = max(1, scan_maxc - scan_minc)
    bar_um = scan_size_um / 5.0
    return ScaleInfo(
        pixels_per_um=scan_width_px / scan_size_um,
        bar_pixels=scan_width_px / 5.0,
        bar_um=bar_um,
        detected=True,
        source="spm",
    )


def component_features(gray: np.ndarray, prop: measure._regionprops.RegionProperties, pixels_per_um: float) -> list[float]:
    minr, minc, maxr, maxc = prop.bbox
    patch = gray[minr:maxr, minc:maxc]
    mask = prop.image
    values = patch[mask]
    h = maxr - minr
    w = maxc - minc
    perimeter = float(prop.perimeter or 0.0)
    area = float(prop.area)
    px_per_um = max(float(pixels_per_um), 1e-9)
    area_um2 = area / (px_per_um * px_per_um)
    perimeter_um = perimeter / px_per_um
    circularity = 4.0 * math.pi * area / (perimeter * perimeter + 1e-6)
    intensity_hist, _ = np.histogram(values, bins=8, range=(0, 1), density=True)
    resized = transform.resize(patch, (16, 16), anti_aliasing=True, preserve_range=True)
    major_axis = prop.axis_major_length if hasattr(prop, "axis_major_length") else prop.major_axis_length
    minor_axis = prop.axis_minor_length if hasattr(prop, "axis_minor_length") else prop.minor_axis_length
    return [
        area_um2,
        float(major_axis) / px_per_um,
        float(minor_axis) / px_per_um,
        perimeter_um,
        float(w) / px_per_um,
        float(h) / px_per_um,
        float(prop.eccentricity),
        float(prop.solidity),
        float(prop.extent),
        float(prop.orientation),
        float(w / max(h, 1)),
        float(circularity),
        float(values.mean()) if values.size else 0.0,
        float(values.std()) if values.size else 0.0,
        float(np.percentile(values, 10)) if values.size else 0.0,
        float(np.percentile(values, 50)) if values.size else 0.0,
        float(np.percentile(values, 90)) if values.size else 0.0,
        *[float(x) for x in intensity_hist],
        *[float(x) for x in resized.flatten()[::8]],
    ]


def scan_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Find the plotted AFM scan area inside exported PNGs with labels/scale bars."""
    nonwhite = ~((rgb[:, :, 0] > 245) & (rgb[:, :, 1] > 245) & (rgb[:, :, 2] > 245))
    labels = measure.label(nonwhite)
    candidates = []
    total = rgb.shape[0] * rgb.shape[1]
    for prop in measure.regionprops(labels):
        minr, minc, maxr, maxc = prop.bbox
        h = maxr - minr
        w = maxc - minc
        if prop.area > total * 0.05 and h > 100 and w > 100:
            candidates.append(prop)
    if not candidates:
        return (0, 0, rgb.shape[0], rgb.shape[1])
    prop = max(candidates, key=lambda r: r.area)
    return tuple(int(v) for v in prop.bbox)


def detect_origami(rgb: np.ndarray, min_area: int, max_area: int, threshold_bias: float, pixels_per_um: float | None = None) -> list[OrigamiObject]:
    minr0, minc0, maxr0, maxc0 = scan_bbox(rgb)
    if pixels_per_um is None:
        pixels_per_um = detect_scale_bar(rgb).pixels_per_um
    gray_full = rgb_to_grayscale(rgb)
    gray = gray_full[minr0:maxr0, minc0:maxc0]
    smooth = filters.gaussian(gray, sigma=1.0)
    try:
        threshold = filters.threshold_otsu(smooth)
    except ValueError:
        threshold = float(smooth.mean())
    binary_high = smooth > min(1.0, threshold + threshold_bias)
    binary_low = smooth < max(0.0, threshold - threshold_bias)
    binary = binary_high if binary_high.sum() <= binary_low.sum() else binary_low
    binary = morphology.remove_small_objects(binary, max_size=max(4, min_area // 2))
    binary = morphology.remove_small_holes(binary, max_size=max(8, min_area // 2))
    labels = measure.label(binary)

    objects: list[OrigamiObject] = []
    for prop in measure.regionprops(labels):
        if prop.area < min_area or prop.area > max_area:
            continue
        minr, minc, maxr, maxc = prop.bbox
        if minr <= 1 or minc <= 1 or maxr >= gray.shape[0] - 1 or maxc >= gray.shape[1] - 1:
            continue
        obj = OrigamiObject(
            object_id=len(objects) + 1,
            bbox=(int(minr + minr0), int(minc + minc0), int(maxr + minr0), int(maxc + minc0)),
            centroid=(float(prop.centroid[0] + minr0), float(prop.centroid[1] + minc0)),
            area=float(prop.area),
            features=component_features(gray, prop, pixels_per_um),
        )
        objects.append(obj)
    return objects


class OrigamiCounterApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1360x850")

        self.workspace = Path.cwd()
        self.output_dir = self.workspace / OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        self.labels_path = self.output_dir / LABELS_FILE
        self.model_path = self.output_dir / MODEL_FILE

        self.images: list[Path] = []
        self.current_image: Path | None = None
        self.rgb: np.ndarray | None = None
        self.objects: list[OrigamiObject] = []
        self.labels: dict[str, dict[str, str]] = {}
        self.training_labels: dict[str, dict[str, str]] = {}
        self.scale_calibrations: dict[str, dict[str, float]] = {}
        self.states: list[str] = self.load_states()
        self.current_state = StringVar(value=self.states[0] if self.states else "A")
        self.model: Pipeline | None = None
        self.scale = 1.0
        self.image_offset = (0, 0)
        self.fit_to_window = True
        self.tk_image: ImageTk.PhotoImage | None = None
        self.overlay_ids: list[int] = []
        self.scan_bounds: tuple[int, int, int, int] | None = None
        self.area_box_mode = False
        self.scale_box_mode = False
        self.drag_start: tuple[int, int] | None = None
        self.drag_preview_id: int | None = None
        self.pan_start: tuple[int, int] | None = None

        self.min_area = StringVar(value="30")
        self.max_area = StringVar(value="2000")
        self.threshold_bias = StringVar(value="0.00")
        self.status = StringVar(value="Open a root folder to begin.")
        self.counts_text = StringVar(value="")
        self.training_status = StringVar(value="")
        self.scale_bar_um = StringVar(value="1.0")
        self.scale_status = StringVar(value="")
        self.analysis_status = StringVar(value="Load an all_images folder to begin.")
        self.analysis_rows: list[dict[str, str | float]] = []
        self.analysis_csv_path: Path | None = None
        self.plot_previews: dict[str, Image.Image] = {}
        self.plot_preview_photo: ImageTk.PhotoImage | None = None
        self.min_area_um2: float | None = None
        self.max_area_um2: float | None = None

        self.build_ui()
        self.open_root(self.workspace)

    def build_ui(self) -> None:
        menubar = Menu(self.root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Root", command=self.choose_root)
        file_menu.add_command(label="Save Labels", command=self.save_labels)
        file_menu.add_command(label="Export Counts", command=self.batch_count)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=BOTH, expand=True)
        classify_tab = Frame(notebook)
        analysis_tab = Frame(notebook)
        notebook.add(classify_tab, text="Classify")
        notebook.add(analysis_tab, text="Analysis")

        main = Frame(classify_tab)
        main.pack(fill=BOTH, expand=True)

        sidebar_container = Frame(main, width=350)
        sidebar_container.pack(side=LEFT, fill="y", padx=8, pady=8)
        sidebar_container.pack_propagate(False)

        self.sidebar_canvas = Canvas(sidebar_container, highlightthickness=0)
        sidebar_scroll = Scrollbar(sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        self.sidebar_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        sidebar_scroll.pack(side=RIGHT, fill="y")

        sidebar = Frame(self.sidebar_canvas)
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", self.on_sidebar_configure)
        self.sidebar_canvas.bind("<Configure>", self.on_sidebar_canvas_configure)
        self.root.bind_all("<MouseWheel>", self.on_global_mousewheel, add="+")

        Button(sidebar, text="Open Root", command=self.choose_root).pack(fill="x")
        Button(sidebar, text="Load Classifier", command=self.load_classifier_from_file).pack(fill="x", pady=(4, 0))
        Button(sidebar, text="Export Classifier", command=self.export_classifier).pack(fill="x", pady=(4, 0))
        Button(sidebar, text="Plot Counts CSV", command=self.plot_counts_csv).pack(fill="x", pady=(4, 0))
        Label(sidebar, text="Images").pack(anchor="w", pady=(10, 2))
        image_list_frame = Frame(sidebar)
        image_list_frame.pack(fill="x")
        self.image_list = Listbox(image_list_frame, height=10, exportselection=False)
        image_scroll = Scrollbar(image_list_frame, orient="vertical", command=self.image_list.yview)
        self.image_list.configure(yscrollcommand=image_scroll.set)
        self.image_list.pack(side=LEFT, fill="x", expand=True)
        image_scroll.pack(side=RIGHT, fill="y")
        self.image_list.bind("<<ListboxSelect>>", self.on_image_select)
        self.image_list.bind("<MouseWheel>", self.on_image_list_mousewheel)

        labels_box = ttk.LabelFrame(sidebar, text="Training Data Images")
        labels_box.pack(fill="x", pady=8)
        label_list_frame = Frame(labels_box)
        label_list_frame.pack(fill="x", padx=6, pady=(4, 2))
        self.label_summary = Listbox(label_list_frame, height=5, exportselection=False)
        label_scroll = Scrollbar(label_list_frame, orient="vertical", command=self.label_summary.yview)
        self.label_summary.configure(yscrollcommand=label_scroll.set)
        self.label_summary.pack(side=LEFT, fill="x", expand=True)
        label_scroll.pack(side=RIGHT, fill="y")
        self.label_summary.bind("<Double-Button-1>", self.load_selected_labeled_image)
        self.label_summary.bind("<MouseWheel>", self.on_label_summary_mousewheel)
        row = Frame(labels_box)
        row.pack(fill="x", padx=6, pady=(2, 6))
        Button(row, text="Open", command=self.load_selected_labeled_image).pack(side=LEFT, fill="x", expand=True)
        Button(row, text="Clear", command=self.clear_selected_labeled_image).pack(side=RIGHT, fill="x", expand=True, padx=(4, 0))

        params = ttk.LabelFrame(sidebar, text="Detection")
        params.pack(fill="x", pady=8)
        self.add_labeled_entry(params, "Scale bar um", self.scale_bar_um)
        Label(params, textvariable=self.scale_status, justify=LEFT).pack(fill="x", padx=6, pady=(0, 4))
        self.add_labeled_entry(params, "Min area", self.min_area)
        self.add_labeled_entry(params, "Max area", self.max_area)
        self.add_labeled_entry(params, "Threshold bias", self.threshold_bias)
        Button(params, text="Box Scale", command=self.start_scale_box).pack(fill="x", padx=6, pady=(4, 0))
        Button(params, text="Box Area", command=self.start_area_box).pack(fill="x", padx=6, pady=(4, 0))
        Button(params, text="Detect", command=self.detect_current).pack(fill="x", padx=6, pady=4)

        state_box = ttk.LabelFrame(sidebar, text="States")
        state_box.pack(fill="x", pady=8)
        self.state_combo = ttk.Combobox(state_box, textvariable=self.current_state, values=self.states, state="readonly")
        self.state_combo.pack(fill="x", padx=6, pady=4)
        row = Frame(state_box)
        row.pack(fill="x", padx=6, pady=4)
        self.new_state = StringVar()
        Entry(row, textvariable=self.new_state).pack(side=LEFT, fill="x", expand=True)
        Button(row, text="Add", command=self.add_state).pack(side=RIGHT, padx=(4, 0))
        Button(state_box, text="Clear Selected Image Labels", command=self.clear_image_labels).pack(fill="x", padx=6, pady=4)

        train_box = ttk.LabelFrame(sidebar, text="Classifier")
        train_box.pack(fill="x", pady=8)
        Button(train_box, text="Add Current Image Labels to Training Data", command=self.add_current_labels_to_training).pack(fill="x", padx=6, pady=(4, 2))
        Button(train_box, text="Train Current Image", command=self.train_current_image_model).pack(fill="x", padx=6, pady=2)
        Button(train_box, text="Train All Labeled Images", command=self.train_model).pack(fill="x", padx=6, pady=(2, 4))
        self.training_progress = ttk.Progressbar(train_box, mode="indeterminate")
        self.training_progress.pack(fill="x", padx=6, pady=(0, 4))
        Label(train_box, textvariable=self.training_status, justify=LEFT).pack(fill="x", padx=6)
        Button(train_box, text="Classify Image", command=self.classify_current).pack(fill="x", padx=6, pady=4)
        Button(train_box, text="Export Current Results", command=self.export_current_results).pack(fill="x", padx=6, pady=4)
        Button(train_box, text="Classify + Export All Images", command=self.classify_export_all_images).pack(fill="x", padx=6, pady=4)
        Button(train_box, text="Batch Count", command=self.batch_count).pack(fill="x", padx=6, pady=4)
        Label(train_box, textvariable=self.counts_text, justify=LEFT).pack(fill="x", padx=6, pady=6)

        viewer = Frame(main)
        viewer.pack(side=RIGHT, fill=BOTH, expand=True, padx=(0, 8), pady=8)
        toolbar = Frame(viewer)
        toolbar.pack(side=TOP, fill="x")
        Button(toolbar, text="Zoom +", command=lambda: self.zoom(1.25)).pack(side=LEFT)
        Button(toolbar, text="Zoom -", command=lambda: self.zoom(0.8)).pack(side=LEFT, padx=4)
        Button(toolbar, text="Fit", command=self.fit_image).pack(side=LEFT)
        Label(toolbar, text="Pan: right-drag or scrollbars").pack(side=LEFT, padx=(8, 0))
        Label(toolbar, textvariable=self.status).pack(side=LEFT, padx=12)

        canvas_frame = Frame(viewer)
        canvas_frame.pack(fill=BOTH, expand=True)
        self.canvas = Canvas(canvas_frame, background="#202124", highlightthickness=0)
        h_scroll = Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        v_scroll = Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_canvas)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.pan_canvas)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        self.build_analysis_tab(analysis_tab)

    def add_labeled_entry(self, parent: Frame, label: str, var: StringVar) -> None:
        row = Frame(parent)
        row.pack(fill="x", padx=6, pady=3)
        Label(row, text=label, width=14, anchor="w").pack(side=LEFT)
        Entry(row, textvariable=var, width=10).pack(side=RIGHT)

    def build_analysis_tab(self, parent: Frame) -> None:
        toolbar = Frame(parent)
        toolbar.pack(fill="x", padx=8, pady=8)
        Button(toolbar, text="Load all_images Folder", command=self.load_analysis_folder).pack(side=LEFT)
        Button(toolbar, text="Generate Plot Previews", command=self.generate_analysis_plot_previews).pack(side=LEFT, padx=6)
        Button(toolbar, text="Save Selected Plot", command=self.save_selected_plot_preview).pack(side=LEFT)
        Button(toolbar, text="Save All Plots", command=self.save_all_plot_previews).pack(side=LEFT, padx=6)
        Label(toolbar, textvariable=self.analysis_status).pack(side=LEFT, padx=12)

        summary_box = ttk.LabelFrame(parent, text="Summary")
        summary_box.pack(fill="x", padx=8, pady=(0, 8))
        self.analysis_summary = Listbox(summary_box, height=6, exportselection=False)
        self.analysis_summary.pack(fill="x", padx=6, pady=6)

        plot_box = ttk.LabelFrame(parent, text="Plot Preview")
        plot_box.pack(fill="x", padx=8, pady=(0, 8))
        plot_inner = Frame(plot_box)
        plot_inner.pack(fill="x", padx=6, pady=6)
        self.plot_list = Listbox(plot_inner, height=6, exportselection=False)
        self.plot_list.pack(side=LEFT, fill="y")
        self.plot_list.bind("<<ListboxSelect>>", self.on_plot_preview_select)
        self.plot_preview_label = Label(plot_inner, text="Generate plot previews to view them here.", background="#f0f0f0", width=90, height=18)
        self.plot_preview_label.pack(side=LEFT, fill="both", expand=True, padx=(8, 0))

        table_frame = Frame(parent)
        table_frame.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))
        columns = ("kind", "origami_label", "image", "total", "primary_fraction", "secondary_fraction", "primary_secondary_ratio")
        self.analysis_table = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "kind": "Row",
            "origami_label": "Origami",
            "image": "Image",
            "total": "Total",
            "primary_fraction": "Frac 1st State",
            "secondary_fraction": "Frac 2nd State",
            "primary_secondary_ratio": "1st/2nd Ratio",
        }
        widths = {
            "kind": 90,
            "origami_label": 90,
            "image": 260,
            "total": 80,
            "primary_fraction": 110,
            "secondary_fraction": 110,
            "primary_secondary_ratio": 110,
        }
        for col in columns:
            self.analysis_table.heading(col, text=headings[col])
            self.analysis_table.column(col, width=widths[col], anchor="w")
        y_scroll = Scrollbar(table_frame, orient="vertical", command=self.analysis_table.yview)
        x_scroll = Scrollbar(table_frame, orient="horizontal", command=self.analysis_table.xview)
        self.analysis_table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.analysis_table.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    def on_sidebar_configure(self, _event=None) -> None:
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def on_sidebar_canvas_configure(self, event) -> None:
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)

    def on_image_list_mousewheel(self, event) -> str:
        self.image_list.yview_scroll(-1 * int(event.delta / 120), "units")
        return "break"

    def on_label_summary_mousewheel(self, event) -> str:
        self.label_summary.yview_scroll(-1 * int(event.delta / 120), "units")
        return "break"

    def on_global_mousewheel(self, event) -> None:
        if not hasattr(self, "sidebar_canvas"):
            return
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget in {getattr(self, "image_list", None), getattr(self, "label_summary", None)}:
            return
        x = self.sidebar_canvas.winfo_pointerx()
        y = self.sidebar_canvas.winfo_pointery()
        left = self.sidebar_canvas.winfo_rootx()
        top = self.sidebar_canvas.winfo_rooty()
        right = left + self.sidebar_canvas.winfo_width()
        bottom = top + self.sidebar_canvas.winfo_height()
        if left <= x <= right and top <= y <= bottom:
            self.sidebar_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def load_labels(self) -> dict[str, dict[str, str]]:
        if not self.labels_path.exists():
            return {}
        try:
            with self.labels_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("labels", {})
        except Exception:
            return {}

    def load_training_labels(self) -> dict[str, dict[str, str]]:
        if not self.labels_path.exists():
            return {}
        try:
            with self.labels_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("training_labels", {})
        except Exception:
            return {}

    def load_scale_calibrations(self) -> dict[str, dict[str, float]]:
        if not self.labels_path.exists():
            return {}
        try:
            with self.labels_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("scale_calibrations", {})
        except Exception:
            return {}

    def load_states(self) -> list[str]:
        if self.labels_path.exists():
            try:
                with self.labels_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                states = data.get("states")
                if states:
                    return list(states)
            except Exception:
                pass
        return ["A", "B", "C"]

    def load_model(self) -> Pipeline | None:
        if self.model_path.exists():
            try:
                return joblib.load(self.model_path)
            except Exception:
                return None
        return None

    def choose_root(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.workspace)
        if folder:
            self.open_root(Path(folder))

    def open_root(self, root: Path) -> None:
        self.workspace = root
        self.output_dir = self.workspace / OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        self.labels_path = self.output_dir / LABELS_FILE
        self.model_path = self.output_dir / MODEL_FILE
        self.labels = {}
        self.training_labels = {}
        self.scale_calibrations = {}
        self.states = self.load_states()
        self.state_combo.configure(values=self.states)
        if self.current_state.get() not in self.states and self.states:
            self.current_state.set(self.states[0])
        self.model = None
        self.images = discover_images(root)
        self.image_list.delete(0, END)
        for path in self.images:
            self.image_list.insert(END, str(path.relative_to(root)))
        self.refresh_label_summary()
        self.status.set(f"Found {len(self.images)} origami PNG images.")
        if self.images:
            self.image_list.selection_set(0)
            self.load_image(self.images[0])

    def label_count_for_image(self, path: Path) -> int:
        return sum(1 for label in self.training_labels.get(image_key(path), {}).values() if label)

    def labeled_image_paths(self) -> list[Path]:
        return [path for path in self.images if self.label_count_for_image(path) > 0]

    def refresh_label_summary(self) -> None:
        self.label_summary.delete(0, END)
        for path in self.labeled_image_paths():
            count = self.label_count_for_image(path)
            self.label_summary.insert(END, f"{count:4d} labels  {path.relative_to(self.workspace)}")

    def selected_labeled_image_path(self) -> Path | None:
        selection = self.label_summary.curselection()
        labeled = self.labeled_image_paths()
        if not selection or selection[0] >= len(labeled):
            return None
        return labeled[selection[0]]

    def load_selected_labeled_image(self, _event=None) -> None:
        path = self.selected_labeled_image_path()
        if path is None:
            return
        try:
            index = self.images.index(path)
        except ValueError:
            return
        self.image_list.selection_clear(0, END)
        self.image_list.selection_set(index)
        self.image_list.see(index)
        self.load_image(path)

    def clear_selected_labeled_image(self) -> None:
        path = self.selected_labeled_image_path()
        if path is None:
            messagebox.showinfo("No labeled image selected", "Select an image in the Labeled Images list first.")
            return
        self.clear_labels_for_path(path, clear_working=False, clear_training=True)

    def load_image(self, path: Path) -> None:
        self.current_image = path
        self.rgb = load_rgb(path)
        self.scan_bounds = scan_bbox(self.rgb)
        self.objects = []
        self.status.set(f"Loaded {path.name}. Click Detect.")
        self.counts_text.set("")
        saved_scale = self.scale_calibrations.get(image_key(path), {})
        spm_scale = scale_info_from_spm(self.rgb, path)
        if spm_scale is not None:
            self.scale_bar_um.set(format_um(spm_scale.bar_um))
        elif "bar_um" in saved_scale:
            self.scale_bar_um.set(format_um(float(saved_scale["bar_um"])))
        self.update_scale_status()
        self.fit_image()

    def current_bar_um(self) -> float:
        try:
            value = float(self.scale_bar_um.get())
        except ValueError:
            value = 1.0
        return max(value, 1e-9)

    def current_scale_info(self, rgb: np.ndarray | None = None, path: Path | None = None) -> ScaleInfo:
        image = rgb if rgb is not None else self.rgb
        scale_path = path or (self.current_image if rgb is None else None)
        if scale_path is not None:
            if image is not None:
                spm_scale = scale_info_from_spm(image, scale_path)
                if spm_scale is not None:
                    return spm_scale
            saved = self.scale_calibrations.get(image_key(scale_path))
            if saved and "pixels_per_um" in saved:
                bar_um = float(saved.get("bar_um", self.current_bar_um()))
                pixels_per_um = float(saved["pixels_per_um"])
                return ScaleInfo(pixels_per_um=pixels_per_um, bar_pixels=pixels_per_um * bar_um, bar_um=bar_um, detected=True, source="saved")
        if image is None:
            return ScaleInfo(pixels_per_um=1.0, bar_pixels=1.0, bar_um=self.current_bar_um(), detected=False, source="fallback")
        return detect_scale_bar(image, self.current_bar_um())

    def update_scale_status(self) -> None:
        if self.rgb is None:
            self.scale_status.set("")
            return
        scale_info = self.current_scale_info()
        note = scale_info.source if scale_info.source != "bar" else "auto bar"
        if scale_info.source == "spm":
            self.scale_status.set(f"spm metadata: {scale_info.pixels_per_um:.1f} px/um; displayed bar {format_um(scale_info.bar_um)} um")
        else:
            self.scale_status.set(f"{note}: {scale_info.pixels_per_um:.1f} px/um ({scale_info.bar_pixels:.0f}px bar)")

    def pixel_area_bounds(self, rgb: np.ndarray, path: Path | None = None) -> tuple[int, int, ScaleInfo]:
        scale_info = self.current_scale_info(rgb, path)
        if self.min_area_um2 is not None and self.max_area_um2 is not None:
            factor = scale_info.pixels_per_um * scale_info.pixels_per_um
            min_area = max(4, int(round(self.min_area_um2 * factor)))
            max_area = max(min_area + 10, int(round(self.max_area_um2 * factor)))
            return min_area, max_area, scale_info
        return int(float(self.min_area.get())), int(float(self.max_area.get())), scale_info

    def on_image_select(self, _event=None) -> None:
        selection = self.image_list.curselection()
        if selection:
            self.load_image(self.images[selection[0]])

    def fit_image(self) -> None:
        if self.rgb is None:
            return
        self.fit_to_window = True
        ch = max(self.canvas.winfo_height(), 100)
        cw = max(self.canvas.winfo_width(), 100)
        h, w = self.rgb.shape[:2]
        self.scale = min(cw / w, ch / h)
        self.redraw()

    def zoom(self, factor: float) -> None:
        self.fit_to_window = False
        self.scale = max(0.1, min(8.0, self.scale * factor))
        self.redraw()

    def on_canvas_configure(self, _event=None) -> None:
        if self.fit_to_window:
            self.fit_image()
        else:
            self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.rgb is None:
            return
        pil = Image.fromarray(self.rgb)
        size = (max(1, int(pil.width * self.scale)), max(1, int(pil.height * self.scale)))
        display = pil.resize(size, Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display)
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        offset_x = max(0, (cw - size[0]) // 2)
        offset_y = max(0, (ch - size[1]) // 2)
        self.image_offset = (offset_x, offset_y)
        self.canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, max(cw, size[0] + offset_x), max(ch, size[1] + offset_y)))
        self.draw_overlays()

    def draw_overlays(self) -> None:
        if not self.objects:
            return
        for obj in self.objects:
            minr, minc, maxr, maxc = obj.bbox
            offset_x, offset_y = self.image_offset
            x1 = minc * self.scale + offset_x
            y1 = minr * self.scale + offset_y
            x2 = maxc * self.scale + offset_x
            y2 = maxr * self.scale + offset_y
            label = obj.label or obj.prediction
            color = state_color(label, self.states) if label else UNLABELED_COLOR
            dash = () if obj.label or not obj.prediction else (5, 3)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, dash=dash)
            if label:
                conf = f" {obj.confidence:.2f}" if obj.confidence is not None and not obj.label else ""
                self.canvas.create_text(x1 + 3, y1 + 3, text=f"{label}{conf}", anchor="nw", fill=color, font=("Segoe UI", 10, "bold"))

    def start_area_box(self) -> None:
        if self.rgb is None:
            return
        self.area_box_mode = True
        self.scale_box_mode = False
        self.drag_start = None
        self.status.set("Drag a box around one representative origami, then release.")

    def start_scale_box(self) -> None:
        if self.rgb is None:
            return
        if self.current_image is not None and scale_info_from_spm(self.rgb, self.current_image) is not None:
            self.status.set("This image has .spm scan-size metadata, so PNG scale-bar calibration is not needed.")
            return
        self.scale_box_mode = True
        self.area_box_mode = False
        self.drag_start = None
        self.status.set("Fallback only: drag a tight box across the PNG scale bar, then release.")

    def canvas_to_image(self, canvas_x: int, canvas_y: int) -> tuple[float, float] | None:
        offset_x, offset_y = self.image_offset
        world_x = self.canvas.canvasx(canvas_x)
        world_y = self.canvas.canvasy(canvas_y)
        x = (world_x - offset_x) / self.scale
        y = (world_y - offset_y) / self.scale
        if self.rgb is None or x < 0 or y < 0 or x >= self.rgb.shape[1] or y >= self.rgb.shape[0]:
            return None
        return x, y

    def detect_current(self, silent: bool = False) -> None:
        if self.rgb is None or self.current_image is None:
            return
        try:
            min_area, max_area, scale_info = self.pixel_area_bounds(self.rgb, self.current_image)
            if self.min_area_um2 is None or self.max_area_um2 is None:
                factor = scale_info.pixels_per_um * scale_info.pixels_per_um
                self.min_area_um2 = min_area / factor
                self.max_area_um2 = max_area / factor
            bias = float(self.threshold_bias.get())
            self.min_area.set(str(min_area))
            self.max_area.set(str(max_area))
            self.update_scale_status()
            self.objects = detect_origami(self.rgb, min_area, max_area, bias, scale_info.pixels_per_um)
            stored = self.labels.get(image_key(self.current_image), {})
            for obj in self.objects:
                obj.label = stored.get(str(obj.object_id))
            self.status.set(f"Detected {len(self.objects)} candidate origami in {self.current_image.name} using {scale_info.pixels_per_um:.1f} px/um.")
            self.update_counts_text()
            self.redraw()
        except Exception as exc:
            if not silent:
                messagebox.showerror("Detection failed", str(exc))

    def on_canvas_press(self, event) -> None:
        self.drag_start = (event.x, event.y)
        if self.drag_preview_id is not None:
            self.canvas.delete(self.drag_preview_id)
            self.drag_preview_id = None

    def on_canvas_drag(self, event) -> None:
        if not (self.area_box_mode or self.scale_box_mode) or self.drag_start is None:
            return
        if self.drag_preview_id is not None:
            self.canvas.delete(self.drag_preview_id)
        x0, y0 = self.drag_start
        world_x0 = self.canvas.canvasx(x0)
        world_y0 = self.canvas.canvasy(y0)
        world_x1 = self.canvas.canvasx(event.x)
        world_y1 = self.canvas.canvasy(event.y)
        self.drag_preview_id = self.canvas.create_rectangle(
            world_x0,
            world_y0,
            world_x1,
            world_y1,
            outline="#ff5a5f",
            width=2,
            dash=(4, 3),
        )

    def start_pan(self, event) -> None:
        self.pan_start = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)

    def pan_canvas(self, event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_mousewheel(self, event) -> None:
        self.canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def on_shift_mousewheel(self, event) -> None:
        self.canvas.xview_scroll(-1 * int(event.delta / 120), "units")

    def on_canvas_release(self, event) -> None:
        if self.drag_start is None:
            return
        start = self.drag_start
        self.drag_start = None
        if self.area_box_mode:
            self.finish_area_box(start, (event.x, event.y))
            return
        if self.scale_box_mode:
            self.finish_scale_box(start, (event.x, event.y))
            return
        if abs(event.x - start[0]) <= 3 and abs(event.y - start[1]) <= 3:
            self.label_at_canvas_point(event.x, event.y)

    def finish_scale_box(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        self.scale_box_mode = False
        if self.drag_preview_id is not None:
            self.canvas.delete(self.drag_preview_id)
            self.drag_preview_id = None
        p1 = self.canvas_to_image(*start)
        p2 = self.canvas_to_image(*end)
        if p1 is None or p2 is None or self.current_image is None:
            self.status.set("Scale box was outside the image. Try again.")
            return
        x1, _y1 = p1
        x2, _y2 = p2
        bar_pixels = abs(x2 - x1)
        if bar_pixels < 10:
            self.status.set("Scale box was too small. Drag across the full scale bar.")
            return
        bar_um = self.current_bar_um()
        pixels_per_um = bar_pixels / bar_um
        key = image_key(self.current_image)
        self.scale_calibrations[key] = {"bar_um": bar_um, "pixels_per_um": pixels_per_um}
        self.save_labels(silent=True)
        self.update_scale_status()
        self.status.set(f"Saved scale for this image: {pixels_per_um:.1f} px/um from a {bar_pixels:.0f}px bar.")
        self.detect_current(silent=True)

    def finish_area_box(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        self.area_box_mode = False
        if self.drag_preview_id is not None:
            self.canvas.delete(self.drag_preview_id)
            self.drag_preview_id = None
        p1 = self.canvas_to_image(*start)
        p2 = self.canvas_to_image(*end)
        if p1 is None or p2 is None or self.rgb is None:
            self.status.set("Area box was outside the image. Try again.")
            return
        x1, y1 = p1
        x2, y2 = p2
        minc, maxc = sorted((int(round(x1)), int(round(x2))))
        minr, maxr = sorted((int(round(y1)), int(round(y2))))
        if maxc - minc < 4 or maxr - minr < 4:
            self.status.set("Area box was too small. Drag around one origami.")
            return
        if self.scan_bounds is not None:
            scan_minr, scan_minc, scan_maxr, scan_maxc = self.scan_bounds
            if minr < scan_minr or maxr > scan_maxr or minc < scan_minc or maxc > scan_maxc:
                self.status.set("Draw the area box inside the AFM scan region, not over labels or scale bars.")
                return

        crop = self.rgb[minr:maxr, minc:maxc]
        estimated_area = self.estimate_area_from_box(crop)
        scale_info = self.current_scale_info()
        estimated_area_um2 = estimated_area / (scale_info.pixels_per_um * scale_info.pixels_per_um)
        self.min_area_um2 = max(1e-9, estimated_area_um2 * 0.35)
        self.max_area_um2 = max(self.min_area_um2 + 1e-9, estimated_area_um2 * 3.0)
        min_area = max(4, int(round(self.min_area_um2 * scale_info.pixels_per_um * scale_info.pixels_per_um)))
        max_area = max(min_area + 10, int(round(self.max_area_um2 * scale_info.pixels_per_um * scale_info.pixels_per_um)))
        self.min_area.set(str(min_area))
        self.max_area.set(str(max_area))
        self.status.set(f"Area calibrated: {estimated_area_um2:.4f} um^2 ({estimated_area:.0f} px here). Detecting...")
        self.detect_current(silent=True)

    def estimate_area_from_box(self, crop: np.ndarray) -> float:
        gray = rgb_to_grayscale(crop)
        smooth = filters.gaussian(gray, sigma=1.0)
        try:
            threshold = filters.threshold_otsu(smooth)
        except ValueError:
            return float(crop.shape[0] * crop.shape[1])
        high = smooth > threshold
        low = smooth < threshold
        binary = high if high.sum() <= low.sum() else low
        binary = morphology.remove_small_objects(binary, max_size=4)
        labels = measure.label(binary)
        areas = [float(prop.area) for prop in measure.regionprops(labels)]
        if areas:
            return max(areas)
        return float(crop.shape[0] * crop.shape[1] * 0.3)

    def label_at_canvas_point(self, canvas_x: int, canvas_y: int) -> None:
        if not self.objects or self.current_image is None:
            return
        point = self.canvas_to_image(canvas_x, canvas_y)
        if point is None:
            return
        x, y = point
        hit = None
        for obj in self.objects:
            minr, minc, maxr, maxc = obj.bbox
            if minc <= x <= maxc and minr <= y <= maxr:
                hit = obj
                break
        if hit is None:
            return
        hit.label = self.current_state.get()
        key = image_key(self.current_image)
        self.labels.setdefault(key, {})[str(hit.object_id)] = hit.label
        self.save_labels(silent=True)
        self.refresh_label_summary()
        self.update_counts_text()
        self.redraw()

    def add_state(self) -> None:
        value = self.new_state.get().strip()
        if not value:
            return
        if value not in self.states:
            self.states.append(value)
            self.state_combo.configure(values=self.states)
        self.current_state.set(value)
        self.new_state.set("")
        self.save_labels(silent=True)

    def clear_image_labels(self) -> None:
        if self.current_image is None:
            return
        self.clear_labels_for_path(self.current_image, clear_working=True, clear_training=True)

    def clear_labels_for_path(self, path: Path, clear_working: bool = True, clear_training: bool = True) -> None:
        key = image_key(path)
        if clear_working:
            self.labels.pop(key, None)
        if clear_training:
            self.training_labels.pop(key, None)
        if clear_working and self.current_image is not None and key == image_key(self.current_image):
            for obj in self.objects:
                obj.label = None
                obj.prediction = None
                obj.confidence = None
            self.update_counts_text()
            self.redraw()
        self.save_labels(silent=True)
        self.refresh_label_summary()
        self.status.set(f"Cleared {'training data' if clear_training and not clear_working else 'labels'} for {path.name}.")

    def add_current_labels_to_training(self) -> None:
        if self.current_image is None:
            return
        key = image_key(self.current_image)
        current_labels = {obj_id: label for obj_id, label in self.labels.get(key, {}).items() if label}
        if not current_labels:
            messagebox.showinfo("No labels to add", "Label or classify the current image before adding it to training data.")
            return
        self.training_labels[key] = dict(current_labels)
        self.save_labels(silent=True)
        self.refresh_label_summary()
        self.status.set(f"Added {len(current_labels)} labels from {self.current_image.name} to training data.")
        messagebox.showinfo("Training data updated", f"Added {len(current_labels)} labels from the current image to training data.")

    def save_labels(self, silent: bool = False) -> None:
        payload = {
            "states": self.states,
            "labels": self.labels,
            "training_labels": self.training_labels,
            "scale_calibrations": self.scale_calibrations,
        }
        with self.labels_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if not silent:
            self.status.set(f"Saved labels to {self.labels_path}.")

    def labeled_training_rows(self, paths: list[Path] | None = None) -> tuple[list[list[float]], list[str]]:
        features: list[list[float]] = []
        targets: list[str] = []
        for path in paths or self.images:
            stored = self.labels.get(image_key(path), {})
            training_stored = self.training_labels.get(image_key(path), {})
            if not training_stored:
                continue
            rgb = load_rgb(path)
            min_area, max_area, scale_info = self.pixel_area_bounds(rgb, path)
            objects = detect_origami(rgb, min_area, max_area, float(self.threshold_bias.get()), scale_info.pixels_per_um)
            for obj in objects:
                label = training_stored.get(str(obj.object_id))
                if label:
                    features.append(obj.features)
                    targets.append(label)
        return features, targets

    def train_current_image_model(self) -> None:
        if self.current_image is None:
            return
        self.train_model(paths=[self.current_image], scope_name="current image")

    def train_model(self, paths: list[Path] | None = None, scope_name: str = "all labeled images") -> None:
        selected_paths = paths or self.labeled_image_paths()
        if not selected_paths:
            messagebox.showwarning("No labels", "No labeled images are available for training.")
            return
        self.training_status.set("Training...")
        self.status.set(f"Training classifier from {scope_name}...")
        self.training_progress.start(12)
        self.root.update_idletasks()
        try:
            features, targets = self.labeled_training_rows(selected_paths)
            classes = sorted(set(targets))
            if len(classes) < 2:
                self.training_status.set("Need labels for at least two states.")
                messagebox.showwarning("Need more labels", f"Label examples for at least two states in {scope_name} before training.")
                return
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("forest", RandomForestClassifier(n_estimators=300, random_state=7, class_weight="balanced")),
                ]
            )
            model.fit(np.asarray(features, dtype=np.float32), np.asarray(targets))
            self.model = model
            self.save_classifier_bundle(self.model_path)
            counts = {cls: targets.count(cls) for cls in classes}
            image_counts = ", ".join(f"{path.name}: {self.label_count_for_image(path)}" for path in selected_paths)
            message = f"Training complete from {scope_name}. Trained on {len(targets)} labels across {len(selected_paths)} image(s): {counts}"
            self.training_status.set(message)
            self.status.set(f"{message}. Images: {image_counts}")
            messagebox.showinfo("Training complete", f"{message}\n\nImages used:\n{image_counts}")
        except Exception as exc:
            self.training_status.set("Training failed.")
            messagebox.showerror("Training failed", f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self.training_progress.stop()

    def classifier_bundle(self) -> dict:
        return {
            "format_version": MODEL_FORMAT_VERSION,
            "model": self.model,
            "states": self.states,
            "scale_aware": True,
            "feature_count": 57,
        }

    def save_classifier_bundle(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("No classifier has been trained.")
        joblib.dump(self.classifier_bundle(), path)

    def model_from_loaded_object(self, loaded):
        if isinstance(loaded, dict) and "model" in loaded:
            states = loaded.get("states")
            if states:
                for state in states:
                    if state not in self.states:
                        self.states.append(state)
                self.state_combo.configure(values=self.states)
            return loaded["model"]
        return loaded

    def export_classifier(self) -> None:
        if self.model is None:
            messagebox.showwarning("No classifier", "Train a classifier before exporting.")
            return
        default_name = "origami_state_classifier.joblib"
        path = filedialog.asksaveasfilename(
            initialdir=self.output_dir,
            initialfile=default_name,
            defaultextension=".joblib",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.save_classifier_bundle(Path(path))
            self.status.set(f"Exported classifier to {path}.")
            messagebox.showinfo("Classifier exported", f"Saved classifier to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def load_classifier_from_file(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=self.output_dir,
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            loaded = joblib.load(path)
            self.model = self.model_from_loaded_object(loaded)
            self.training_status.set(f"Loaded classifier: {Path(path).name}")
            self.status.set(f"Loaded classifier from {path}.")
            messagebox.showinfo("Classifier loaded", f"Loaded classifier from:\n{path}")
        except Exception as exc:
            messagebox.showerror("Load failed", f"{exc}\n\n{traceback.format_exc()}")

    def classify_current(self) -> None:
        if self.apply_classifier_to_current(show_messages=True):
            self.redraw()

    def apply_classifier_to_current(self, show_messages: bool = True) -> bool:
        if self.model is None:
            if show_messages:
                messagebox.showwarning("No classifier", "Train a classifier first.")
            return False
        if self.current_image is None:
            return False
        if not self.objects:
            self.detect_current(silent=True)
        if not self.objects:
            return False
        features = np.asarray([obj.features for obj in self.objects], dtype=np.float32)
        try:
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
        except ValueError as exc:
            if show_messages:
                messagebox.showwarning("Retrain classifier", f"The saved classifier is not compatible with the current scale-aware features. Please train again.\n\n{exc}")
            return False
        key = image_key(self.current_image)
        stored = self.labels.setdefault(key, {})
        accepted = 0
        kept_manual = 0
        for obj, pred, probs in zip(self.objects, predictions, probabilities):
            obj.prediction = str(pred)
            obj.confidence = float(np.max(probs))
            if obj.label:
                stored[str(obj.object_id)] = obj.label
                kept_manual += 1
            else:
                obj.label = obj.prediction
                stored[str(obj.object_id)] = obj.label
                accepted += 1
        self.save_labels(silent=True)
        self.refresh_label_summary()
        self.status.set(f"Classified {len(self.objects)} origami. Added {accepted} bootstrap labels; kept {kept_manual} existing manual labels.")
        self.update_counts_text()
        return True

    def current_counts_row(self) -> dict[str, str | int]:
        if self.current_image is None:
            raise ValueError("No image is loaded.")
        counts = {state: 0 for state in self.states}
        for obj in self.objects:
            label = obj.label or obj.prediction
            if label:
                counts[label] = counts.get(label, 0) + 1
        scale_info = self.current_scale_info(self.rgb, self.current_image)
        row: dict[str, str | int] = {
            "date_folder": self.current_image.parent.name,
            "image": self.current_image.name,
            "path": str(self.current_image),
            "pixels_per_um": f"{scale_info.pixels_per_um:.6g}",
            "total_detected": len(self.objects),
        }
        for state in self.states:
            row[state] = counts.get(state, 0)
        return row

    def annotated_image(self, rgb: np.ndarray | None = None, objects: list[OrigamiObject] | None = None) -> Image.Image:
        image_rgb = rgb if rgb is not None else self.rgb
        image_objects = objects if objects is not None else self.objects
        if image_rgb is None:
            raise ValueError("No image is loaded.")
        image = Image.fromarray(image_rgb).convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in image_objects:
            minr, minc, maxr, maxc = obj.bbox
            label = obj.label or obj.prediction
            color = state_color(label, self.states) if label else UNLABELED_COLOR
            draw.rectangle((minc, minr, maxc, maxr), outline=color, width=3)
            if label:
                text = str(label)
                if obj.confidence is not None and obj.prediction and not obj.label:
                    text = f"{label} {obj.confidence:.2f}"
                draw.text((minc + 3, minr + 3), text, fill=color)
        return image

    def export_current_results(self) -> None:
        if self.current_image is None:
            return
        if not self.apply_classifier_to_current(show_messages=True):
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.current_image.stem)
        output_dir = self.output_dir / CURRENT_RESULTS_DIR / f"{stem}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        row = self.current_counts_row()
        csv_path = output_dir / f"{stem}_counts.csv"
        fieldnames = ["date_folder", "image", "path", "pixels_per_um", "total_detected", *self.states]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

        original_copy = output_dir / self.current_image.name
        Image.fromarray(self.rgb).save(original_copy)
        annotated_copy = output_dir / f"{stem}_annotated.png"
        self.annotated_image().save(annotated_copy)
        self.status.set(f"Exported current results to {output_dir}.")
        messagebox.showinfo("Current results exported", f"Saved counts CSV and image copies to:\n{output_dir}")

    def classify_path(self, path: Path) -> tuple[np.ndarray, list[OrigamiObject], dict[str, str | int]]:
        if self.model is None:
            raise ValueError("Train or load a classifier first.")
        rgb = load_rgb(path)
        min_area, max_area, scale_info = self.pixel_area_bounds(rgb, path)
        objects = detect_origami(rgb, min_area, max_area, float(self.threshold_bias.get()), scale_info.pixels_per_um)
        counts = {state: 0 for state in self.states}
        if objects:
            features = np.asarray([obj.features for obj in objects], dtype=np.float32)
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            for obj, pred, probs in zip(objects, predictions, probabilities):
                obj.prediction = str(pred)
                obj.label = obj.prediction
                obj.confidence = float(np.max(probs))
                counts[obj.label] = counts.get(obj.label, 0) + 1
        row: dict[str, str | int] = {
            "date_folder": path.parent.name,
            "image": path.name,
            "path": str(path),
            "pixels_per_um": f"{scale_info.pixels_per_um:.6g}",
            "total_detected": len(objects),
        }
        for state in self.states:
            row[state] = counts.get(state, 0)
        return rgb, objects, row

    def write_image_result_folder(self, path: Path, rgb: np.ndarray, objects: list[OrigamiObject], row: dict[str, str | int], parent_dir: Path) -> Path:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
        output_dir = parent_dir / stem
        suffix = 2
        while output_dir.exists():
            output_dir = parent_dir / f"{stem}_{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True, exist_ok=True)

        fieldnames = ["date_folder", "image", "path", "pixels_per_um", "total_detected", *self.states]
        with (output_dir / f"{stem}_counts.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        Image.fromarray(rgb).save(output_dir / path.name)
        self.annotated_image(rgb, objects).save(output_dir / f"{stem}_annotated.png")
        return output_dir

    def classify_export_all_images(self) -> None:
        if self.model is None:
            messagebox.showwarning("No classifier", "Train or load a classifier first.")
            return
        if not self.images:
            messagebox.showwarning("No images", "Open a root folder with origami PNG images first.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_dir / CURRENT_RESULTS_DIR / f"all_images_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        try:
            for idx, path in enumerate(self.images, start=1):
                self.status.set(f"Classifying/exporting {idx}/{len(self.images)}: {path.name}")
                self.root.update_idletasks()
                rgb, objects, row = self.classify_path(path)
                self.write_image_result_folder(path, rgb, objects, row, output_dir)
                rows.append(row)
        except ValueError as exc:
            messagebox.showwarning("Classify/export failed", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Classify/export failed", f"{exc}\n\n{traceback.format_exc()}")
            return

        summary_path = output_dir / "all_image_counts.csv"
        fieldnames = ["date_folder", "image", "path", "pixels_per_um", "total_detected", *self.states]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.status.set(f"Classified and exported {len(rows)} images to {output_dir}.")
        messagebox.showinfo("Classify/export complete", f"Saved per-image folders and summary CSV to:\n{output_dir}")

    def update_counts_text(self, use_predictions: bool = False) -> None:
        counts: dict[str, int] = {}
        for obj in self.objects:
            label = obj.prediction if use_predictions and obj.prediction else obj.label
            if label:
                counts[label] = counts.get(label, 0) + 1
        if counts:
            lines = [f"{state}: {counts.get(state, 0)}" for state in self.states if state in counts or use_predictions]
            lines.append(f"Total labeled/counted: {sum(counts.values())}")
            self.counts_text.set("\n".join(lines))
        else:
            self.counts_text.set(f"Detected: {len(self.objects)}")

    def plot_counts_csv(self) -> None:
        csv_path = filedialog.askopenfilename(
            initialdir=self.output_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not csv_path:
            return
        try:
            output_dir = self.generate_count_plots(Path(csv_path))
            self.status.set(f"Generated plots in {output_dir}.")
            messagebox.showinfo("Plots generated", f"Saved plots and metrics to:\n{output_dir}")
        except Exception as exc:
            messagebox.showerror("Plotting failed", f"{exc}\n\n{traceback.format_exc()}")

    def generate_count_plots(self, csv_path: Path) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        if not rows:
            raise ValueError("The selected CSV has no rows.")

        meta_columns = {"date_folder", "image", "path", "pixels_per_um", "total_detected"}
        state_columns = []
        for field in fieldnames:
            if field in meta_columns or field.startswith("fraction_"):
                continue
            try:
                [float(row.get(field, 0) or 0) for row in rows]
            except ValueError:
                continue
            state_columns.append(field)
        if not state_columns:
            raise ValueError("No state count columns were found in the selected CSV.")

        labels = [row.get("image") or f"image_{idx + 1}" for idx, row in enumerate(rows)]
        short_labels = [label[:28] + ("..." if len(label) > 28 else "") for label in labels]
        counts = np.asarray([[float(row.get(state, 0) or 0) for state in state_columns] for row in rows], dtype=float)
        totals = counts.sum(axis=1)
        totals_for_fraction = np.where(totals > 0, totals, 1)
        fractions = counts / totals_for_fraction[:, None]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_dir / PLOTS_DIR / f"{csv_path.stem}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / "fraction_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            metric_fields = ["image", "path", "total_count", *[f"count_{s}" for s in state_columns], *[f"fraction_{s}" for s in state_columns]]
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writeheader()
            for idx, row in enumerate(rows):
                metric_row = {
                    "image": labels[idx],
                    "path": row.get("path", ""),
                    "total_count": f"{totals[idx]:.6g}",
                }
                for j, state in enumerate(state_columns):
                    metric_row[f"count_{state}"] = f"{counts[idx, j]:.6g}"
                    metric_row[f"fraction_{state}"] = f"{fractions[idx, j]:.6g}"
                writer.writerow(metric_row)

        x = np.arange(len(rows))
        fig_width = max(8, min(24, len(rows) * 0.55 + 4))

        plt.figure(figsize=(fig_width, 5))
        bottom = np.zeros(len(rows))
        for j, state in enumerate(state_columns):
            plt.bar(x, counts[:, j], bottom=bottom, label=state, color=STATE_COLORS[j % len(STATE_COLORS)])
            bottom += counts[:, j]
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylabel("Origami count")
        plt.title("Origami counts by image")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "counts_stacked.png", dpi=180)
        plt.close()

        plt.figure(figsize=(fig_width, 5))
        bottom = np.zeros(len(rows))
        for j, state in enumerate(state_columns):
            plt.bar(x, fractions[:, j], bottom=bottom, label=state, color=STATE_COLORS[j % len(STATE_COLORS)])
            bottom += fractions[:, j]
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Fraction of classified origami")
        plt.title("State fractions by image")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "fractions_stacked.png", dpi=180)
        plt.close()

        plt.figure(figsize=(fig_width, 4))
        plt.bar(x, totals, color="#555555")
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylabel("Total classified origami")
        plt.title("Total origami detected/classified")
        plt.tight_layout()
        plt.savefig(output_dir / "total_counts.png", dpi=180)
        plt.close()

        if len(state_columns) >= 2:
            a_idx, b_idx = 0, 1
            plt.figure(figsize=(6, 6))
            plt.scatter(fractions[:, a_idx], fractions[:, b_idx], s=np.maximum(totals, 10) * 3, alpha=0.75)
            for idx, label in enumerate(short_labels):
                plt.annotate(str(idx + 1), (fractions[idx, a_idx], fractions[idx, b_idx]), fontsize=8)
            plt.xlabel(f"Fraction {state_columns[a_idx]}")
            plt.ylabel(f"Fraction {state_columns[b_idx]}")
            plt.title(f"{state_columns[a_idx]} vs {state_columns[b_idx]} fraction")
            plt.xlim(-0.03, 1.03)
            plt.ylim(-0.03, 1.03)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(output_dir / f"fraction_{state_columns[a_idx]}_vs_{state_columns[b_idx]}.png", dpi=180)
            plt.close()

        return output_dir

    def plot_image_from_current_figure(self) -> Image.Image:
        import matplotlib.pyplot as plt

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150)
        plt.close()
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def build_count_plot_previews(self, csv_path: Path) -> dict[str, Image.Image]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        if not rows:
            raise ValueError("The selected CSV has no rows.")

        state_columns = self.count_state_columns(fieldnames, rows)
        if not state_columns:
            raise ValueError("No state count columns were found in the selected CSV.")

        labels = [row.get("image") or f"image_{idx + 1}" for idx, row in enumerate(rows)]
        short_labels = [label[:28] + ("..." if len(label) > 28 else "") for label in labels]
        counts = np.asarray([[float(row.get(state, 0) or 0) for state in state_columns] for row in rows], dtype=float)
        totals = counts.sum(axis=1)
        totals_for_fraction = np.where(totals > 0, totals, 1)
        fractions = counts / totals_for_fraction[:, None]
        x = np.arange(len(rows))
        fig_width = max(8, min(24, len(rows) * 0.55 + 4))

        previews: dict[str, Image.Image] = {}

        plt.figure(figsize=(fig_width, 5))
        bottom = np.zeros(len(rows))
        for j, state in enumerate(state_columns):
            plt.bar(x, counts[:, j], bottom=bottom, label=state, color=STATE_COLORS[j % len(STATE_COLORS)])
            bottom += counts[:, j]
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylabel("Origami count")
        plt.title("Origami counts by image")
        plt.legend()
        plt.tight_layout()
        previews["counts_stacked.png"] = self.plot_image_from_current_figure()

        plt.figure(figsize=(fig_width, 5))
        bottom = np.zeros(len(rows))
        for j, state in enumerate(state_columns):
            plt.bar(x, fractions[:, j], bottom=bottom, label=state, color=STATE_COLORS[j % len(STATE_COLORS)])
            bottom += fractions[:, j]
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Fraction of classified origami")
        plt.title("State fractions by image")
        plt.legend()
        plt.tight_layout()
        previews["fractions_stacked.png"] = self.plot_image_from_current_figure()

        plt.figure(figsize=(fig_width, 4))
        plt.bar(x, totals, color="#555555")
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylabel("Total classified origami")
        plt.title("Total origami detected/classified")
        plt.tight_layout()
        previews["total_counts.png"] = self.plot_image_from_current_figure()

        if len(state_columns) >= 2:
            a_idx, b_idx = 0, 1
            plt.figure(figsize=(6, 6))
            plt.scatter(fractions[:, a_idx], fractions[:, b_idx], s=np.maximum(totals, 10) * 3, alpha=0.75)
            for idx, _label in enumerate(short_labels):
                plt.annotate(str(idx + 1), (fractions[idx, a_idx], fractions[idx, b_idx]), fontsize=8)
            plt.xlabel(f"Fraction {state_columns[a_idx]}")
            plt.ylabel(f"Fraction {state_columns[b_idx]}")
            plt.title(f"{state_columns[a_idx]} vs {state_columns[b_idx]} fraction")
            plt.xlim(-0.03, 1.03)
            plt.ylim(-0.03, 1.03)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            previews[f"fraction_{state_columns[a_idx]}_vs_{state_columns[b_idx]}.png"] = self.plot_image_from_current_figure()

        return previews

    def parse_origami_label(self, image_name: str) -> str:
        match = re.search(r"origami([0-9]+f?)_", image_name, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        match = re.search(r"origami([0-9]+f?)", image_name, re.IGNORECASE)
        return match.group(1).lower() if match else "unknown"

    def count_state_columns(self, fieldnames: list[str], rows: list[dict[str, str]]) -> list[str]:
        meta_columns = {"date_folder", "image", "path", "pixels_per_um", "total_detected"}
        state_columns = []
        for field in fieldnames:
            if field in meta_columns or field.startswith("fraction_") or field.startswith("count_"):
                continue
            try:
                [float(row.get(field, 0) or 0) for row in rows]
            except ValueError:
                continue
            state_columns.append(field)
        return state_columns

    def read_counts_from_all_images_folder(self, folder: Path) -> tuple[list[dict[str, str]], list[str], Path]:
        summary = folder / "all_image_counts.csv"
        if summary.exists():
            with summary.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
            return rows, fieldnames, summary

        rows: list[dict[str, str]] = []
        fieldnames: list[str] = []
        for csv_path in sorted(folder.rglob("*_counts.csv")):
            if csv_path.name == "all_image_counts.csv":
                continue
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
                for field in reader.fieldnames or []:
                    if field not in fieldnames:
                        fieldnames.append(field)
        if not rows:
            raise ValueError("No counts CSV files were found in the selected folder.")

        combined = folder / "analysis_loaded_counts.csv"
        with combined.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return rows, fieldnames, combined

    def load_analysis_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.output_dir / CURRENT_RESULTS_DIR)
        if not folder:
            return
        try:
            rows, fieldnames, csv_path = self.read_counts_from_all_images_folder(Path(folder))
            self.populate_analysis(rows, fieldnames, csv_path)
        except Exception as exc:
            messagebox.showerror("Analysis load failed", f"{exc}\n\n{traceback.format_exc()}")

    def populate_analysis(self, rows: list[dict[str, str]], fieldnames: list[str], csv_path: Path) -> None:
        state_columns = self.count_state_columns(fieldnames, rows)
        if not state_columns:
            raise ValueError("No state count columns were found.")
        primary = state_columns[0]
        secondary = state_columns[1] if len(state_columns) > 1 else None
        self.analysis_csv_path = csv_path
        self.analysis_rows = []
        self.analysis_table.delete(*self.analysis_table.get_children())
        self.analysis_summary.delete(0, END)

        grouped: dict[str, dict[str, float]] = {}
        for row in rows:
            image_name = row.get("image", "")
            origami_label = self.parse_origami_label(image_name)
            counts = {state: float(row.get(state, 0) or 0) for state in state_columns}
            total = sum(counts.values())
            primary_fraction = counts[primary] / total if total else 0.0
            secondary_fraction = counts[secondary] / total if total and secondary else 0.0
            ratio = counts[primary] / counts[secondary] if secondary and counts[secondary] else float("inf") if counts[primary] else 0.0
            display_ratio = "inf" if math.isinf(ratio) else f"{ratio:.3f}"
            self.analysis_table.insert(
                "",
                END,
                values=("image", origami_label, image_name, f"{total:.0f}", f"{primary_fraction:.3f}", f"{secondary_fraction:.3f}", display_ratio),
            )
            metric_row: dict[str, str | float] = {
                "kind": "image",
                "origami_label": origami_label,
                "image": image_name,
                "path": row.get("path", ""),
                "total": total,
                f"fraction_{primary}": primary_fraction,
            }
            if secondary:
                metric_row[f"fraction_{secondary}"] = secondary_fraction
                metric_row[f"{primary}_to_{secondary}_ratio"] = ratio
            for state, count in counts.items():
                metric_row[f"count_{state}"] = count
            self.analysis_rows.append(metric_row)

            group = grouped.setdefault(origami_label, {"n_images": 0.0, "total": 0.0, **{state: 0.0 for state in state_columns}})
            group["n_images"] += 1
            group["total"] += total
            for state, count in counts.items():
                group[state] += count

        self.analysis_table.insert("", END, values=("", "", "", "", "", "", ""))
        for origami_label, group in sorted(grouped.items(), key=lambda item: item[0]):
            total = group["total"]
            primary_fraction = group[primary] / total if total else 0.0
            secondary_fraction = group[secondary] / total if total and secondary else 0.0
            ratio = group[primary] / group[secondary] if secondary and group[secondary] else float("inf") if group[primary] else 0.0
            display_ratio = "inf" if math.isinf(ratio) else f"{ratio:.3f}"
            self.analysis_table.insert(
                "",
                END,
                values=("group", origami_label, f"{int(group['n_images'])} image(s)", f"{total:.0f}", f"{primary_fraction:.3f}", f"{secondary_fraction:.3f}", display_ratio),
            )

        metrics_path = csv_path.parent / "analysis_metrics.csv"
        metric_fields = sorted({key for metric_row in self.analysis_rows for key in metric_row.keys()})
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writeheader()
            writer.writerows(self.analysis_rows)

        self.analysis_summary.insert(END, f"Loaded: {csv_path.parent}")
        self.analysis_summary.insert(END, f"Images: {len(rows)}")
        self.analysis_summary.insert(END, f"Origami labels: {', '.join(sorted(grouped.keys()))}")
        self.analysis_summary.insert(END, f"States: {', '.join(state_columns)}")
        self.analysis_summary.insert(END, f"Primary metric: fraction {primary}")
        if secondary:
            self.analysis_summary.insert(END, f"Comparison metric: fraction {primary} vs {secondary}, ratio {primary}/{secondary}")
        self.analysis_summary.insert(END, f"Saved metrics: {metrics_path.name}")
        self.analysis_status.set(f"Loaded {len(rows)} image rows from {csv_path.name}.")

    def generate_analysis_plots(self) -> None:
        if self.analysis_csv_path is None:
            messagebox.showinfo("No analysis folder loaded", "Load an all_images folder first.")
            return
        try:
            output_dir = self.generate_count_plots(self.analysis_csv_path)
            self.analysis_status.set(f"Generated plots in {output_dir}.")
            messagebox.showinfo("Plots generated", f"Saved plots to:\n{output_dir}")
        except Exception as exc:
            messagebox.showerror("Plotting failed", f"{exc}\n\n{traceback.format_exc()}")

    def generate_analysis_plot_previews(self) -> None:
        if self.analysis_csv_path is None:
            messagebox.showinfo("No analysis folder loaded", "Load an all_images folder first.")
            return
        try:
            self.plot_previews = self.build_count_plot_previews(self.analysis_csv_path)
            self.plot_list.delete(0, END)
            for name in self.plot_previews:
                self.plot_list.insert(END, name)
            if self.plot_previews:
                self.plot_list.selection_set(0)
                self.show_plot_preview(next(iter(self.plot_previews)))
            self.analysis_status.set(f"Generated {len(self.plot_previews)} plot preview(s). Select one to view or save.")
        except Exception as exc:
            messagebox.showerror("Plot preview failed", f"{exc}\n\n{traceback.format_exc()}")

    def selected_plot_preview_name(self) -> str | None:
        selection = self.plot_list.curselection()
        if not selection:
            return None
        return self.plot_list.get(selection[0])

    def on_plot_preview_select(self, _event=None) -> None:
        name = self.selected_plot_preview_name()
        if name:
            self.show_plot_preview(name)

    def show_plot_preview(self, name: str) -> None:
        image = self.plot_previews.get(name)
        if image is None:
            return
        preview = ImageOps.contain(image, (900, 360))
        self.plot_preview_photo = ImageTk.PhotoImage(preview)
        self.plot_preview_label.configure(image=self.plot_preview_photo, text="")

    def save_selected_plot_preview(self) -> None:
        name = self.selected_plot_preview_name()
        if not name or name not in self.plot_previews:
            messagebox.showinfo("No plot selected", "Generate plot previews and select one plot first.")
            return
        path = filedialog.asksaveasfilename(
            initialdir=self.output_dir / PLOTS_DIR,
            initialfile=name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        self.plot_previews[name].save(path)
        self.analysis_status.set(f"Saved plot: {path}")

    def save_all_plot_previews(self) -> None:
        if not self.plot_previews:
            messagebox.showinfo("No plot previews", "Generate plot previews first.")
            return
        folder = filedialog.askdirectory(initialdir=self.output_dir / PLOTS_DIR)
        if not folder:
            return
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        for name, image in self.plot_previews.items():
            image.save(folder_path / name)
        self.analysis_status.set(f"Saved {len(self.plot_previews)} plot(s) to {folder_path}.")
        messagebox.showinfo("Plots saved", f"Saved {len(self.plot_previews)} plot(s) to:\n{folder_path}")

    def batch_count(self) -> None:
        if self.model is None:
            messagebox.showwarning("No classifier", "Train a classifier first.")
            return
        output = self.output_dir / COUNTS_FILE
        rows = []
        for idx, path in enumerate(self.images, start=1):
            self.status.set(f"Counting {idx}/{len(self.images)}: {path.name}")
            self.root.update_idletasks()
            rgb = load_rgb(path)
            min_area, max_area, scale_info = self.pixel_area_bounds(rgb, path)
            objects = detect_origami(rgb, min_area, max_area, float(self.threshold_bias.get()), scale_info.pixels_per_um)
            row = {
                "date_folder": path.parent.name,
                "image": path.name,
                "path": str(path),
                "pixels_per_um": f"{scale_info.pixels_per_um:.6g}",
                "total_detected": len(objects),
            }
            for state in self.states:
                row[state] = 0
            if objects:
                features = np.asarray([obj.features for obj in objects], dtype=np.float32)
                try:
                    predictions = self.model.predict(features)
                except ValueError as exc:
                    messagebox.showwarning("Retrain classifier", f"The saved classifier is not compatible with the current scale-aware features. Please train again.\n\n{exc}")
                    return
                for pred in predictions:
                    pred = str(pred)
                    row[pred] = row.get(pred, 0) + 1
            rows.append(row)
        fieldnames = ["date_folder", "image", "path", "pixels_per_um", "total_detected", *self.states]
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.status.set(f"Wrote counts for {len(rows)} images to {output}.")
        messagebox.showinfo("Batch count complete", f"Wrote {output}")


def main() -> int:
    root = Tk()
    app = OrigamiCounterApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
