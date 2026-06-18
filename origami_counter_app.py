from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, Button, Canvas, Entry, Frame, Label, Listbox, Menu, Scrollbar, StringVar, Tk, Toplevel, filedialog, messagebox, simpledialog, ttk

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageTk
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


@dataclass
class SpmImageSection:
    offset: int
    data_length: int
    bytes_per_pixel: int
    width: int
    height: int
    label: str
    frame_direction: str = ""


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


def parse_spm_z_nm_per_lsb(spm_path: Path) -> float | None:
    try:
        text = spm_header_text(spm_path.read_bytes())
    except OSError:
        return None
    sens_match = re.search(r"\\@Sens\. Zsens:\s*V\s*([0-9.eE+-]+)\s*nm/V", text)
    if not sens_match:
        return None
    height_start = text.lower().find("height")
    search_text = text[height_start:] if height_start >= 0 else text
    scale_match = re.search(r"\\@\d+:Z scale:[^\r\n]*\(([0-9.eE+-]+)\s*V/LSB\)\s*([0-9.eE+-]+)\s*V", search_text)
    if not scale_match:
        return None
    try:
        sens_nm_per_v = float(sens_match.group(1))
        volts_per_lsb = float(scale_match.group(1))
        scale_volts = float(scale_match.group(2))
    except ValueError:
        return None
    value = sens_nm_per_v * volts_per_lsb * scale_volts
    return value if value > 0 else None


def spm_header_text(data: bytes) -> str:
    marker = b"\\*File list end"
    end = data.find(marker)
    if end >= 0:
        line_end = data.find(b"\n", end)
        end = line_end + 1 if line_end >= 0 else end + len(marker)
    else:
        end = min(len(data), 256_000)
    return data[:end].decode("latin-1", errors="ignore")


def spm_section_value(section: str, name: str) -> str | None:
    match = re.search(r"\\" + re.escape(name) + r":\s*([^\r\n]+)", section)
    return match.group(1).strip() if match else None


def parse_spm_image_sections(data: bytes) -> list[SpmImageSection]:
    text = spm_header_text(data)
    starts = [match.start() for match in re.finditer(r"\\\*Ciao image list", text)]
    sections: list[SpmImageSection] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        section = text[start:end]
        try:
            offset = int(spm_section_value(section, "Data offset") or "")
            data_length = int(spm_section_value(section, "Data length") or "")
            bytes_per_pixel = int(spm_section_value(section, "Bytes/pixel") or "")
            width = int(spm_section_value(section, "Samps/line") or "")
            height = int(spm_section_value(section, "Number of lines") or spm_section_value(section, "Lines") or "")
        except ValueError:
            continue
        label_match = re.search(r"\\@\d+:Image Data:[^\r\n]+", section)
        label = label_match.group(0) if label_match else ""
        if offset < 0 or data_length <= 0 or bytes_per_pixel not in {2, 4} or width <= 0 or height <= 0:
            continue
        if offset + data_length > len(data):
            continue
        sections.append(SpmImageSection(offset, data_length, bytes_per_pixel, width, height, label, spm_section_value(section, "Frame direction") or ""))
    return sections


def preferred_spm_height_section(sections: list[SpmImageSection]) -> SpmImageSection | None:
    if not sections:
        return None
    height_sections = [section for section in sections if "height" in section.label.lower()]
    non_error = [section for section in height_sections if "error" not in section.label.lower()]
    if non_error:
        return non_error[0]
    if height_sections:
        return height_sections[0]
    return sections[0]


def read_spm_height_array(spm_path: Path) -> np.ndarray:
    data = spm_path.read_bytes()
    section = preferred_spm_height_section(parse_spm_image_sections(data))
    if section is None:
        raise ValueError(f"No readable image channel found in {spm_path.name}.")
    dtype = "<i4" if section.bytes_per_pixel == 4 else "<i2"
    expected = section.width * section.height
    arr = np.frombuffer(data[section.offset : section.offset + section.data_length], dtype=dtype, count=expected)
    if arr.size != expected:
        raise ValueError(f"Image data in {spm_path.name} is shorter than expected.")
    image = arr.astype(np.float32).reshape(section.height, section.width)
    if section.frame_direction.lower() == "down":
        image = np.flipud(image)
    return image


def plane_flatten_height(height: np.ndarray) -> np.ndarray:
    arr = np.asarray(height, dtype=np.float32)
    y_idx, x_idx = np.indices(arr.shape, dtype=np.float32)
    step = max(1, int(max(arr.shape) / 256))
    z = arr[::step, ::step].ravel()
    x = x_idx[::step, ::step].ravel()
    y = y_idx[::step, ::step].ravel()
    valid = np.isfinite(z)
    if valid.sum() < 3:
        return arr - np.nanmedian(arr)
    design = np.column_stack([x[valid], y[valid], np.ones(valid.sum(), dtype=np.float32)])
    coeffs, *_ = np.linalg.lstsq(design, z[valid], rcond=None)
    plane = coeffs[0] * x_idx + coeffs[1] * y_idx + coeffs[2]
    return arr - plane.astype(np.float32)


def line_flatten_height(height: np.ndarray) -> np.ndarray:
    arr = np.asarray(height, dtype=np.float32).copy()
    x = np.arange(arr.shape[1], dtype=np.float32)
    for row_idx in range(arr.shape[0]):
        row = arr[row_idx]
        valid = np.isfinite(row)
        if valid.sum() >= 2:
            coeffs = np.polyfit(x[valid], row[valid], 1)
            row = row - (coeffs[0] * x + coeffs[1])
        row = row - np.nanmedian(row)
        arr[row_idx] = row
    return arr


def flatten_spm_height(height: np.ndarray) -> np.ndarray:
    return line_flatten_height(plane_flatten_height(height))


def height_to_uint8(height: np.ndarray) -> np.ndarray:
    arr = np.asarray(height, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def height_contrast_limits(height: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(height, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    # NanoScope auto display scaling permits tiny high/low outliers to saturate instead of
    # letting one speck set the whole color range and darken the image.
    lo, hi = np.percentile(finite, [0.01, 99.95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


NANOSCOPE_HEIGHT_COLORS = np.asarray(
    [
        (0, 0, 0),
        (0, 0, 0),
        (8, 0, 0),
        (26, 0, 0),
        (44, 0, 0),
        (63, 0, 0),
        (81, 0, 0),
        (99, 19, 0),
        (117, 47, 0),
        (135, 74, 0),
        (153, 102, 0),
        (171, 130, 30),
        (189, 158, 80),
        (207, 185, 129),
        (226, 213, 179),
        (244, 241, 228),
        (253, 255, 253),
    ],
    dtype=np.float32,
)


def apply_height_colormap(height: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((np.asarray(height, dtype=np.float32) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    positions = np.linspace(0.0, 1.0, len(NANOSCOPE_HEIGHT_COLORS), dtype=np.float32)
    flat = scaled.ravel()
    channels = [np.interp(flat, positions, NANOSCOPE_HEIGHT_COLORS[:, idx]) for idx in range(3)]
    return np.stack(channels, axis=1).reshape(*scaled.shape, 3).astype(np.uint8)


def draw_spm_scale_bar(draw: ImageDraw.ImageDraw, image_width: int, image_height: int, scan_size_um: float | None) -> None:
    if scan_size_um is None or scan_size_um <= 0:
        return
    bar_um = scan_size_um / 5.0
    bar_px = max(24, int(round(image_width * bar_um / scan_size_um)))
    x1 = image_width - bar_px - 24
    x2 = image_width - 24
    y = image_height + 25
    draw.line((x1, y, x2, y), fill=(0, 0, 0), width=5)
    label = f"{format_um(bar_um)} um"
    draw.text((x1, y + 8), label, fill=(0, 0, 0))


def render_spm_png(height: np.ndarray, scan_size_um: float | None, z_nm_per_lsb: float | None = None) -> Image.Image:
    flattened = flatten_spm_height(height)
    lo, hi = height_contrast_limits(flattened)
    scan_rgb = apply_height_colormap(flattened, lo, hi)
    scan = Image.fromarray(scan_rgb, mode="RGB")
    scan_w, scan_h = scan.size

    colorbar_w = 24
    gap = 24
    right_margin = 74
    bottom_margin = 62
    canvas_w = scan_w + gap + colorbar_w + right_margin
    canvas_h = scan_h + bottom_margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(scan, (0, 0))

    colorbar_values = np.linspace(hi, lo, scan_h, dtype=np.float32)[:, None]
    colorbar_rgb = apply_height_colormap(np.repeat(colorbar_values, colorbar_w, axis=1), lo, hi)
    colorbar = Image.fromarray(colorbar_rgb, mode="RGB")
    colorbar_x = scan_w + gap
    canvas.paste(colorbar, (colorbar_x, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((colorbar_x, 0, colorbar_x + colorbar_w - 1, scan_h - 1), outline=(0, 0, 0), width=1)
    label_x = colorbar_x + colorbar_w + 6
    if z_nm_per_lsb is not None and z_nm_per_lsb > 0:
        top_label = f"{hi * z_nm_per_lsb:.2g} nm"
        bottom_label = f"{lo * z_nm_per_lsb:.2g} nm"
    else:
        top_label = f"{hi:.2g}"
        bottom_label = f"{lo:.2g}"
    draw.text((label_x, 0), top_label, fill=(0, 0, 0))
    draw.text((label_x, scan_h - 12), bottom_label, fill=(0, 0, 0))
    draw.text((20, scan_h + 26), "Height Sensor", fill=(0, 0, 0))

    draw_spm_scale_bar(draw, scan_w, scan_h, scan_size_um)
    return canvas


def convert_spm_to_png(spm_path: Path, png_path: Path) -> None:
    height = read_spm_height_array(spm_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    render_spm_png(height, parse_spm_scan_size_um(spm_path), parse_spm_z_nm_per_lsb(spm_path)).save(png_path)


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
        self.analysis_x_state = StringVar(value="")
        self.analysis_y_state = StringVar(value="")
        self.analysis_delta_state = StringVar(value="")
        self.analysis_plot_group = StringVar(value="Images")
        self.analysis_x_axis_order = StringVar(value="Origami first")
        self.analysis_state_labels: dict[str, str] = {}
        self.analysis_rows: list[dict[str, str | float]] = []
        self.analysis_source_rows: list[dict[str, str]] = []
        self.analysis_fieldnames: list[str] = []
        self.analysis_table_row_map: dict[str, dict[str, str]] = {}
        self.analysis_next_row_id = 1
        self.analysis_scale_filter: set[str] | None = None
        self.analysis_origami_filter: set[str] | None = None
        self.analysis_dataset_filter: set[str] | None = None
        self.analysis_state_filter: set[str] | None = None
        self.analysis_scale_options: list[str] = []
        self.analysis_origami_options: list[str] = []
        self.analysis_dataset_options: list[str] = []
        self.analysis_state_options: list[str] = []
        self.analysis_filter_status = StringVar(value="Filters: all")
        self.analysis_csv_path: Path | None = None
        self.plot_previews: dict[str, Image.Image] = {}
        self.plot_preview_photo: ImageTk.PhotoImage | None = None
        self.analysis_image_photo: ImageTk.PhotoImage | None = None
        self.analysis_review_photo: ImageTk.PhotoImage | None = None
        self.analysis_review_rows: list[dict[str, str]] = []
        self.analysis_review_selected_id: str | None = None
        self.analysis_review_image: Image.Image | None = None
        self.analysis_review_zoom = 1.0
        self.analysis_review_fit_to_window = True
        self.plot_preview_zoom = 1.0
        self.current_plot_preview_name: str | None = None
        self.plot_preview_canvas_image_id: int | None = None
        self.min_area_um2: float | None = None
        self.max_area_um2: float | None = None

        self.build_ui()
        self.open_root(self.workspace)

    def build_ui(self) -> None:
        menubar = Menu(self.root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Root", command=self.choose_root)
        file_menu.add_command(label="Import SPM Folder", command=self.import_spm_folder)
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
        Button(sidebar, text="Import SPM Folder", command=self.import_spm_folder).pack(fill="x", pady=(4, 0))
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
        Button(toolbar, text="Load 2 Folders", command=self.load_analysis_comparison_folders).pack(side=LEFT, padx=(6, 0))
        Button(toolbar, text="Generate Plot Previews", command=self.generate_analysis_plot_previews).pack(side=LEFT, padx=6)
        Button(toolbar, text="Save Selected Plot", command=self.save_selected_plot_preview).pack(side=LEFT)
        Button(toolbar, text="Save All Plots", command=self.save_all_plot_previews).pack(side=LEFT, padx=6)
        Button(toolbar, text="Save Cleaned Dataset", command=self.save_cleaned_analysis_dataset).pack(side=LEFT)
        Label(toolbar, textvariable=self.analysis_status).pack(side=LEFT, padx=12)

        summary_box = ttk.LabelFrame(parent, text="Summary")
        summary_box.pack(fill="x", padx=8, pady=(0, 8))
        self.analysis_summary = Listbox(summary_box, height=6, exportselection=False)
        self.analysis_summary.pack(fill="x", padx=6, pady=6)

        analysis_notebook = ttk.Notebook(parent)
        analysis_notebook.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))
        plots_tab = Frame(analysis_notebook)
        review_tab = Frame(analysis_notebook)
        analysis_notebook.add(plots_tab, text="Plots")
        analysis_notebook.add(review_tab, text="Review Images")

        analysis_panes = ttk.Panedwindow(plots_tab, orient="vertical")
        analysis_panes.pack(fill=BOTH, expand=True)

        plot_box = ttk.LabelFrame(analysis_panes, text="Plot Preview")
        analysis_panes.add(plot_box, weight=2)
        plot_toolbar = Frame(plot_box)
        plot_toolbar.pack(fill="x", padx=6, pady=(6, 0))
        plot_toolbar_top = Frame(plot_toolbar)
        plot_toolbar_top.pack(fill="x")
        plot_toolbar_bottom = Frame(plot_toolbar)
        plot_toolbar_bottom.pack(fill="x", pady=(4, 0))
        Button(plot_toolbar_top, text="Zoom Out", command=lambda: self.adjust_plot_preview_zoom(0.8)).pack(side=LEFT)
        Button(plot_toolbar_top, text="Reset", command=self.reset_plot_preview_zoom).pack(side=LEFT, padx=6)
        Button(plot_toolbar_top, text="Zoom In", command=lambda: self.adjust_plot_preview_zoom(1.25)).pack(side=LEFT)
        self.plot_zoom_status = StringVar(value="100%")
        Label(plot_toolbar_top, textvariable=self.plot_zoom_status).pack(side=LEFT, padx=10)
        Label(plot_toolbar_top, text="Group").pack(side=LEFT, padx=(16, 4))
        self.analysis_group_combo = ttk.Combobox(
            plot_toolbar_top,
            textvariable=self.analysis_plot_group,
            values=["Images", "Dataset", "Origami", "Scale", "Origami + Scale", "Dataset + Origami", "Dataset + Scale", "Dataset + Origami + Scale"],
            width=14,
            state="readonly",
        )
        self.analysis_group_combo.pack(side=LEFT)
        self.analysis_group_combo.bind("<<ComboboxSelected>>", self.on_analysis_plot_group_select)
        Label(plot_toolbar_top, text="X Order").pack(side=LEFT, padx=(10, 4))
        self.analysis_x_order_combo = ttk.Combobox(
            plot_toolbar_top,
            textvariable=self.analysis_x_axis_order,
            values=["Origami first", "Dataset first"],
            width=12,
            state="readonly",
        )
        self.analysis_x_order_combo.pack(side=LEFT)
        self.analysis_x_order_combo.bind("<<ComboboxSelected>>", self.on_analysis_plot_order_select)
        Button(plot_toolbar_top, text="Filter Datasets", command=self.choose_analysis_dataset_filter).pack(side=LEFT, padx=(10, 0))
        Button(plot_toolbar_top, text="Filter Scales", command=self.choose_analysis_scale_filter).pack(side=LEFT, padx=(10, 0))
        Button(plot_toolbar_top, text="Filter Origami", command=self.choose_analysis_origami_filter).pack(side=LEFT, padx=(6, 0))
        Button(plot_toolbar_top, text="Filter States", command=self.choose_analysis_state_filter).pack(side=LEFT, padx=(6, 0))
        Button(plot_toolbar_top, text="Clear Filters", command=self.clear_analysis_plot_filters).pack(side=LEFT, padx=(6, 0))
        Label(plot_toolbar_bottom, text="Scatter X").pack(side=LEFT)
        self.analysis_x_combo = ttk.Combobox(plot_toolbar_bottom, textvariable=self.analysis_x_state, values=[], width=10, state="readonly")
        self.analysis_x_combo.pack(side=LEFT)
        self.analysis_x_combo.bind("<<ComboboxSelected>>", self.on_analysis_scatter_select)
        Label(plot_toolbar_bottom, text="Y").pack(side=LEFT, padx=(8, 4))
        self.analysis_y_combo = ttk.Combobox(plot_toolbar_bottom, textvariable=self.analysis_y_state, values=[], width=10, state="readonly")
        self.analysis_y_combo.pack(side=LEFT)
        self.analysis_y_combo.bind("<<ComboboxSelected>>", self.on_analysis_scatter_select)
        Label(plot_toolbar_bottom, text="Delta State").pack(side=LEFT, padx=(10, 4))
        self.analysis_delta_combo = ttk.Combobox(plot_toolbar_bottom, textvariable=self.analysis_delta_state, values=[], width=10, state="readonly")
        self.analysis_delta_combo.pack(side=LEFT)
        self.analysis_delta_combo.bind("<<ComboboxSelected>>", self.on_analysis_delta_state_select)
        Button(plot_toolbar_bottom, text="Rename States", command=self.rename_analysis_state_labels).pack(side=LEFT, padx=8)
        Button(plot_toolbar_bottom, text="Update Scatter", command=self.generate_analysis_plot_previews).pack(side=LEFT, padx=8)
        Button(plot_toolbar_bottom, text="Overlay Datasets", command=self.generate_dataset_overlay_plot_preview).pack(side=LEFT)
        Label(plot_toolbar_bottom, textvariable=self.analysis_filter_status).pack(side=LEFT, padx=(8, 0))

        plot_inner = Frame(plot_box)
        plot_inner.pack(fill=BOTH, expand=True, padx=6, pady=6)
        self.plot_list = Listbox(plot_inner, height=6, exportselection=False)
        self.plot_list.pack(side=LEFT, fill="y")
        self.plot_list.bind("<<ListboxSelect>>", self.on_plot_preview_select)
        preview_frame = Frame(plot_inner)
        preview_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(8, 0))
        self.plot_preview_canvas = Canvas(preview_frame, background="#f0f0f0", highlightthickness=0)
        plot_y_scroll = Scrollbar(preview_frame, orient="vertical", command=self.plot_preview_canvas.yview)
        plot_x_scroll = Scrollbar(preview_frame, orient="horizontal", command=self.plot_preview_canvas.xview)
        self.plot_preview_canvas.configure(xscrollcommand=plot_x_scroll.set, yscrollcommand=plot_y_scroll.set)
        self.plot_preview_canvas.grid(row=0, column=0, sticky="nsew")
        plot_y_scroll.grid(row=0, column=1, sticky="ns")
        plot_x_scroll.grid(row=1, column=0, sticky="ew")
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.plot_preview_canvas.create_text(16, 16, text="Generate plot previews to view them here.", anchor="nw")
        self.plot_preview_canvas.bind("<Configure>", lambda _event: self.render_plot_preview())

        table_frame = Frame(analysis_panes)
        analysis_panes.add(table_frame, weight=3)
        columns = ("kind", "dataset", "origami_label", "scale", "image", "total", "state_fractions")
        self.analysis_table = ttk.Treeview(table_frame, columns=columns, show="headings")
        headings = {
            "kind": "Row",
            "dataset": "Dataset",
            "origami_label": "Origami",
            "scale": "Scale",
            "image": "Image",
            "total": "Total",
            "state_fractions": "State Fractions",
        }
        widths = {
            "kind": 90,
            "dataset": 130,
            "origami_label": 90,
            "scale": 90,
            "image": 260,
            "total": 80,
            "state_fractions": 360,
        }
        for col in columns:
            self.analysis_table.heading(col, text=headings[col])
            self.analysis_table.column(col, width=widths[col], anchor="w")
        self.analysis_table.bind("<Double-1>", self.on_analysis_table_double_click)
        self.analysis_table.bind("<Delete>", self.delete_selected_analysis_rows)
        self.analysis_table.bind("<BackSpace>", self.delete_selected_analysis_rows)
        y_scroll = Scrollbar(table_frame, orient="vertical", command=self.analysis_table.yview)
        x_scroll = Scrollbar(table_frame, orient="horizontal", command=self.analysis_table.xview)
        self.analysis_table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.analysis_table.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.build_analysis_review_tab(review_tab)

    def build_analysis_review_tab(self, parent: Frame) -> None:
        review_toolbar = Frame(parent)
        review_toolbar.pack(fill="x", padx=6, pady=6)
        Button(review_toolbar, text="Previous", command=self.select_previous_review_image).pack(side=LEFT)
        Button(review_toolbar, text="Next", command=self.select_next_review_image).pack(side=LEFT, padx=6)
        Button(review_toolbar, text="Delete From Dataset", command=self.delete_selected_review_image).pack(side=LEFT)
        Button(review_toolbar, text="Zoom Out", command=lambda: self.adjust_analysis_review_zoom(0.8)).pack(side=LEFT, padx=(12, 0))
        Button(review_toolbar, text="Fit", command=self.fit_analysis_review_image).pack(side=LEFT, padx=6)
        Button(review_toolbar, text="Zoom In", command=lambda: self.adjust_analysis_review_zoom(1.25)).pack(side=LEFT)
        self.analysis_review_status = StringVar(value="Load an analysis folder to review images.")
        Label(review_toolbar, textvariable=self.analysis_review_status).pack(side=LEFT, padx=12)

        review_body = Frame(parent)
        review_body.pack(fill=BOTH, expand=True, padx=6, pady=(0, 6))
        self.analysis_review_list = Listbox(review_body, width=42, exportselection=False)
        self.analysis_review_list.pack(side=LEFT, fill="y")
        self.analysis_review_list.bind("<<ListboxSelect>>", self.on_analysis_review_select)
        self.analysis_review_list.bind("<Delete>", self.delete_selected_review_image)
        self.analysis_review_list.bind("<BackSpace>", self.delete_selected_review_image)

        review_image_frame = Frame(review_body)
        review_image_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(8, 0))
        self.analysis_review_canvas = Canvas(review_image_frame, background="#f0f0f0", highlightthickness=0)
        review_y_scroll = Scrollbar(review_image_frame, orient="vertical", command=self.analysis_review_canvas.yview)
        review_x_scroll = Scrollbar(review_image_frame, orient="horizontal", command=self.analysis_review_canvas.xview)
        self.analysis_review_canvas.configure(xscrollcommand=review_x_scroll.set, yscrollcommand=review_y_scroll.set)
        self.analysis_review_canvas.grid(row=0, column=0, sticky="nsew")
        review_y_scroll.grid(row=0, column=1, sticky="ns")
        review_x_scroll.grid(row=1, column=0, sticky="ew")
        review_image_frame.rowconfigure(0, weight=1)
        review_image_frame.columnconfigure(0, weight=1)
        self.analysis_review_canvas.bind("<Delete>", self.delete_selected_review_image)
        self.analysis_review_canvas.bind("<BackSpace>", self.delete_selected_review_image)
        self.analysis_review_canvas.bind("<Configure>", lambda _event: self.render_analysis_review_image())
        for widget in (parent, self.analysis_review_list, self.analysis_review_canvas):
            widget.bind("<Left>", self.previous_review_image_key)
            widget.bind("<Up>", self.previous_review_image_key)
            widget.bind("<Right>", self.next_review_image_key)
            widget.bind("<Down>", self.next_review_image_key)
            widget.bind("<Delete>", self.delete_selected_review_image)
            widget.bind("<BackSpace>", self.delete_selected_review_image)
        self.root.bind_all("<Left>", self.previous_review_image_key)
        self.root.bind_all("<Right>", self.next_review_image_key)
        self.root.bind_all("<Delete>", self.delete_selected_review_image)
        self.root.bind_all("<BackSpace>", self.delete_selected_review_image)

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

    def import_spm_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.workspace, title="Choose folder containing SPM files")
        if not folder:
            return
        source_root = Path(folder)
        spm_paths = sorted(path for path in source_root.rglob("*.spm") if path.is_file())
        if not spm_paths:
            messagebox.showinfo("No SPM files", "No .spm files were found in the selected folder.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = source_root / OUTPUT_DIR / f"spm_png_{timestamp}"
        converted = 0
        failed: list[str] = []
        self.status.set(f"Converting {len(spm_paths)} SPM file(s) to flattened PNG...")
        self.root.update_idletasks()
        for idx, spm_path in enumerate(spm_paths, start=1):
            try:
                relative = spm_path.relative_to(source_root)
                png_path = output_root / relative.parent / f"{spm_path.stem}.png"
                self.status.set(f"Converting SPM {idx}/{len(spm_paths)}: {spm_path.name}")
                self.root.update_idletasks()
                convert_spm_to_png(spm_path, png_path)
                shutil.copy2(spm_path, png_path.with_suffix(".spm"))
                converted += 1
            except Exception as exc:
                failed.append(f"{spm_path.name}: {exc}")

        if converted == 0:
            messagebox.showerror("SPM import failed", "No SPM files could be converted.\n\n" + "\n".join(failed[:8]))
            self.status.set("SPM import failed.")
            return

        self.open_root(output_root)
        message = f"Converted {converted} SPM file(s) to:\n{output_root}"
        if failed:
            message += f"\n\nFailed: {len(failed)}\n" + "\n".join(failed[:8])
        self.status.set(f"Converted {converted} SPM file(s). Loaded converted PNG folder.")
        messagebox.showinfo("SPM import complete", message)

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
        return self.classify_path_with_settings(path, float(self.min_area.get()), float(self.max_area.get()), float(self.threshold_bias.get()), use_physical_area=True)

    def classify_path_with_settings(self, path: Path, min_area_value: float, max_area_value: float, threshold_bias: float, use_physical_area: bool = False) -> tuple[np.ndarray, list[OrigamiObject], dict[str, str | int]]:
        if self.model is None:
            raise ValueError("Train or load a classifier first.")
        rgb = load_rgb(path)
        scale_info = self.current_scale_info(rgb, path)
        if use_physical_area and self.min_area_um2 is not None and self.max_area_um2 is not None:
            factor = scale_info.pixels_per_um * scale_info.pixels_per_um
            min_area = max(4, int(round(self.min_area_um2 * factor)))
            max_area = max(min_area + 10, int(round(self.max_area_um2 * factor)))
        else:
            min_area = int(round(min_area_value))
            max_area = int(round(max_area_value))
        objects = detect_origami(rgb, min_area, max_area, threshold_bias, scale_info.pixels_per_um)
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

    def write_all_images_summary(self, output_dir: Path, rows: list[dict[str, str | int]]) -> None:
        summary_path = output_dir / "all_image_counts.csv"
        fieldnames = ["date_folder", "image", "path", "pixels_per_um", "total_detected", *self.states]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

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
        if messagebox.askyesno("Inspect before export?", "Review each classified image before saving?\n\nChoose No to bypass inspection and export all images immediately."):
            self.start_classify_export_review()
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

        self.write_all_images_summary(output_dir, rows)
        self.status.set(f"Classified and exported {len(rows)} images to {output_dir}.")
        messagebox.showinfo("Classify/export complete", f"Saved per-image folders and summary CSV to:\n{output_dir}")

    def start_classify_export_review(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.export_review_output_dir = self.output_dir / CURRENT_RESULTS_DIR / f"all_images_{timestamp}"
        self.export_review_output_dir.mkdir(parents=True, exist_ok=True)
        self.export_review_index = 0
        self.export_review_rows: list[dict[str, str | int]] = []
        self.export_review_skipped: list[Path] = []
        self.export_review_current: tuple[np.ndarray, list[OrigamiObject], dict[str, str | int]] | None = None
        self.export_review_photo: ImageTk.PhotoImage | None = None

        window = Toplevel(self.root)
        self.export_review_window = window
        window.title("Inspect Classify + Export")
        window.geometry("1200x850")

        toolbar = Frame(window)
        toolbar.pack(fill="x", padx=8, pady=8)
        Label(toolbar, text="Min area").pack(side=LEFT)
        self.export_review_min_area = StringVar(value=self.min_area.get())
        Entry(toolbar, textvariable=self.export_review_min_area, width=8).pack(side=LEFT, padx=(4, 8))
        Label(toolbar, text="Max area").pack(side=LEFT)
        self.export_review_max_area = StringVar(value=self.max_area.get())
        Entry(toolbar, textvariable=self.export_review_max_area, width=8).pack(side=LEFT, padx=(4, 8))
        Label(toolbar, text="Bias").pack(side=LEFT)
        self.export_review_bias = StringVar(value=self.threshold_bias.get())
        Entry(toolbar, textvariable=self.export_review_bias, width=8).pack(side=LEFT, padx=(4, 8))
        Button(toolbar, text="Rerun", command=self.rerun_export_review_current).pack(side=LEFT)
        Button(toolbar, text="Save + Next", command=self.save_export_review_current).pack(side=LEFT, padx=8)
        Button(toolbar, text="Skip Image", command=self.skip_export_review_current).pack(side=LEFT)
        Button(toolbar, text="Finish", command=self.finish_export_review).pack(side=LEFT, padx=8)
        self.export_review_status = StringVar(value="")
        Label(toolbar, textvariable=self.export_review_status).pack(side=LEFT, padx=12)

        frame = Frame(window)
        frame.pack(fill=BOTH, expand=True, padx=8, pady=(0, 8))
        self.export_review_canvas = Canvas(frame, background="#f0f0f0", highlightthickness=0)
        y_scroll = Scrollbar(frame, orient="vertical", command=self.export_review_canvas.yview)
        x_scroll = Scrollbar(frame, orient="horizontal", command=self.export_review_canvas.xview)
        self.export_review_canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.export_review_canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        window.bind("<Return>", lambda _event: self.save_export_review_current())
        window.bind("<space>", lambda _event: self.rerun_export_review_current())
        window.bind("<Delete>", lambda _event: self.skip_export_review_current())
        window.protocol("WM_DELETE_WINDOW", self.finish_export_review)
        self.load_export_review_current()

    def load_export_review_current(self) -> None:
        if self.export_review_index >= len(self.images):
            self.finish_export_review()
            return
        path = self.images[self.export_review_index]
        self.export_review_status.set(f"Classifying {self.export_review_index + 1}/{len(self.images)}: {path.name}")
        self.export_review_window.update_idletasks()
        self.rerun_export_review_current()

    def rerun_export_review_current(self) -> None:
        if self.export_review_index >= len(self.images):
            return
        path = self.images[self.export_review_index]
        try:
            rgb, objects, row = self.classify_path_with_settings(
                path,
                float(self.export_review_min_area.get()),
                float(self.export_review_max_area.get()),
                float(self.export_review_bias.get()),
                use_physical_area=False,
            )
        except Exception as exc:
            messagebox.showerror("Classification failed", f"{exc}\n\n{traceback.format_exc()}")
            return
        self.export_review_current = (rgb, objects, row)
        preview = self.annotated_image(rgb, objects)
        self.export_review_photo = ImageTk.PhotoImage(preview)
        self.export_review_canvas.delete("all")
        self.export_review_canvas.create_image(0, 0, image=self.export_review_photo, anchor="nw")
        self.export_review_canvas.configure(scrollregion=(0, 0, preview.width, preview.height))
        counts = ", ".join(f"{state}: {row.get(state, 0)}" for state in self.states)
        self.export_review_status.set(f"{self.export_review_index + 1}/{len(self.images)}  {path.name}  total: {row['total_detected']}  {counts}")

    def save_export_review_current(self) -> None:
        if self.export_review_current is None:
            return
        path = self.images[self.export_review_index]
        rgb, objects, row = self.export_review_current
        self.write_image_result_folder(path, rgb, objects, row, self.export_review_output_dir)
        self.export_review_rows.append(row)
        self.export_review_index += 1
        self.export_review_current = None
        self.load_export_review_current()

    def skip_export_review_current(self) -> None:
        if self.export_review_index < len(self.images):
            self.export_review_skipped.append(self.images[self.export_review_index])
            self.export_review_index += 1
            self.export_review_current = None
            self.load_export_review_current()

    def finish_export_review(self) -> None:
        if not hasattr(self, "export_review_output_dir"):
            return
        self.write_all_images_summary(self.export_review_output_dir, self.export_review_rows)
        if hasattr(self, "export_review_window") and self.export_review_window.winfo_exists():
            self.export_review_window.destroy()
        self.status.set(f"Reviewed export saved {len(self.export_review_rows)} image(s), skipped {len(self.export_review_skipped)}.")
        messagebox.showinfo(
            "Reviewed export complete",
            f"Saved reviewed export to:\n{self.export_review_output_dir}\n\nSaved: {len(self.export_review_rows)}\nSkipped: {len(self.export_review_skipped)}",
        )

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

    def state_display_label(self, state: str) -> str:
        label = self.analysis_state_labels.get(state, "").strip()
        return label or state

    def generate_count_plots(self, csv_path: Path) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows, state_columns, group_mode = self.read_count_plot_data(csv_path)

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
            metric_fields = ["grouping", "label", "path", "total_count", *[f"count_{s}" for s in state_columns], *[f"fraction_{s}" for s in state_columns]]
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writeheader()
            for idx, row in enumerate(rows):
                metric_row = {
                    "grouping": group_mode,
                    "label": labels[idx],
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
            plt.bar(x, counts[:, j], bottom=bottom, label=self.state_display_label(state), color=STATE_COLORS[j % len(STATE_COLORS)])
            bottom += counts[:, j]
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylabel("Origami count")
        plt.title(f"Origami counts by {group_mode.lower()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "counts_stacked.png", dpi=180, bbox_inches="tight")
        plt.close()

        self.plot_fraction_stacked_bars(plt, rows, state_columns, group_mode)
        plt.savefig(output_dir / "fractions_stacked.png", dpi=180, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(fig_width, 4))
        plt.bar(x, totals, color="#555555")
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylabel("Total classified origami")
        plt.title(f"Total origami detected/classified by {group_mode.lower()}")
        plt.tight_layout()
        plt.savefig(output_dir / "total_counts.png", dpi=180, bbox_inches="tight")
        plt.close()

        scatter_states = self.selected_scatter_states(state_columns)
        if scatter_states is not None:
            a_state, b_state = self.plot_fraction_scatter(plt, rows, fractions, totals, state_columns, group_mode)
            plt.savefig(output_dir / f"fraction_{a_state}_vs_{b_state}.png", dpi=180, bbox_inches="tight")
            plt.close()

        raw_rows, raw_state_columns = self.read_filtered_count_rows_and_states(csv_path)
        ratio_states = self.plot_ab_ratio_change(plt, raw_rows, raw_state_columns)
        if ratio_states is not None:
            a_state, b_state = ratio_states
            plt.savefig(output_dir / f"ab_ratio_change_{a_state}_over_{b_state}.png", dpi=180, bbox_inches="tight")
            plt.close()
        delta_rows, delta_state_columns = self.read_filtered_count_rows_and_states(csv_path)
        delta_state = self.plot_state_delta_change(plt, delta_rows, delta_state_columns)
        if delta_state is not None:
            plt.savefig(output_dir / f"state_fraction_delta_{delta_state}.png", dpi=180, bbox_inches="tight")
            plt.close()

        return output_dir

    def plot_image_from_current_figure(self) -> Image.Image:
        import matplotlib.pyplot as plt

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def build_count_plot_previews(self, csv_path: Path) -> dict[str, Image.Image]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows, state_columns, group_mode = self.read_count_plot_data(csv_path)

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
            plt.bar(x, counts[:, j], bottom=bottom, label=self.state_display_label(state), color=STATE_COLORS[j % len(STATE_COLORS)])
            bottom += counts[:, j]
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylabel("Origami count")
        plt.title(f"Origami counts by {group_mode.lower()}")
        plt.legend()
        plt.tight_layout()
        previews["counts_stacked.png"] = self.plot_image_from_current_figure()

        self.plot_fraction_stacked_bars(plt, rows, state_columns, group_mode)
        previews["fractions_stacked.png"] = self.plot_image_from_current_figure()

        plt.figure(figsize=(fig_width, 4))
        plt.bar(x, totals, color="#555555")
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylabel("Total classified origami")
        plt.title(f"Total origami detected/classified by {group_mode.lower()}")
        plt.tight_layout()
        previews["total_counts.png"] = self.plot_image_from_current_figure()

        scatter_states = self.selected_scatter_states(state_columns)
        if scatter_states is not None:
            a_state, b_state = self.plot_fraction_scatter(plt, rows, fractions, totals, state_columns, group_mode)
            previews[f"fraction_{a_state}_vs_{b_state}.png"] = self.plot_image_from_current_figure()

        raw_rows, raw_state_columns = self.read_filtered_count_rows_and_states(csv_path)
        ratio_states = self.plot_ab_ratio_change(plt, raw_rows, raw_state_columns)
        if ratio_states is not None:
            a_state, b_state = ratio_states
            previews[f"ab_ratio_change_{a_state}_over_{b_state}.png"] = self.plot_image_from_current_figure()
        delta_rows, delta_state_columns = self.read_filtered_count_rows_and_states(csv_path)
        delta_state = self.plot_state_delta_change(plt, delta_rows, delta_state_columns)
        if delta_state is not None:
            previews[f"state_fraction_delta_{delta_state}.png"] = self.plot_image_from_current_figure()

        return previews

    def parse_origami_label(self, image_name: str) -> str:
        match = re.search(r"origami([0-9]+f?)_", image_name, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        match = re.search(r"origami([0-9]+f?)", image_name, re.IGNORECASE)
        return match.group(1).lower() if match else "unknown"

    def origami_sort_value(self, image_name: str) -> tuple[int, str]:
        label = self.parse_origami_label(image_name)
        match = re.match(r"([0-9]+)(.*)", label)
        if not match:
            return (10**9, label)
        return (int(match.group(1)), match.group(2))

    def scan_size_sort_value(self, row: dict[str, str]) -> tuple[float, float]:
        try:
            pixels_per_um = float(row.get("pixels_per_um", "") or 0)
        except ValueError:
            pixels_per_um = 0.0

        if pixels_per_um > 0:
            scan_size_um = self.count_row_scan_size_um(row)
            if scan_size_um is not None:
                return (scan_size_um, -pixels_per_um)
            return (1.0 / pixels_per_um, -pixels_per_um)
        return (float("inf"), 0.0)

    def count_row_scan_size_um(self, row: dict[str, str]) -> float | None:
        try:
            pixels_per_um = float(row.get("pixels_per_um", "") or 0)
        except ValueError:
            return None
        if pixels_per_um <= 0:
            return None
        for path in self.count_row_image_candidates(row):
            try:
                spm_scan_size = parse_spm_scan_size_um(paired_spm_path(path))
                if spm_scan_size is not None and spm_scan_size > 0:
                    return spm_scan_size
                rgb = load_rgb(path)
                scan_minr, scan_minc, scan_maxr, scan_maxc = scan_bbox(rgb)
                scan_width_px = scan_maxc - scan_minc
                if scan_width_px > 0:
                    return scan_width_px / pixels_per_um
            except Exception:
                pass
        return None

    def count_row_scale_label(self, row: dict[str, str]) -> str:
        scan_size_um = self.count_row_scan_size_um(row)
        if scan_size_um is not None:
            return f"{format_um(self.normalized_scan_size_um(scan_size_um))} um"
        pixels_per_um = row.get("pixels_per_um", "")
        return f"{pixels_per_um} px/um" if pixels_per_um else ""

    def normalized_scan_size_um(self, scan_size_um: float) -> float:
        common_sizes = [0.37, 0.55, 1.1, 1.67, 3.33, 5.0, 10.0]
        for common_size in common_sizes:
            if abs(scan_size_um - common_size) / common_size <= 0.04:
                return common_size
        return scan_size_um

    def count_row_image_candidates(self, row: dict[str, str]) -> list[Path]:
        candidates: list[Path] = []
        path_text = row.get("path", "")
        if path_text:
            candidates.append(Path(path_text))
        workspace = getattr(self, "workspace", Path.cwd())
        date_folder = row.get("date_folder", "")
        image_name = row.get("image", "")
        analysis_folder = row.get("_analysis_folder", "")
        if analysis_folder and image_name:
            base = Path(analysis_folder)
            candidates.append(base / image_name)
            candidates.append(base / Path(image_name).stem / image_name)
        if date_folder and image_name:
            candidates.append(Path(workspace) / date_folder / image_name)
        if image_name:
            candidates.append(Path(workspace) / image_name)
            if image_name.endswith("_annotated.png"):
                candidates.append(Path(workspace) / image_name.replace("_annotated.png", ".png"))

        existing: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen and candidate.exists():
                seen.add(key)
                existing.append(candidate)
        if not existing and image_name:
            search_names = [image_name]
            if image_name.endswith("_annotated.png"):
                search_names.append(image_name.replace("_annotated.png", ".png"))
            for search_name in search_names:
                for candidate in Path(workspace).rglob(search_name):
                    key = str(candidate)
                    if key not in seen and candidate.is_file():
                        seen.add(key)
                        existing.append(candidate)
                if existing:
                    break
        existing.sort(key=lambda path: ("analysis_output" in path.parts, path.name.endswith("_annotated.png"), str(path)))
        return existing

    def sorted_count_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return sorted(
            rows,
            key=lambda row: (
                row.get("dataset", row.get("_analysis_dataset", "")),
                *self.origami_sort_value(row.get("image", "")),
                *self.scan_size_sort_value(row),
                row.get("image", ""),
            ),
        )

    def analysis_group_mode(self) -> str:
        mode_var = getattr(self, "analysis_plot_group", None)
        mode = mode_var.get() if mode_var is not None else "Images"
        modes = {"Images", "Dataset", "Origami", "Scale", "Origami + Scale", "Dataset + Origami", "Dataset + Scale", "Dataset + Origami + Scale"}
        return mode if mode in modes else "Images"

    def scale_sort_value_from_label(self, scale_label: str) -> float:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", scale_label)
        return float(match.group(1)) if match else float("inf")

    def plot_row_origami_label(self, row: dict[str, str]) -> str:
        label = row.get("origami_label", "")
        if label:
            return label
        image_name = row.get("image", "")
        if image_name.lower().startswith("origami"):
            return self.parse_origami_label(image_name)
        return "all"

    def plot_row_scale_label(self, row: dict[str, str]) -> str:
        scale_label = row.get("scale", "")
        if scale_label:
            return scale_label
        return self.count_row_scale_label(row) or "all"

    def sorted_origami_labels(self, labels: set[str]) -> list[str]:
        return sorted(labels, key=lambda label: self.origami_sort_value(f"origami{label}_") if label != "all" else (10**9, "all"))

    def sorted_scale_labels(self, labels: set[str]) -> list[str]:
        return sorted(labels, key=lambda label: self.scale_sort_value_from_label(label) if label != "all" else float("inf"))

    def dataset_order_for_rows(self, rows: list[dict[str, str]]) -> list[str]:
        labels: list[str] = []
        source_rows = getattr(self, "analysis_source_rows", None) or rows
        for row in source_rows:
            label = row.get("dataset", row.get("_analysis_dataset", ""))
            if label and label not in labels:
                labels.append(label)
        for row in rows:
            label = row.get("dataset", row.get("_analysis_dataset", ""))
            if label and label not in labels:
                labels.append(label)
        return labels

    def fraction_stacked_bar_rows(self, rows: list[dict[str, str]], state_columns: list[str]) -> tuple[list[dict[str, str]], bool]:
        dataset_order = self.dataset_order_for_rows(rows)
        if len(dataset_order) < 2:
            return rows, False

        dataset_index = {label: idx for idx, label in enumerate(dataset_order)}
        order_mode_var = getattr(self, "analysis_x_axis_order", None)
        dataset_first = order_mode_var is not None and order_mode_var.get() == "Dataset first"
        grouped: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in rows:
            dataset_label = row.get("dataset", row.get("_analysis_dataset", ""))
            if not dataset_label:
                continue
            origami_label = self.plot_row_origami_label(row)
            scale_label = self.plot_row_scale_label(row)
            key = (origami_label, scale_label, dataset_label)
            label_parts = []
            if dataset_first:
                label_parts.append(dataset_label)
                if origami_label != "all":
                    label_parts.append(f"origami{origami_label}")
                if scale_label != "all":
                    label_parts.append(scale_label)
            else:
                if origami_label != "all":
                    label_parts.append(f"origami{origami_label}")
                if scale_label != "all":
                    label_parts.append(scale_label)
                label_parts.append(dataset_label)
            group = grouped.setdefault(
                key,
                {
                    "date_folder": "",
                    "image": "\n".join(label_parts),
                    "path": "",
                    "dataset": dataset_label,
                    "pixels_per_um": "",
                    "total_detected": "0",
                    "origami_label": origami_label,
                    "scale": scale_label,
                },
            )
            group["total_detected"] = f"{float(group.get('total_detected', 0) or 0) + float(row.get('total_detected', 0) or 0):.6g}"
            for state in state_columns:
                group[state] = f"{float(group.get(state, 0) or 0) + float(row.get(state, 0) or 0):.6g}"

        def comparison_sort_key(row: dict[str, str]) -> tuple:
            origami_key = self.origami_sort_value(f"origami{row.get('origami_label', 'all')}_")
            scale_key = (self.scale_sort_value_from_label(row.get("scale", "all")), row.get("scale", ""))
            dataset_key = (dataset_index.get(row.get("dataset", ""), len(dataset_index)), row.get("dataset", ""))
            if dataset_first:
                return (*dataset_key, *origami_key, *scale_key)
            return (*origami_key, *scale_key, *dataset_key)

        comparison_rows = sorted(
            grouped.values(),
            key=comparison_sort_key,
        )
        return comparison_rows or rows, bool(comparison_rows)

    def plot_fraction_stacked_bars(self, plt, rows: list[dict[str, str]], state_columns: list[str], group_mode: str) -> None:
        plot_rows, comparison_mode = self.fraction_stacked_bar_rows(rows, state_columns)
        labels = [row.get("image") or f"group_{idx + 1}" for idx, row in enumerate(plot_rows)]
        short_labels = [label[:34] + ("..." if len(label) > 34 else "") for label in labels]
        counts = np.asarray([[float(row.get(state, 0) or 0) for state in state_columns] for row in plot_rows], dtype=float)
        totals = counts.sum(axis=1)
        totals_for_fraction = np.where(totals > 0, totals, 1)
        fractions = counts / totals_for_fraction[:, None]
        x = np.arange(len(plot_rows))
        fig_width = max(8, min(28, len(plot_rows) * 0.62 + 4))

        plt.figure(figsize=(fig_width, 5))
        bottom = np.zeros(len(plot_rows))
        for j, state in enumerate(state_columns):
            plt.bar(x, fractions[:, j], bottom=bottom, label=self.state_display_label(state), color=STATE_COLORS[j % len(STATE_COLORS)])
            bottom += fractions[:, j]
        if comparison_mode:
            dataset_order = self.dataset_order_for_rows(plot_rows)
            if len(dataset_order) >= 2:
                second_dataset = dataset_order[1]
                outlined_x = [idx for idx, row in enumerate(plot_rows) if row.get("dataset", "") == second_dataset]
                if outlined_x:
                    plt.bar(
                        outlined_x,
                        np.ones(len(outlined_x)),
                        width=0.82,
                        fill=False,
                        edgecolor="#1f77b4",
                        linewidth=2.2,
                        label=f"{second_dataset} outline",
                    )
        plt.xticks(x, short_labels, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Fraction of classified origami")
        if comparison_mode:
            order_mode_var = getattr(self, "analysis_x_axis_order", None)
            dataset_first = order_mode_var is not None and order_mode_var.get() == "Dataset first"
            order_text = "dataset first, then origami/scale" if dataset_first else "origami/scale first, then dataset"
            title_group = f"dataset comparison ({order_text})"
        else:
            title_group = group_mode.lower()
        plt.title(f"State fractions by {title_group}")
        plt.legend()
        plt.tight_layout()

    def ab_ratio_states(self, state_columns: list[str]) -> tuple[str, str] | None:
        a_state = next((state for state in state_columns if state.lower() == "a"), None)
        b_state = next((state for state in state_columns if state.lower() == "b"), None)
        if a_state is None or b_state is None:
            return None
        return a_state, b_state

    def ab_ratio_change_rows(self, rows: list[dict[str, str]], state_columns: list[str]) -> tuple[list[dict[str, float | str]], tuple[str, str] | None, list[str]]:
        ratio_states = self.ab_ratio_states(state_columns)
        if ratio_states is None:
            return [], None, []
        a_state, b_state = ratio_states
        dataset_order = self.dataset_order_for_rows(rows)
        if len(dataset_order) != 2:
            return [], ratio_states, dataset_order
        dataset_a, dataset_b = dataset_order

        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for row in rows:
            dataset_label = row.get("dataset", row.get("_analysis_dataset", ""))
            if dataset_label not in {dataset_a, dataset_b}:
                continue
            origami_label = self.parse_origami_label(row.get("image", ""))
            key = (origami_label, dataset_label)
            bucket = grouped.setdefault(key, {state: 0.0 for state in state_columns})
            bucket.setdefault("_total", 0.0)
            for state in state_columns:
                value = float(row.get(state, 0) or 0)
                bucket[state] = bucket.get(state, 0.0) + value
                bucket["_total"] = bucket.get("_total", 0.0) + value

        ratio_rows: list[dict[str, float | str]] = []
        origami_labels = self.sorted_origami_labels({origami_label for origami_label, _dataset_label in grouped})
        for origami_label in origami_labels:
            first_counts = grouped.get((origami_label, dataset_a))
            second_counts = grouped.get((origami_label, dataset_b))
            if first_counts is None or second_counts is None:
                continue
            first_total = first_counts.get("_total", 0.0)
            second_total = second_counts.get("_total", 0.0)
            if first_total <= 0 or second_total <= 0:
                continue
            first_a_fraction = first_counts.get(a_state, 0.0) / first_total
            first_b_fraction = first_counts.get(b_state, 0.0) / first_total
            second_a_fraction = second_counts.get(a_state, 0.0) / second_total
            second_b_fraction = second_counts.get(b_state, 0.0) / second_total
            if first_b_fraction <= 0 or second_b_fraction <= 0:
                continue
            first_ratio = first_a_fraction / first_b_fraction
            second_ratio = second_a_fraction / second_b_fraction
            ratio_rows.append(
                {
                    "origami_label": origami_label,
                    "first_ratio": first_ratio,
                    "second_ratio": second_ratio,
                    "change": second_ratio - first_ratio,
                }
            )
        return ratio_rows, ratio_states, dataset_order

    def plot_ab_ratio_change(self, plt, rows: list[dict[str, str]], state_columns: list[str]) -> tuple[str, str] | None:
        ratio_rows, ratio_states, dataset_order = self.ab_ratio_change_rows(rows, state_columns)
        if ratio_states is None or len(dataset_order) != 2 or not ratio_rows:
            return None
        a_state, b_state = ratio_states
        dataset_a, dataset_b = dataset_order
        labels = [f"origami{row['origami_label']}" for row in ratio_rows]
        changes = np.asarray([float(row["change"]) for row in ratio_rows], dtype=float)
        x = np.arange(len(ratio_rows))
        fig_width = max(8, min(24, len(ratio_rows) * 0.6 + 4))

        plt.figure(figsize=(fig_width, 5))
        colors = ["#2ca02c" if change >= 0 else "#d62728" for change in changes]
        plt.bar(x, changes, color=colors, edgecolor="#222222", linewidth=0.6)
        plt.axhline(0, color="#333333", linewidth=1.0)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel(f"Change in fraction {self.state_display_label(a_state)}/fraction {self.state_display_label(b_state)}")
        plt.title(f"Fraction {self.state_display_label(a_state)}/{self.state_display_label(b_state)} ratio change: {dataset_b} minus {dataset_a}")
        plt.grid(axis="y", alpha=0.25)
        for idx, row in enumerate(ratio_rows):
            change = float(row["change"])
            first_ratio = float(row["first_ratio"])
            second_ratio = float(row["second_ratio"])
            label = f"{first_ratio:.2g} -> {second_ratio:.2g}"
            va = "bottom" if change >= 0 else "top"
            offset = 3 if change >= 0 else -3
            plt.annotate(label, (idx, change), textcoords="offset points", xytext=(0, offset), ha="center", va=va, fontsize=8)
        plt.tight_layout()
        return a_state, b_state

    def selected_delta_state(self, state_columns: list[str]) -> str | None:
        if not state_columns:
            return None
        delta_var = getattr(self, "analysis_delta_state", None)
        state = delta_var.get() if delta_var is not None else ""
        if state not in state_columns:
            state = state_columns[0]
            if delta_var is not None:
                delta_var.set(state)
        return state

    def state_delta_change_rows(self, rows: list[dict[str, str]], state_columns: list[str], state: str) -> tuple[list[dict[str, float | str]], list[str]]:
        dataset_order = self.dataset_order_for_rows(rows)
        if len(dataset_order) != 2 or state not in state_columns:
            return [], dataset_order
        dataset_a, dataset_b = dataset_order

        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for row in rows:
            dataset_label = row.get("dataset", row.get("_analysis_dataset", ""))
            if dataset_label not in {dataset_a, dataset_b}:
                continue
            origami_label = self.parse_origami_label(row.get("image", ""))
            key = (origami_label, dataset_label)
            bucket = grouped.setdefault(key, {state_name: 0.0 for state_name in state_columns})
            bucket.setdefault("_total", 0.0)
            for state_name in state_columns:
                value = float(row.get(state_name, 0) or 0)
                bucket[state_name] = bucket.get(state_name, 0.0) + value
                bucket["_total"] = bucket.get("_total", 0.0) + value

        delta_rows: list[dict[str, float | str]] = []
        origami_labels = self.sorted_origami_labels({origami_label for origami_label, _dataset_label in grouped})
        for origami_label in origami_labels:
            first_counts = grouped.get((origami_label, dataset_a))
            second_counts = grouped.get((origami_label, dataset_b))
            if first_counts is None or second_counts is None:
                continue
            first_total = first_counts.get("_total", 0.0)
            second_total = second_counts.get("_total", 0.0)
            if first_total <= 0 or second_total <= 0:
                continue
            first_fraction = first_counts.get(state, 0.0) / first_total
            second_fraction = second_counts.get(state, 0.0) / second_total
            delta_rows.append(
                {
                    "origami_label": origami_label,
                    "first_fraction": first_fraction,
                    "second_fraction": second_fraction,
                    "change": second_fraction - first_fraction,
                }
            )
        return delta_rows, dataset_order

    def plot_state_delta_change(self, plt, rows: list[dict[str, str]], state_columns: list[str]) -> str | None:
        state = self.selected_delta_state(state_columns)
        if state is None:
            return None
        delta_rows, dataset_order = self.state_delta_change_rows(rows, state_columns, state)
        if len(dataset_order) != 2 or not delta_rows:
            return None
        dataset_a, dataset_b = dataset_order
        labels = [f"origami{row['origami_label']}" for row in delta_rows]
        changes = np.asarray([float(row["change"]) for row in delta_rows], dtype=float)
        x = np.arange(len(delta_rows))
        fig_width = max(8, min(24, len(delta_rows) * 0.6 + 4))

        plt.figure(figsize=(fig_width, 5))
        colors = ["#2ca02c" if change >= 0 else "#d62728" for change in changes]
        plt.bar(x, changes, color=colors, edgecolor="#222222", linewidth=0.6)
        plt.axhline(0, color="#333333", linewidth=1.0)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel(f"Delta fraction {self.state_display_label(state)}")
        plt.title(f"Fraction {self.state_display_label(state)} change: {dataset_b} minus {dataset_a}")
        plt.grid(axis="y", alpha=0.25)
        for idx, row in enumerate(delta_rows):
            change = float(row["change"])
            first_fraction = float(row["first_fraction"])
            second_fraction = float(row["second_fraction"])
            label = f"{first_fraction:.2g} -> {second_fraction:.2g}"
            va = "bottom" if change >= 0 else "top"
            offset = 3 if change >= 0 else -3
            plt.annotate(label, (idx, change), textcoords="offset points", xytext=(0, offset), ha="center", va=va, fontsize=8)
        plt.tight_layout()
        return state

    def plot_fraction_scatter(self, plt, rows: list[dict[str, str]], fractions: np.ndarray, totals: np.ndarray, state_columns: list[str], group_mode: str) -> tuple[str, str] | None:
        scatter_states = self.selected_scatter_states(state_columns)
        if scatter_states is None:
            return None
        a_state, b_state = scatter_states
        a_label = self.state_display_label(a_state)
        b_label = self.state_display_label(b_state)
        a_idx, b_idx = state_columns.index(a_state), state_columns.index(b_state)
        origami_labels = [self.plot_row_origami_label(row) for row in rows]
        scale_labels = [self.plot_row_scale_label(row) for row in rows]
        sorted_origami = self.sorted_origami_labels(set(origami_labels))
        sorted_scales = self.sorted_scale_labels(set(scale_labels))
        color_map = {label: STATE_COLORS[idx % len(STATE_COLORS)] for idx, label in enumerate(sorted_origami)}
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
        marker_map = {label: markers[idx % len(markers)] for idx, label in enumerate(sorted_scales)}

        plt.figure(figsize=(8, 6))
        for idx, row in enumerate(rows):
            origami_label = origami_labels[idx]
            scale_label = scale_labels[idx]
            plt.scatter(
                fractions[idx, a_idx],
                fractions[idx, b_idx],
                s=max(float(totals[idx]), 10) * 3,
                alpha=0.78,
                color=color_map[origami_label],
                marker=marker_map[scale_label],
                edgecolors="#222222",
                linewidths=0.5,
            )
        plt.xlabel(f"Fraction {a_label}")
        plt.ylabel(f"Fraction {b_label}")
        plt.title(f"{a_label} vs {b_label} fraction by {group_mode.lower()}")
        plt.xlim(-0.03, 1.03)
        plt.ylim(-0.03, 1.03)
        plt.grid(True, alpha=0.25)

        color_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", label=f"origami{label}" if label != "all" else "all origami", markerfacecolor=color_map[label], markeredgecolor="#222222", markersize=8)
            for label in sorted_origami
        ]
        shape_handles = [
            plt.Line2D([0], [0], marker=marker_map[label], linestyle="", label=label, color="#444444", markerfacecolor="#444444", markersize=8)
            for label in sorted_scales
        ]
        first_legend = plt.legend(handles=color_handles, title="Origami", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        plt.gca().add_artist(first_legend)
        plt.legend(handles=shape_handles, title="Scale", loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0)
        plt.tight_layout()
        return a_state, b_state

    def dataset_overlay_plot_rows(self, rows: list[dict[str, str]], state_columns: list[str], dataset_labels: list[str]) -> list[dict[str, object]]:
        dataset_a, dataset_b = dataset_labels
        grouped: dict[tuple[str, str, str], dict[str, object]] = {}
        for row in rows:
            dataset_label = row.get("dataset", row.get("_analysis_dataset", ""))
            if dataset_label not in {dataset_a, dataset_b}:
                continue
            origami_label = self.parse_origami_label(row.get("image", ""))
            scale_label = self.count_row_scale_label(row) or "unknown scale"
            key = (origami_label, scale_label, dataset_label)
            group = grouped.setdefault(
                key,
                {
                    "dataset": dataset_label,
                    "origami_label": origami_label,
                    "scale": scale_label,
                    "total": 0.0,
                    "counts": {state: 0.0 for state in state_columns},
                },
            )
            counts = group["counts"]
            if not isinstance(counts, dict):
                continue
            for state in state_columns:
                value = float(row.get(state, 0) or 0)
                counts[state] = float(counts.get(state, 0.0)) + value
                group["total"] = float(group["total"]) + value

        paired_rows: list[dict[str, object]] = []
        pair_keys = sorted(
            {(origami_label, scale_label) for origami_label, scale_label, _dataset_label in grouped},
            key=lambda key: (*self.origami_sort_value(f"origami{key[0]}_"), self.scale_sort_value_from_label(key[1]), key[1]),
        )
        for origami_label, scale_label in pair_keys:
            a_row = grouped.get((origami_label, scale_label, dataset_a))
            b_row = grouped.get((origami_label, scale_label, dataset_b))
            if a_row is None or b_row is None:
                continue
            if float(a_row.get("total", 0.0) or 0.0) <= 0 or float(b_row.get("total", 0.0) or 0.0) <= 0:
                continue
            paired_rows.append(
                {
                    "origami_label": origami_label,
                    "scale": scale_label,
                    dataset_a: a_row,
                    dataset_b: b_row,
                }
            )
        return paired_rows

    def plot_dataset_overlay(self, plt, paired_rows: list[dict[str, object]], state_columns: list[str], dataset_labels: list[str]) -> tuple[str, str] | None:
        scatter_states = self.selected_scatter_states(state_columns)
        if scatter_states is None:
            return None
        dataset_a, dataset_b = dataset_labels
        x_state, y_state = scatter_states
        x_label = self.state_display_label(x_state)
        y_label = self.state_display_label(y_state)

        origami_labels = [str(row["origami_label"]) for row in paired_rows]
        scale_labels = [str(row["scale"]) for row in paired_rows]
        sorted_origami = self.sorted_origami_labels(set(origami_labels))
        sorted_scales = self.sorted_scale_labels(set(scale_labels))
        color_map = {label: STATE_COLORS[idx % len(STATE_COLORS)] for idx, label in enumerate(sorted_origami)}
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
        marker_map = {label: markers[idx % len(markers)] for idx, label in enumerate(sorted_scales)}

        def fraction(point: dict[str, object], state: str) -> float:
            counts = point.get("counts", {})
            if not isinstance(counts, dict):
                return 0.0
            total = float(point.get("total", 0.0) or 0.0)
            return float(counts.get(state, 0.0) or 0.0) / total if total > 0 else 0.0

        plt.figure(figsize=(8.8, 6.2))
        for row in paired_rows:
            origami_label = str(row["origami_label"])
            scale_label = str(row["scale"])
            a_point = row[dataset_a]
            b_point = row[dataset_b]
            if not isinstance(a_point, dict) or not isinstance(b_point, dict):
                continue
            x_a, y_a = fraction(a_point, x_state), fraction(a_point, y_state)
            x_b, y_b = fraction(b_point, x_state), fraction(b_point, y_state)
            color = color_map[origami_label]
            marker = marker_map[scale_label]
            plt.annotate(
                "",
                xy=(x_b, y_b),
                xytext=(x_a, y_a),
                arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.72, "lw": 1.6, "shrinkA": 4, "shrinkB": 4},
            )
            plt.scatter(x_a, y_a, s=85, marker=marker, facecolors="white", edgecolors=color, linewidths=1.6, alpha=0.92)
            plt.scatter(x_b, y_b, s=85, marker=marker, color=color, edgecolors="#222222", linewidths=0.6, alpha=0.92)

        plt.xlabel(f"Fraction {x_label}")
        plt.ylabel(f"Fraction {y_label}")
        plt.title(f"Dataset overlay: {dataset_a} to {dataset_b}")
        plt.xlim(-0.03, 1.03)
        plt.ylim(-0.03, 1.03)
        plt.grid(True, alpha=0.25)

        color_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", label=f"origami{label}" if label != "all" else "all origami", markerfacecolor=color_map[label], markeredgecolor="#222222", markersize=8)
            for label in sorted_origami
        ]
        shape_handles = [
            plt.Line2D([0], [0], marker=marker_map[label], linestyle="", label=label, color="#444444", markerfacecolor="#444444", markersize=8)
            for label in sorted_scales
        ]
        dataset_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", label=dataset_a, markerfacecolor="white", markeredgecolor="#444444", markersize=8),
            plt.Line2D([0], [0], marker="o", linestyle="", label=dataset_b, markerfacecolor="#444444", markeredgecolor="#222222", markersize=8),
        ]
        first_legend = plt.legend(handles=color_handles, title="Origami", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        plt.gca().add_artist(first_legend)
        second_legend = plt.legend(handles=shape_handles, title="Scale", loc="center left", bbox_to_anchor=(1.02, 0.48), borderaxespad=0)
        plt.gca().add_artist(second_legend)
        plt.legend(handles=dataset_handles, title="Dataset", loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0)
        plt.tight_layout()
        return x_state, y_state

    def grouped_count_plot_rows(self, rows: list[dict[str, str]], state_columns: list[str]) -> list[dict[str, str]]:
        mode = self.analysis_group_mode()
        if mode == "Images":
            return rows

        grouped: dict[tuple[str, ...], dict[str, str]] = {}
        for row in rows:
            image_name = row.get("image", "")
            dataset_label = row.get("dataset", row.get("_analysis_dataset", ""))
            origami_label = self.parse_origami_label(image_name)
            scale_label = self.count_row_scale_label(row)
            dataset_label = row.get("dataset", row.get("_analysis_dataset", "dataset"))
            if mode == "Dataset":
                key = (dataset_label,)
                label = dataset_label
                sort_key = dataset_label
                group_origami_label = "all"
                group_scale_label = "all"
            elif mode == "Origami":
                key = (origami_label,)
                label = f"origami{origami_label}"
                sort_key = f"{self.origami_sort_value(image_name)[0]:09d}_{origami_label}"
                group_origami_label = origami_label
                group_scale_label = "all"
            elif mode == "Scale":
                key = (scale_label,)
                label = scale_label
                sort_key = f"{self.scale_sort_value_from_label(scale_label):012.6f}_{scale_label}"
                group_origami_label = "all"
                group_scale_label = scale_label
            elif mode == "Origami + Scale":
                key = (origami_label, scale_label)
                label = f"origami{origami_label} {scale_label}"
                sort_key = f"{self.origami_sort_value(image_name)[0]:09d}_{self.scale_sort_value_from_label(scale_label):012.6f}_{origami_label}_{scale_label}"
                group_origami_label = origami_label
                group_scale_label = scale_label
            elif mode == "Dataset + Origami":
                key = (dataset_label, origami_label)
                label = f"{dataset_label} origami{origami_label}"
                sort_key = f"{dataset_label}_{self.origami_sort_value(image_name)[0]:09d}_{origami_label}"
                group_origami_label = origami_label
                group_scale_label = "all"
            elif mode == "Dataset + Scale":
                key = (dataset_label, scale_label)
                label = f"{dataset_label} {scale_label}"
                sort_key = f"{dataset_label}_{self.scale_sort_value_from_label(scale_label):012.6f}_{scale_label}"
                group_origami_label = "all"
                group_scale_label = scale_label
            else:
                key = (dataset_label, origami_label, scale_label)
                label = f"{dataset_label} origami{origami_label} {scale_label}"
                sort_key = f"{dataset_label}_{self.origami_sort_value(image_name)[0]:09d}_{self.scale_sort_value_from_label(scale_label):012.6f}_{origami_label}_{scale_label}"
                group_origami_label = origami_label
                group_scale_label = scale_label

            group = grouped.setdefault(
                key,
                {
                    "date_folder": "",
                    "image": label,
                    "path": "",
                    "dataset": dataset_label,
                    "pixels_per_um": "",
                    "total_detected": "0",
                    "origami_label": group_origami_label,
                    "scale": group_scale_label,
                    "_sort_key": sort_key,
                },
            )
            group["total_detected"] = f"{float(group.get('total_detected', 0) or 0) + float(row.get('total_detected', 0) or 0):.6g}"
            for state in state_columns:
                group[state] = f"{float(group.get(state, 0) or 0) + float(row.get(state, 0) or 0):.6g}"

        sorted_groups = sorted(grouped.values(), key=lambda row: row.get("_sort_key", ""))
        for row in sorted_groups:
            row.pop("_sort_key", None)
        return sorted_groups

    def configure_analysis_filter_options(self, rows: list[dict[str, str]]) -> None:
        self.analysis_dataset_options = sorted({row.get("dataset", row.get("_analysis_dataset", "")) for row in rows if row.get("dataset", row.get("_analysis_dataset", ""))})
        self.analysis_scale_options = sorted({self.count_row_scale_label(row) for row in rows}, key=self.scale_sort_value_from_label)
        self.analysis_origami_options = sorted({self.parse_origami_label(row.get("image", "")) for row in rows}, key=lambda label: self.origami_sort_value(f"origami{label}_"))
        self.analysis_state_options = self.count_state_columns(self.analysis_fieldnames, rows)
        if self.analysis_dataset_filter is not None:
            self.analysis_dataset_filter &= set(self.analysis_dataset_options)
            if not self.analysis_dataset_filter:
                self.analysis_dataset_filter = None
        if self.analysis_scale_filter is not None:
            self.analysis_scale_filter &= set(self.analysis_scale_options)
            if not self.analysis_scale_filter:
                self.analysis_scale_filter = None
        if self.analysis_origami_filter is not None:
            self.analysis_origami_filter &= set(self.analysis_origami_options)
            if not self.analysis_origami_filter:
                self.analysis_origami_filter = None
        if self.analysis_state_filter is not None:
            self.analysis_state_filter &= set(self.analysis_state_options)
            if not self.analysis_state_filter:
                self.analysis_state_filter = None
        self.update_analysis_filter_status()

    def filtered_count_plot_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        filtered = []
        for row in rows:
            if self.analysis_dataset_filter is not None and row.get("dataset", row.get("_analysis_dataset", "")) not in self.analysis_dataset_filter:
                continue
            if self.analysis_scale_filter is not None and self.count_row_scale_label(row) not in self.analysis_scale_filter:
                continue
            if self.analysis_origami_filter is not None and self.parse_origami_label(row.get("image", "")) not in self.analysis_origami_filter:
                continue
            filtered.append(row)
        return filtered

    def update_analysis_filter_status(self) -> None:
        dataset_text = "all datasets" if self.analysis_dataset_filter is None else f"{len(self.analysis_dataset_filter)} dataset(s)"
        scale_text = "all scales" if self.analysis_scale_filter is None else f"{len(self.analysis_scale_filter)} scale(s)"
        origami_text = "all origami" if self.analysis_origami_filter is None else f"{len(self.analysis_origami_filter)} origami"
        state_text = "all states" if self.analysis_state_filter is None else f"{len(self.analysis_state_filter)} state(s)"
        self.analysis_filter_status.set(f"Filters: {dataset_text}, {scale_text}, {origami_text}, {state_text}")

    def filtered_state_columns(self, state_columns: list[str]) -> list[str]:
        state_filter = getattr(self, "analysis_state_filter", None)
        if state_filter is None:
            return state_columns
        return [state for state in state_columns if state in state_filter]

    def read_filtered_count_rows_and_states(self, csv_path: Path, apply_state_filter: bool = True) -> tuple[list[dict[str, str]], list[str]]:
        if self.analysis_csv_path is not None and csv_path == self.analysis_csv_path:
            rows = [dict(row) for row in self.analysis_source_rows]
            fieldnames = list(self.analysis_fieldnames)
        else:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
        if not rows:
            raise ValueError("The selected CSV has no rows.")
        rows = self.sorted_count_rows(rows)
        rows = self.filtered_count_plot_rows(rows)
        if not rows:
            raise ValueError("No rows match the current plot filters.")

        state_columns = self.count_state_columns(fieldnames, rows)
        if apply_state_filter:
            state_columns = self.filtered_state_columns(state_columns)
        if not state_columns:
            raise ValueError("No state count columns match the current state filter.")
        return rows, state_columns

    def read_count_plot_data(self, csv_path: Path) -> tuple[list[dict[str, str]], list[str], str]:
        rows, state_columns = self.read_filtered_count_rows_and_states(csv_path)
        self.configure_analysis_state_selectors(state_columns)
        plot_rows = self.grouped_count_plot_rows(rows, state_columns)
        return plot_rows, state_columns, self.analysis_group_mode()

    def configure_analysis_state_selectors(self, state_columns: list[str]) -> None:
        if hasattr(self, "analysis_x_combo"):
            self.analysis_x_combo.configure(values=state_columns)
            self.analysis_y_combo.configure(values=state_columns)
        delta_options = state_columns
        if hasattr(self, "analysis_delta_combo"):
            self.analysis_delta_combo.configure(values=delta_options)
        for state in state_columns:
            self.analysis_state_labels.setdefault(state, state)
        if state_columns and self.analysis_x_state.get() not in state_columns:
            self.analysis_x_state.set(state_columns[0])
        if len(state_columns) > 1 and (self.analysis_y_state.get() not in state_columns or self.analysis_y_state.get() == self.analysis_x_state.get()):
            self.analysis_y_state.set(state_columns[1])
        elif len(state_columns) == 1:
            self.analysis_y_state.set(state_columns[0])
        if delta_options and self.analysis_delta_state.get() not in delta_options:
            self.analysis_delta_state.set(delta_options[0])

    def selected_scatter_states(self, state_columns: list[str]) -> tuple[str, str] | None:
        if len(state_columns) < 2:
            return None
        x_state = self.analysis_x_state.get()
        y_state = self.analysis_y_state.get()
        if x_state not in state_columns:
            x_state = state_columns[0]
        if y_state not in state_columns or y_state == x_state:
            y_state = next((state for state in state_columns if state != x_state), state_columns[1])
        return x_state, y_state

    def rename_analysis_state_labels(self) -> None:
        if self.analysis_csv_path is None or not self.analysis_source_rows:
            messagebox.showinfo("No analysis dataset", "Load an analysis folder first.")
            return
        state_columns = self.count_state_columns(self.analysis_fieldnames, self.analysis_source_rows)
        if not state_columns:
            messagebox.showinfo("No states", "No state columns were found.")
            return

        window = Toplevel(self.root)
        window.title("Rename State Labels")
        window.geometry("360x240")
        entries: dict[str, StringVar] = {}
        body = Frame(window)
        body.pack(fill=BOTH, expand=True, padx=10, pady=10)
        for state in state_columns:
            row = Frame(body)
            row.pack(fill="x", pady=4)
            Label(row, text=state, width=8, anchor="w").pack(side=LEFT)
            var = StringVar(value=self.state_display_label(state))
            entries[state] = var
            Entry(row, textvariable=var).pack(side=LEFT, fill="x", expand=True)

        controls = Frame(window)
        controls.pack(fill="x", padx=10, pady=(0, 10))

        def apply() -> None:
            for state, var in entries.items():
                value = var.get().strip()
                self.analysis_state_labels[state] = value or state
            window.destroy()
            if self.plot_previews:
                self.generate_analysis_plot_previews()

        Button(controls, text="Apply", command=apply).pack(side=RIGHT)

    def count_state_columns(self, fieldnames: list[str], rows: list[dict[str, str]]) -> list[str]:
        meta_columns = {"date_folder", "image", "path", "pixels_per_um", "total_detected", "scale", "total", "total_count", "dataset"}
        state_columns = []
        for field in fieldnames:
            if field in meta_columns or field.startswith("fraction_") or field.startswith("count_") or field.endswith("_ratio"):
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

    def annotate_analysis_rows(self, rows: list[dict[str, str]], folder: Path, label: str) -> list[dict[str, str]]:
        annotated = []
        for row in rows:
            item = dict(row)
            item["dataset"] = label
            item["_analysis_dataset"] = label
            item["_analysis_folder"] = str(folder)
            annotated.append(item)
        return annotated

    def prompt_analysis_dataset_name(self, folder: Path, prompt: str) -> str | None:
        name = simpledialog.askstring(
            "Dataset name",
            f"{prompt}\n\nFolder:\n{folder}\n\nName to use in plots and tables:",
            initialvalue=folder.name,
            parent=self.root,
        )
        if name is None:
            return None
        name = name.strip()
        return name or folder.name

    def safe_dataset_filename_part(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "dataset"

    def merged_fieldnames(self, fieldname_lists: list[list[str]]) -> list[str]:
        merged: list[str] = []
        for fieldnames in fieldname_lists:
            for field in fieldnames:
                if field not in merged:
                    merged.append(field)
        if "dataset" not in merged:
            merged.insert(0, "dataset")
        return merged

    def cleaned_dataset_fieldnames(self) -> list[str]:
        fields = [field for field in self.analysis_fieldnames if field and not field.startswith("_") and field != "dataset"]
        if fields:
            return fields
        state_columns = self.count_state_columns([], self.analysis_source_rows)
        return ["date_folder", "image", "path", "pixels_per_um", "total_detected", *state_columns]

    def result_folder_for_analysis_row(self, row: dict[str, str]) -> Path | None:
        image_path = self.classified_image_path_for_row(row)
        if image_path is not None:
            return image_path.parent
        if self.analysis_csv_path is None:
            return None
        image_name = row.get("image", "")
        if not image_name:
            return None
        stem = Path(image_name).stem
        candidate = self.analysis_csv_path.parent / stem
        return candidate if candidate.exists() and candidate.is_dir() else None

    def unique_cleaned_folder_path(self, parent: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = parent / f"all_images_cleaned_{timestamp}"
        suffix = 2
        while path.exists():
            path = parent / f"all_images_cleaned_{timestamp}_{suffix}"
            suffix += 1
        return path

    def save_cleaned_analysis_dataset(self) -> None:
        if self.analysis_csv_path is None or not self.analysis_source_rows:
            messagebox.showinfo("No analysis dataset", "Load and review an all_images folder first.")
            return
        initial_dir = self.analysis_csv_path.parent.parent if self.analysis_csv_path.parent.exists() else self.output_dir / CURRENT_RESULTS_DIR
        parent_dir = filedialog.askdirectory(initialdir=initial_dir, title="Choose where to save the cleaned all_images folder")
        if not parent_dir:
            return
        output_dir = self.unique_cleaned_folder_path(Path(parent_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        fieldnames = self.cleaned_dataset_fieldnames()
        clean_rows = []
        copied = 0
        missing_folders = []
        for row in self.sorted_count_rows([dict(row) for row in self.analysis_source_rows]):
            clean_row = {field: row.get(field, "") for field in fieldnames}
            clean_rows.append(clean_row)
            source_folder = self.result_folder_for_analysis_row(row)
            if source_folder is None:
                missing_folders.append(row.get("image", "unknown"))
                continue
            destination = output_dir / source_folder.name
            suffix = 2
            while destination.exists():
                destination = output_dir / f"{source_folder.name}_{suffix}"
                suffix += 1
            shutil.copytree(source_folder, destination)
            copied += 1

        summary_path = output_dir / "all_image_counts.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)

        self.analysis_status.set(f"Saved cleaned dataset with {len(clean_rows)} rows to {output_dir}.")
        message = f"Saved cleaned dataset to:\n{output_dir}\n\nRows: {len(clean_rows)}\nCopied image folders: {copied}"
        if missing_folders:
            message += f"\nMissing image folders: {len(missing_folders)}"
        messagebox.showinfo("Cleaned dataset saved", message)

    def load_analysis_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.output_dir / CURRENT_RESULTS_DIR)
        if not folder:
            return
        try:
            folder_path = Path(folder)
            rows, fieldnames, csv_path = self.read_counts_from_all_images_folder(folder_path)
            rows = self.annotate_analysis_rows(rows, folder_path, folder_path.name)
            fieldnames = self.merged_fieldnames([fieldnames])
            self.populate_analysis(rows, fieldnames, csv_path)
        except Exception as exc:
            messagebox.showerror("Analysis load failed", f"{exc}\n\n{traceback.format_exc()}")

    def load_analysis_comparison_folders(self) -> None:
        first = filedialog.askdirectory(initialdir=self.output_dir / CURRENT_RESULTS_DIR, title="Choose first all_images folder")
        if not first:
            return
        second = filedialog.askdirectory(initialdir=self.output_dir / CURRENT_RESULTS_DIR, title="Choose second all_images folder")
        if not second:
            return
        try:
            first_folder = Path(first)
            second_folder = Path(second)
            first_label = self.prompt_analysis_dataset_name(first_folder, "Name the first dataset")
            if first_label is None:
                return
            second_label = self.prompt_analysis_dataset_name(second_folder, "Name the second dataset")
            if second_label is None:
                return
            first_rows, first_fields, _first_csv = self.read_counts_from_all_images_folder(first_folder)
            second_rows, second_fields, _second_csv = self.read_counts_from_all_images_folder(second_folder)
            rows = [
                *self.annotate_analysis_rows(first_rows, first_folder, first_label),
                *self.annotate_analysis_rows(second_rows, second_folder, second_label),
            ]
            fieldnames = self.merged_fieldnames([first_fields, second_fields])
            comparison_csv = first_folder.parent / f"comparison_{self.safe_dataset_filename_part(first_label)}_vs_{self.safe_dataset_filename_part(second_label)}.csv"
            self.populate_analysis(rows, fieldnames, comparison_csv)
        except Exception as exc:
            messagebox.showerror("Comparison load failed", f"{exc}\n\n{traceback.format_exc()}")

    def ensure_analysis_row_ids(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            if "_analysis_id" not in row:
                row["_analysis_id"] = str(self.analysis_next_row_id)
                self.analysis_next_row_id += 1

    def populate_analysis(self, rows: list[dict[str, str]], fieldnames: list[str], csv_path: Path) -> None:
        rows = [dict(row) for row in rows]
        self.ensure_analysis_row_ids(rows)
        self.analysis_source_rows = rows
        self.analysis_fieldnames = list(fieldnames)
        self.configure_analysis_filter_options(rows)
        state_columns = self.count_state_columns(fieldnames, rows)
        if not state_columns:
            raise ValueError("No state count columns were found.")
        rows = self.sorted_count_rows(rows)
        self.configure_analysis_state_selectors(state_columns)
        self.analysis_csv_path = csv_path
        self.analysis_rows = []
        self.analysis_table_row_map = {}
        self.analysis_table.delete(*self.analysis_table.get_children())
        self.analysis_summary.delete(0, END)

        grouped: dict[str, dict[str, float]] = {}
        grouped_by_dataset: dict[str, dict[str, float]] = {}
        grouped_by_scale: dict[str, dict[str, float]] = {}
        for row in rows:
            image_name = row.get("image", "")
            dataset_label = row.get("dataset", row.get("_analysis_dataset", ""))
            origami_label = self.parse_origami_label(image_name)
            scale_label = self.count_row_scale_label(row)
            counts = {state: float(row.get(state, 0) or 0) for state in state_columns}
            total = sum(counts.values())
            fractions = {state: counts[state] / total if total else 0.0 for state in state_columns}
            fraction_text = ", ".join(f"{self.state_display_label(state)}: {fractions[state]:.3f}" for state in state_columns)
            item_id = f"analysis_row_{row['_analysis_id']}"
            self.analysis_table.insert(
                "",
                END,
                iid=item_id,
                values=("image", dataset_label, origami_label, scale_label, image_name, f"{total:.0f}", fraction_text),
            )
            self.analysis_table_row_map[item_id] = row
            metric_row: dict[str, str | float] = {
                "kind": "image",
                "dataset": dataset_label,
                "origami_label": origami_label,
                "scale": scale_label,
                "image": image_name,
                "path": row.get("path", ""),
                "total": total,
            }
            for state, count in counts.items():
                metric_row[f"count_{state}"] = count
                metric_row[f"fraction_{state}"] = fractions[state]
            for a_idx, a_state in enumerate(state_columns):
                for b_state in state_columns[a_idx + 1 :]:
                    ratio = counts[a_state] / counts[b_state] if counts[b_state] else float("inf") if counts[a_state] else 0.0
                    metric_row[f"{a_state}_to_{b_state}_ratio"] = ratio
            self.analysis_rows.append(metric_row)

            group = grouped.setdefault(origami_label, {"n_images": 0.0, "total": 0.0, **{state: 0.0 for state in state_columns}})
            group["n_images"] += 1
            group["total"] += total
            for state, count in counts.items():
                group[state] += count

            dataset_group = grouped_by_dataset.setdefault(dataset_label or "dataset", {"n_images": 0.0, "total": 0.0, **{state: 0.0 for state in state_columns}})
            dataset_group["n_images"] += 1
            dataset_group["total"] += total
            for state, count in counts.items():
                dataset_group[state] += count

            scale_group = grouped_by_scale.setdefault(scale_label, {"n_images": 0.0, "total": 0.0, **{state: 0.0 for state in state_columns}})
            scale_group["n_images"] += 1
            scale_group["total"] += total
            for state, count in counts.items():
                scale_group[state] += count

        self.analysis_table.insert("", END, values=("", "", "", "", "", "", ""))
        for dataset_label, group in sorted(grouped_by_dataset.items(), key=lambda item: item[0]):
            total = group["total"]
            fractions = {state: group[state] / total if total else 0.0 for state in state_columns}
            fraction_text = ", ".join(f"{self.state_display_label(state)}: {fractions[state]:.3f}" for state in state_columns)
            self.analysis_table.insert(
                "",
                END,
                values=("dataset group", dataset_label, "all", "all", f"{int(group['n_images'])} image(s)", f"{total:.0f}", fraction_text),
            )

        self.analysis_table.insert("", END, values=("", "", "", "", "", "", ""))
        for origami_label, group in sorted(grouped.items(), key=lambda item: self.origami_sort_value(f"origami{item[0]}_")):
            total = group["total"]
            fractions = {state: group[state] / total if total else 0.0 for state in state_columns}
            fraction_text = ", ".join(f"{self.state_display_label(state)}: {fractions[state]:.3f}" for state in state_columns)
            self.analysis_table.insert(
                "",
                END,
                values=("group", "all", origami_label, "all", f"{int(group['n_images'])} image(s)", f"{total:.0f}", fraction_text),
            )

        self.analysis_table.insert("", END, values=("", "", "", "", "", "", ""))
        for scale_label, group in sorted(grouped_by_scale.items(), key=lambda item: self.scale_sort_value_from_label(item[0])):
            total = group["total"]
            fractions = {state: group[state] / total if total else 0.0 for state in state_columns}
            fraction_text = ", ".join(f"{self.state_display_label(state)}: {fractions[state]:.3f}" for state in state_columns)
            self.analysis_table.insert(
                "",
                END,
                values=("scale group", "all", "all", scale_label, f"{int(group['n_images'])} image(s)", f"{total:.0f}", fraction_text),
            )

        metrics_path = csv_path.parent / "analysis_metrics.csv"
        metric_fields = sorted({key for metric_row in self.analysis_rows for key in metric_row.keys()})
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writeheader()
            writer.writerows(self.analysis_rows)

        self.analysis_summary.insert(END, f"Loaded: {csv_path.parent}")
        self.analysis_summary.insert(END, f"Images: {len(rows)}")
        dataset_labels = sorted({row.get("dataset", row.get("_analysis_dataset", "")) for row in rows if row.get("dataset", row.get("_analysis_dataset", ""))})
        if dataset_labels:
            self.analysis_summary.insert(END, f"Datasets: {', '.join(dataset_labels)}")
        sorted_group_labels = sorted(grouped.keys(), key=lambda label: self.origami_sort_value(f"origami{label}_"))
        self.analysis_summary.insert(END, f"Origami labels: {', '.join(sorted_group_labels)}")
        sorted_scale_labels = sorted(grouped_by_scale.keys(), key=self.scale_sort_value_from_label)
        self.analysis_summary.insert(END, f"Scales: {', '.join(sorted_scale_labels)}")
        self.analysis_summary.insert(END, f"States: {', '.join(self.state_display_label(state) for state in state_columns)}")
        scatter_states = self.selected_scatter_states(state_columns)
        if scatter_states is not None:
            self.analysis_summary.insert(END, f"Scatter metric: fraction {self.state_display_label(scatter_states[0])} vs fraction {self.state_display_label(scatter_states[1])}")
        self.analysis_summary.insert(END, f"Saved metrics: {metrics_path.name}")
        self.analysis_status.set(f"Loaded {len(rows)} image rows from {csv_path.name}.")
        self.refresh_analysis_review_list()

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

    def generate_dataset_overlay_plot_preview(self) -> None:
        if self.analysis_csv_path is None or not self.analysis_source_rows:
            messagebox.showinfo("No analysis folder loaded", "Load two all_images folders first.")
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rows = self.filtered_count_plot_rows([dict(row) for row in self.analysis_source_rows])
            if not rows:
                messagebox.showinfo("No matching data", "No rows match the current plot filters.")
                return
            state_columns = self.filtered_state_columns(self.count_state_columns(self.analysis_fieldnames, rows))
            if len(state_columns) < 2:
                messagebox.showinfo("Not enough states", "Choose at least two classification states for the overlay.")
                return
            self.configure_analysis_state_selectors(state_columns)

            dataset_labels: list[str] = []
            for row in rows:
                label = row.get("dataset", row.get("_analysis_dataset", ""))
                if label and label not in dataset_labels:
                    dataset_labels.append(label)
            if len(dataset_labels) != 2:
                messagebox.showinfo(
                    "Need two datasets",
                    f"Overlay requires exactly two datasets after filtering. Current filters include {len(dataset_labels)} dataset(s).",
                )
                return

            paired_rows = self.dataset_overlay_plot_rows(rows, state_columns, dataset_labels)
            if not paired_rows:
                messagebox.showinfo(
                    "No shared groups",
                    "No origami/scale groups are present in both datasets with the current filters.",
                )
                return

            scatter_states = self.plot_dataset_overlay(plt, paired_rows, state_columns, dataset_labels)
            if scatter_states is None:
                messagebox.showinfo("Not enough states", "Choose two different states for the overlay axes.")
                return

            x_state, y_state = scatter_states
            name = f"dataset_overlay_{x_state}_vs_{y_state}.png"
            self.plot_previews[name] = self.plot_image_from_current_figure()
            self.plot_list.delete(0, END)
            for plot_name in self.plot_previews:
                self.plot_list.insert(END, plot_name)
            index = list(self.plot_previews).index(name)
            self.plot_list.selection_clear(0, END)
            self.plot_list.selection_set(index)
            self.plot_list.see(index)
            self.show_plot_preview(name)
            self.analysis_status.set(f"Overlayed {len(paired_rows)} shared origami/scale group(s) from {dataset_labels[0]} to {dataset_labels[1]}.")
        except Exception as exc:
            messagebox.showerror("Dataset overlay failed", f"{exc}\n\n{traceback.format_exc()}")

    def on_analysis_scatter_select(self, _event=None) -> None:
        if self.analysis_csv_path is not None:
            self.generate_analysis_plot_previews()

    def on_analysis_delta_state_select(self, _event=None) -> None:
        if self.analysis_csv_path is not None:
            self.generate_analysis_plot_previews()

    def on_analysis_plot_group_select(self, _event=None) -> None:
        if self.analysis_csv_path is not None:
            self.generate_analysis_plot_previews()

    def on_analysis_plot_order_select(self, _event=None) -> None:
        if self.analysis_csv_path is not None:
            self.generate_analysis_plot_previews()

    def choose_analysis_scale_filter(self) -> None:
        self.choose_analysis_filter(
            title="Filter Scales",
            options=self.analysis_scale_options,
            selected=self.analysis_scale_filter,
            apply_callback=self.apply_analysis_scale_filter,
            display=lambda value: value,
        )

    def choose_analysis_dataset_filter(self) -> None:
        self.choose_analysis_filter(
            title="Filter Datasets",
            options=self.analysis_dataset_options,
            selected=self.analysis_dataset_filter,
            apply_callback=self.apply_analysis_dataset_filter,
            display=lambda value: value,
        )

    def choose_analysis_origami_filter(self) -> None:
        self.choose_analysis_filter(
            title="Filter Origami",
            options=self.analysis_origami_options,
            selected=self.analysis_origami_filter,
            apply_callback=self.apply_analysis_origami_filter,
            display=lambda value: f"origami{value}",
        )

    def choose_analysis_state_filter(self) -> None:
        self.choose_analysis_filter(
            title="Filter States",
            options=self.analysis_state_options,
            selected=self.analysis_state_filter,
            apply_callback=self.apply_analysis_state_filter,
            display=self.state_display_label,
        )

    def choose_analysis_filter(self, title: str, options: list[str], selected: set[str] | None, apply_callback, display) -> None:
        if not options:
            messagebox.showinfo(title, "Load an analysis folder first.")
            return
        window = Toplevel(self.root)
        window.title(title)
        window.geometry("320x420")

        frame = Frame(window)
        frame.pack(fill=BOTH, expand=True, padx=8, pady=8)
        listbox = Listbox(frame, selectmode="multiple", exportselection=False)
        listbox.pack(fill=BOTH, expand=True)
        for option in options:
            listbox.insert(END, display(option))
        active_selection = set(options) if selected is None else set(selected)
        for idx, option in enumerate(options):
            if option in active_selection:
                listbox.selection_set(idx)

        controls = Frame(window)
        controls.pack(fill="x", padx=8, pady=(0, 8))

        def select_all() -> None:
            listbox.selection_set(0, END)

        def clear_all() -> None:
            listbox.selection_clear(0, END)

        def apply() -> None:
            chosen = {options[idx] for idx in listbox.curselection()}
            apply_callback(chosen if chosen and len(chosen) < len(options) else None)
            window.destroy()

        Button(controls, text="All", command=select_all).pack(side=LEFT)
        Button(controls, text="None", command=clear_all).pack(side=LEFT, padx=6)
        Button(controls, text="Apply", command=apply).pack(side=RIGHT)

    def apply_analysis_scale_filter(self, selected: set[str] | None) -> None:
        self.analysis_scale_filter = selected
        self.update_analysis_filter_status()
        if self.analysis_csv_path is not None:
            self.generate_analysis_plot_previews()

    def apply_analysis_dataset_filter(self, selected: set[str] | None) -> None:
        self.analysis_dataset_filter = selected
        self.update_analysis_filter_status()
        if self.analysis_csv_path is not None:
            self.generate_analysis_plot_previews()

    def apply_analysis_origami_filter(self, selected: set[str] | None) -> None:
        self.analysis_origami_filter = selected
        self.update_analysis_filter_status()
        if self.analysis_csv_path is not None:
            self.generate_analysis_plot_previews()

    def apply_analysis_state_filter(self, selected: set[str] | None) -> None:
        self.analysis_state_filter = selected
        self.update_analysis_filter_status()
        if self.analysis_csv_path is not None:
            self.generate_analysis_plot_previews()

    def clear_analysis_plot_filters(self) -> None:
        self.analysis_dataset_filter = None
        self.analysis_scale_filter = None
        self.analysis_origami_filter = None
        self.analysis_state_filter = None
        self.update_analysis_filter_status()
        if self.analysis_csv_path is not None:
            self.generate_analysis_plot_previews()

    def on_analysis_table_double_click(self, event) -> None:
        item_id = self.analysis_table.identify_row(event.y)
        if not item_id:
            return
        row = self.analysis_table_row_map.get(item_id)
        if row is None:
            return
        image_path = self.classified_image_path_for_row(row)
        if image_path is None:
            messagebox.showinfo("Image not found", "Could not find an exported classified image for this row.")
            return
        self.show_analysis_image(image_path)

    def classified_image_path_for_row(self, row: dict[str, str]) -> Path | None:
        image_name = row.get("image", "")
        if not image_name:
            return None
        image_path = Path(image_name)
        stem = image_path.stem
        annotated_name = image_name if image_name.endswith("_annotated.png") else f"{stem}_annotated.png"
        candidates: list[Path] = []

        row_folder = row.get("_analysis_folder", "")
        if row_folder:
            base = Path(row_folder)
            candidates.append(base / stem / annotated_name)
            candidates.extend(base.rglob(annotated_name))
            candidates.append(base / image_name)

        if self.analysis_csv_path is not None:
            base = self.analysis_csv_path.parent
            candidates.append(base / stem / annotated_name)
            candidates.extend(base.rglob(annotated_name))
            candidates.append(base / image_name)

        path_text = row.get("path", "")
        if path_text:
            path = Path(path_text)
            candidates.append(path.with_name(annotated_name))
            candidates.append(path)

        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def show_analysis_image(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        window = Toplevel(self.root)
        window.title(str(image_path.name))
        window.geometry("1100x800")

        frame = Frame(window)
        frame.pack(fill=BOTH, expand=True)
        canvas = Canvas(frame, background="#f0f0f0", highlightthickness=0)
        y_scroll = Scrollbar(frame, orient="vertical", command=canvas.yview)
        x_scroll = Scrollbar(frame, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, image.width, image.height))
        window.analysis_image_photo = photo
        self.analysis_image_photo = photo

    def refresh_analysis_review_list(self, selected_id: str | None = None) -> None:
        if not hasattr(self, "analysis_review_list"):
            return
        selected_id = selected_id or self.analysis_review_selected_id
        self.analysis_review_rows = self.sorted_count_rows([dict(row) for row in self.analysis_source_rows])
        self.analysis_review_list.delete(0, END)
        selected_index = 0
        for idx, row in enumerate(self.analysis_review_rows):
            image_name = row.get("image", "")
            scale_label = self.count_row_scale_label(row)
            dataset_label = row.get("dataset", row.get("_analysis_dataset", ""))
            label = f"{dataset_label}  origami{self.parse_origami_label(image_name)}  {scale_label}  {image_name}"
            self.analysis_review_list.insert(END, label)
            if selected_id is not None and row.get("_analysis_id") == selected_id:
                selected_index = idx
        if self.analysis_review_rows:
            selected_index = min(selected_index, len(self.analysis_review_rows) - 1)
            self.analysis_review_list.selection_set(selected_index)
            self.analysis_review_list.see(selected_index)
            self.analysis_review_list.focus_set()
            self.show_analysis_review_row(self.analysis_review_rows[selected_index])
        else:
            self.analysis_review_selected_id = None
            self.analysis_review_image = None
            self.analysis_review_canvas.delete("all")
            self.analysis_review_canvas.create_text(16, 16, text="No images in this analysis dataset.", anchor="nw")
            self.analysis_review_canvas.configure(scrollregion=(0, 0, 400, 200))
            self.analysis_review_status.set("No images in this analysis dataset.")

    def clear_analysis_plot_previews(self) -> None:
        self.plot_previews = {}
        self.plot_list.delete(0, END)
        self.current_plot_preview_name = None
        self.render_plot_preview()

    def selected_analysis_review_index(self) -> int | None:
        selection = self.analysis_review_list.curselection()
        if not selection:
            return None
        index = selection[0]
        return index if 0 <= index < len(self.analysis_review_rows) else None

    def on_analysis_review_select(self, _event=None) -> None:
        index = self.selected_analysis_review_index()
        if index is not None:
            self.show_analysis_review_row(self.analysis_review_rows[index])

    def show_analysis_review_row(self, row: dict[str, str]) -> None:
        self.analysis_review_selected_id = row.get("_analysis_id")
        self.analysis_review_list.focus_set()
        self.analysis_review_canvas.delete("all")
        image_path = self.classified_image_path_for_row(row)
        if image_path is None:
            self.analysis_review_image = None
            self.analysis_review_canvas.create_text(16, 16, text="Classified image not found for this row.", anchor="nw")
            self.analysis_review_canvas.configure(scrollregion=(0, 0, 500, 200))
            self.analysis_review_status.set(row.get("image", "Image not found"))
            return
        self.analysis_review_image = Image.open(image_path).convert("RGB")
        self.analysis_review_fit_to_window = True
        self.render_analysis_review_image()
        index = self.selected_analysis_review_index()
        position = f"{index + 1}/{len(self.analysis_review_rows)}" if index is not None else f"1/{len(self.analysis_review_rows)}"
        self.analysis_review_status.set(f"{position}  {image_path.name}")

    def render_analysis_review_image(self) -> None:
        if not hasattr(self, "analysis_review_canvas") or self.analysis_review_image is None:
            return
        image = self.analysis_review_image
        if self.analysis_review_fit_to_window:
            canvas_w = max(self.analysis_review_canvas.winfo_width(), 100)
            canvas_h = max(self.analysis_review_canvas.winfo_height(), 100)
            self.analysis_review_zoom = min(canvas_w / image.width, canvas_h / image.height, 1.0)
        zoom = max(0.05, min(5.0, self.analysis_review_zoom))
        width = max(1, int(round(image.width * zoom)))
        height = max(1, int(round(image.height * zoom)))
        display = image.resize((width, height), Image.Resampling.LANCZOS)
        self.analysis_review_canvas.delete("all")
        self.analysis_review_photo = ImageTk.PhotoImage(display)
        self.analysis_review_canvas.create_image(0, 0, image=self.analysis_review_photo, anchor="nw")
        self.analysis_review_canvas.configure(scrollregion=(0, 0, width, height))

    def adjust_analysis_review_zoom(self, factor: float) -> None:
        self.analysis_review_fit_to_window = False
        self.analysis_review_zoom = max(0.05, min(5.0, self.analysis_review_zoom * factor))
        self.render_analysis_review_image()

    def fit_analysis_review_image(self) -> None:
        self.analysis_review_fit_to_window = True
        self.render_analysis_review_image()

    def review_shortcuts_active(self) -> bool:
        if not hasattr(self, "analysis_review_list"):
            return False
        focus = self.root.focus_get()
        if focus is not None:
            try:
                widget = focus
                while widget is not None:
                    if widget in {self.analysis_review_list, self.analysis_review_canvas}:
                        return True
                    widget = widget.master
            except Exception:
                pass
        return bool(self.analysis_review_rows)

    def previous_review_image_key(self, _event=None) -> str | None:
        if not self.review_shortcuts_active():
            return None
        self.select_previous_review_image()
        return "break"

    def next_review_image_key(self, _event=None) -> str | None:
        if not self.review_shortcuts_active():
            return None
        self.select_next_review_image()
        return "break"

    def select_previous_review_image(self) -> None:
        index = self.selected_analysis_review_index()
        if index is None:
            return
        new_index = max(0, index - 1)
        self.analysis_review_list.selection_clear(0, END)
        self.analysis_review_list.selection_set(new_index)
        self.analysis_review_list.see(new_index)
        self.show_analysis_review_row(self.analysis_review_rows[new_index])

    def select_next_review_image(self) -> None:
        index = self.selected_analysis_review_index()
        if index is None:
            return
        new_index = min(len(self.analysis_review_rows) - 1, index + 1)
        self.analysis_review_list.selection_clear(0, END)
        self.analysis_review_list.selection_set(new_index)
        self.analysis_review_list.see(new_index)
        self.show_analysis_review_row(self.analysis_review_rows[new_index])

    def delete_selected_review_image(self, _event=None) -> str:
        if not self.review_shortcuts_active():
            return "break"
        index = self.selected_analysis_review_index()
        if index is None:
            return "break"
        row = self.analysis_review_rows[index]
        next_selected_id = None
        if len(self.analysis_review_rows) > 1:
            next_index = min(index, len(self.analysis_review_rows) - 2)
            remaining_preview = [candidate for candidate in self.analysis_review_rows if candidate.get("_analysis_id") != row.get("_analysis_id")]
            next_selected_id = remaining_preview[next_index].get("_analysis_id")
        self.remove_analysis_rows_by_ids({row.get("_analysis_id")}, confirm=False, selected_review_id=next_selected_id)
        return "break"

    def remove_analysis_rows_by_ids(self, delete_ids: set[str | None], confirm: bool = True, selected_review_id: str | None = None) -> bool:
        delete_ids = {row_id for row_id in delete_ids if row_id is not None}
        if not delete_ids:
            return False
        if confirm and not messagebox.askyesno("Delete selected data", f"Remove {len(delete_ids)} selected image row(s) from this analysis dataset?"):
            return False

        remaining_rows = [row for row in self.analysis_source_rows if row.get("_analysis_id") not in delete_ids]
        if not remaining_rows:
            self.analysis_source_rows = []
            self.analysis_rows = []
            self.analysis_table_row_map = {}
            self.analysis_table.delete(*self.analysis_table.get_children())
            self.analysis_summary.delete(0, END)
            self.clear_analysis_plot_previews()
            self.refresh_analysis_review_list()
            self.analysis_status.set("Removed all image rows from this analysis dataset.")
            return True

        previous_review_id = self.analysis_review_selected_id
        self.analysis_review_selected_id = selected_review_id
        self.populate_analysis(remaining_rows, self.analysis_fieldnames, self.analysis_csv_path or self.workspace / "analysis_filtered_counts.csv")
        self.analysis_review_selected_id = selected_review_id or previous_review_id
        self.clear_analysis_plot_previews()
        self.analysis_status.set(f"Removed {len(delete_ids)} image row(s). Generate plot previews when ready.")
        return True

    def delete_selected_analysis_rows(self, _event=None) -> str:
        selected_items = list(self.analysis_table.selection())
        rows_to_delete = [self.analysis_table_row_map[item_id] for item_id in selected_items if item_id in self.analysis_table_row_map]
        if not rows_to_delete:
            return "break"
        self.remove_analysis_rows_by_ids({row.get("_analysis_id") for row in rows_to_delete}, confirm=True)
        return "break"

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
        self.current_plot_preview_name = name
        self.render_plot_preview()

    def render_plot_preview(self) -> None:
        if not hasattr(self, "plot_preview_canvas"):
            return
        self.plot_preview_canvas.delete("all")
        if not self.current_plot_preview_name:
            self.plot_preview_canvas.create_text(16, 16, text="Generate plot previews to view them here.", anchor="nw")
            self.plot_preview_canvas.configure(scrollregion=(0, 0, 400, 200))
            return
        image = self.plot_previews.get(self.current_plot_preview_name)
        if image is None:
            return
        zoom = max(0.25, min(5.0, self.plot_preview_zoom))
        width = max(1, int(round(image.width * zoom)))
        height = max(1, int(round(image.height * zoom)))
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        self.plot_preview_photo = ImageTk.PhotoImage(resized)
        self.plot_preview_canvas_image_id = self.plot_preview_canvas.create_image(0, 0, image=self.plot_preview_photo, anchor="nw")
        self.plot_preview_canvas.configure(scrollregion=(0, 0, width, height))
        self.plot_zoom_status.set(f"{int(round(zoom * 100))}%")

    def adjust_plot_preview_zoom(self, factor: float) -> None:
        self.plot_preview_zoom = max(0.25, min(5.0, self.plot_preview_zoom * factor))
        self.render_plot_preview()

    def reset_plot_preview_zoom(self) -> None:
        self.plot_preview_zoom = 1.0
        self.render_plot_preview()

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
