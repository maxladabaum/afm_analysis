# DNA origami AFM counting GUI

This folder now includes an interactive desktop tool for labeling DNA origami states, training a classifier, and batch-counting the number of origami in AFM image files.

## Setup

macOS:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

macOS:

Double-click `run_origami_counter.command` in Finder, or run:

```bash
./run_origami_counter.command
```

Windows:

Double-click `run_origami_counter.bat`, or run:

```powershell
.\.venv\Scripts\python origami_counter_app.py
```

## Workflow

1. Click **Open Root** and choose the folder that contains your dated folders.
2. Select an AFM image from the list. File names do not need to start with `origami`; supported rendered image formats are PNG, JPG, JPEG, TIF, and TIFF.
3. Check the scale status for the selected image. When the matching `.spm` file is present, calibration uses the `.spm` scan size metadata, not the PNG scale bar.
4. Optional fallback: click **Box Scale** and drag across the printed scale bar only if an image does not have a usable matching `.spm` file.
5. Click **Box Pixel Area**, drag a box around one representative origami, and release. This sets the pixel-area detection range automatically and runs detection.
6. Click **Detect** again if you adjust the pixel area, threshold, or scale setting.
7. Add or edit state names in the state panel.
8. Select a state, then click detected origami boxes in the image to label examples.
9. Click **Add Current Image Labels to Training Data** when you want the current labels to be used for training.
10. Click **Train Current Image** to train only from the selected image's committed training labels, or **Train All Labeled Images** to train from every image in the training data list.
11. Click **Classify Image** to count the current image, or **Batch Count** to process all images.

Click **Detect + Export All Images** if you only want particle counts and do not need state classification. This opens a review window for every image, lets you rerun detection with adjusted pixel-area/bias settings, and lets you save or skip each image. Approved detections are saved with original images, `*_detected.png` annotated detection images, and count CSVs under `analysis_output/classified_images/detect_all_images_<timestamp>/`.

Instead of entering pixel-area limits directly, you can enter a physical target size in **Target size nm**, such as `100x150`. The app converts that size to area using each image's scale metadata and applies the **Size range multiple**, for example `0.35-3.0`, to compute per-image pixel-area limits.

The detection size fields synchronize when you press Enter or leave a field. Editing **Target size nm** or **Size range multiple** updates the min/max pixel area fields. Editing min/max pixel area updates **Target size nm** as a square-equivalent size, because an area alone cannot determine separate width and height.

The classifier uses scale-normalized size features from the `.spm` scan size when available, so labels from one zoom level can be applied to any other zoom level.

## Polymer persistence length

For polymer-like origami contours, open a folder such as `polymer_example`, switch to the **Polymers** tab, select the AFM image, and use **Analyze Polymers**. The analysis follows the AFM contour workflow described by Lee et al., ACS Nano 2019: threshold the height image, skeletonize unbranched contours, sample each contour at a fixed segment length, and fit the 2D worm-like-chain mean-square end-to-end relation

```text
<R^2> = 4 Lp lc [1 - (2 Lp / lc) (1 - exp(-lc / (2 Lp)))]
```

Use **Min length nm** to reject short debris and **Segment nm** to set the contour sampling interval. The tab can preview the original image, transparent contour overlay, or contours only; adjust **Overlay opacity** to make the detected traces lighter or stronger. It also shows a plot of measured mean-square end-to-end distance against contour separation with the fitted 2D WLC curve used to calculate persistence length. The main polymer view is dynamic: **Show Contours + Fit** displays draggable contour and WLC-fit panes, while **Preview Figure 2b Plot** or **Preview Figure C Plot** switches the right side to a full-height figure preview. Figure 2b shows the AFM image plus detected contours and the same contours translated to a common origin with initial tangents aligned horizontally. Figure C renders paired tiles where each accepted polymer has a cropped AFM image next to its extracted contour. Scale calibration comes from the paired `.spm` scan-size metadata when available; otherwise use **Box Scale** first. **Export Polymer Results** writes a summary CSV, per-contour CSV, mean-square end-to-end CSV, original image, annotated contour image, WLC fit plot, Figure 2b-style contour plot, and Figure C-style crop/contour plot under `analysis_output/polymer_persistence/`.

If your images are still raw Nanoscope `.spm` files, click **Import SPM Folder** and choose the folder containing them. Raw `.spm` file names do not need to start with `origami`. The app reads the height channel, applies plane and line flattening, and opens a per-image import preview where you can tune the z-height display range for each image. Click **Import Current + Next** only after the current image looks right; the app writes that image immediately, then advances to the next one. Click **Skip File** for images that should not be imported. Converted PNGs use NanoScope Color Table 12 with a z-height colorbar and scale bar, are saved to `converted_images/spm_png_<timestamp>/`, and keep each source `.spm` beside its PNG for scan-size metadata. The importer ignores existing `analysis_output/` and `converted_images/` folders, so previous imports are not previewed again as duplicate raw files. The default z range is based on the inner quartile, so isolated high spots can saturate without darkening the entire image.

The **Training Data Images** panel shows every image that has been explicitly added to training data and the number of committed labels for that image. Double-click a row, or select it and click **Open**, to jump to that image. Select a row and click **Clear** to remove that image from the training data list.

Use **Export Classifier** after training to save a portable `.joblib` model. On a later day, start the app, click **Load Classifier**, choose that `.joblib` file, then detect/classify new images.

## Export Results and Plots

After classifying the current image, click **Export Current Results** to save:

- a one-row counts CSV
- a copy of the original PNG
- an annotated PNG with colored classification boxes

These are written under `analysis_output/classified_images/`.

Click **Classify + Export All Images** to run the loaded classifier on every image in the current root folder. This writes an `all_images_<timestamp>/` folder containing one result folder per image plus `all_image_counts.csv` with all counts in one table.

Click **Plot Counts CSV** to load a saved counts CSV, including a single-image export or `origami_counts.csv` from **Batch Count**. The app writes plots and a fraction metrics CSV under `analysis_output/plots/`, including count stacks, fraction stacks, total counts, and a first-vs-second-state fraction scatter.

The **Analysis** tab can load an `all_images_<timestamp>` folder from **Classify + Export All Images**. It reads the combined counts CSV, extracts the origami label from names such as `origami1_` and `origami12f_`, shows per-image and grouped metrics, and saves `analysis_metrics.csv` in the loaded folder. Use **Generate Plot Previews** to view plots in the GUI, then save only the selected plot or all previewed plots.

## Bootstrap training

You can iteratively improve a classifier on one image:

1. Detect origami in an image.
2. Label a small number of examples from each state.
3. Click **Add Current Image Labels to Training Data**.
4. Click **Train Current Image**.
5. Click **Classify Image**. Predictions are saved as editable labels for that image, but not added to training data yet.
6. Correct any wrong labels by selecting the correct state and clicking the misclassified origami.
7. Click **Add Current Image Labels to Training Data** again to commit the corrected labels.
8. Click **Train Current Image** again and repeat.

The same loop works on a new image: classify it, correct mistakes, then train again from the current image or all labeled images.

Labels, model files, and count outputs are written to `analysis_output/`.
