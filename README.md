# DNA origami AFM counting GUI

This folder now includes an interactive desktop tool for labeling DNA origami states, training a classifier, and batch-counting the number of origami in each AFM PNG image.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python origami_counter_app.py
```

## Workflow

1. Click **Open Root** and choose the folder that contains your dated folders.
2. Select an `origami...png` image from the list.
3. Check the scale status for the selected image. When the matching `.spm` file is present, calibration uses the `.spm` scan size metadata, not the PNG scale bar.
4. Optional fallback: click **Box Scale** and drag across the printed scale bar only if an image does not have a usable matching `.spm` file.
5. Click **Box Area**, drag a box around one representative origami, and release. This sets the physical area range automatically and runs detection.
6. Click **Detect** again if you adjust the area, threshold, or scale setting.
7. Add or edit state names in the state panel.
8. Select a state, then click detected origami boxes in the image to label examples.
9. Click **Add Current Image Labels to Training Data** when you want the current labels to be used for training.
10. Click **Train Current Image** to train only from the selected image's committed training labels, or **Train All Labeled Images** to train from every image in the training data list.
11. Click **Classify Image** to count the current image, or **Batch Count** to process all images.

The classifier uses scale-normalized size features from the `.spm` scan size when available, so labels from one zoom level can be applied to any other zoom level.

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
