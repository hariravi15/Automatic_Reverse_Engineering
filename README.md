# Automatic Reverse Engineering of Parametric CAD Models

Official implementation for **"Automatic Reverse Engineering of Parametric CAD Models from Multi-View 2D Projections Using Deep Learning"**, published at the 36th CIRP Design Conference (CIRP Design 2026).

> Hariharan Ravichandran, David Inkermann — Institute of Mechanical Engineering, Technical University of Clausthal
> [Paper (Procedia CIRP 142, 2026, pp. 55–60)](https://doi.org/10.1016/j.procir.2026.05.223) · Open access, CC BY-NC-ND 4.0

This repository turns six orthographic 2D views of a physical part into an editable, dimensionally accurate parametric CAD model, built automatically inside Siemens NX. It combines a Multi-View CNN + Transformer that predicts the *shape* of the part (sketch/extrude/fillet operations) with a classical computer-vision measurement stage that recovers the *dimensions*, then merges both into a single command file that drives NX programmatically.
<img width="928" height="485" alt="image" src="https://github.com/user-attachments/assets/ef35b755-3efc-45ec-8516-665c2846817a" />
<p align="center">
  <br>
  <em>Fig. 1 from the paper — measurement script (solid line) and AI script (dashed line) run in parallel on images from the scanning station, and are merged into a dimensionally accurate command sequence for reconstruction in Siemens NX.</em>
</p>



## Table of Contents

- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Related repositories](#related-repositories)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Train the model](#1-train-the-model)
  - [2. Run inference on new views](#2-run-inference-on-new-views)
  - [3. Calibrate the measurement script](#3-calibrate-the-measurement-script)
  - [4. Measure the physical part](#4-measure-the-physical-part)
  - [5. Merge sequence + dimensions](#5-merge-sequence--dimensions)
  - [6. Generate and run the NX journal](#6-generate-and-run-the-nx-journal)
  - [End-to-end orchestration](#end-to-end-orchestration)
- [Command vocabulary](#command-vocabulary)
- [Results](#results)
- [Known limitations](#known-limitations)
- [Citation](#citation)
- [License](#license)

## How it works

The paper frames reverse engineering for remanufacturing as a three-stage problem, and the code mirrors that structure directly:

1. **Deep Learning Architecture** (`ML/`) — Six 256×256 grayscale orthographic views (top, bottom, left, right, front, back) are each passed through a shared ResNet backbone. The six per-view feature vectors are pooled (MVCNN-style) into a single 512-d shape vector, which conditions a Transformer decoder that autoregressively generates a token sequence describing sketch → extrude → fillet operations. Dimensions are emitted as placeholders, not real numbers — the network only has to get the *topology* right.
2. **Dimensional Calibration** (`Measurment/`) — A separate, purely classical CV pipeline (OpenCV contour analysis) measures the actual part from the same images, calibrated in pixels-per-mm against a known reference object (a sheet of paper for the top view, a sticker of known height for the side view).
3. **Automated CAD Reconstruction** (`Automation/`) — A merge step substitutes the measured real-world dimensions into the placeholder token sequence, converts the result into an [NX_easy](#related-repositories) Python script, and runs it inside Siemens NX to produce a final, fully editable `.prt` file.

The `full.py` orchestrator in `Automation/` stitches all three stages together end-to-end (see [End-to-end orchestration](#end-to-end-orchestration)).

## Repository structure

```
Automatic_Reverse_Engineering/
├── ML/
│   └── MV_unified_3.py       # MVCNN encoder + Transformer decoder: train & test entry point
├── Measurment/
│   ├── calibiration.py       # One-off script to compute PPM (pixels-per-mm) from a reference image
│   └── measurment.py         # Contour-based measurement of the physical part -> dimension JSON
├── Automation/
│   ├── merge_script.py       # Merges the AI's placeholder sequence with the measured dimensions
│   ├── NX.py                 # Converts the merged JSON into an NX_easy Python journal
│   ├── full.py                # Watches an input folder and orchestrates the whole pipeline as a .bat file
│   ├── Onshape.py             # Optional: same idea, targeting Onshape via the onpy API
│   └── Solidworks.py          # Optional: exports/converts an Onshape document towards SolidWorks
└── environment.yml            # Exported conda environment (Windows)
```

## Related repositories

This repo is one of three that together reproduce the paper's pipeline:

| Repository | Role |
|---|---|
| **Automatic_Reverse_Engineering** (this repo) | ML model, measurement script, and automation glue |
| [Dataset_Generation_for_Automatic_Reverse_Engineering](https://github.com/hariravi15/Dataset_Generation_for_Automatic_Reverse_Engineering) | Autodesk Fusion 360 scripts that generate the synthetic training set (cuboids, cylinders, L-clamps, fillets) used to train the model in Section 4.1 of the paper |
| [NX_easy](https://github.com/hariravi15/NX_easy) | The `nx_easy` Python wrapper around the Siemens NX API that `Automation/NX.py` imports (`nx.create_plane`, `nx.sketch_circle`, `nx.extrude`, `nx.export`, ...); ships with its own PDF documentation |

**To reproduce the full pipeline from scratch you need all three.** Clone `NX_easy` somewhere on your `PYTHONPATH` (or into `Automation/`) before running `NX.py`, since it does `import nx_easy as nx`. Clone `Dataset_Generation_for_Automatic_Reverse_Engineering` only if you want to regenerate the synthetic training data yourself rather than training on your own dataset.

## Installation

Requires **Python 3.12**, **Siemens NX** (tested on NX 1980, via `run_journal.exe`) for the reconstruction step, and a CUDA-capable GPU if you intend to train.

```bash
git clone https://github.com/hariravi15/Automatic_Reverse_Engineering.git
cd Automatic_Reverse_Engineering
conda env create -f environment.yml
conda activate Reverse_engineering
```

`environment.yml` is a direct `conda env export` from the development machine, so it's Windows-specific and includes some packages unrelated to this project (e.g. `anomalib`, `shap`, `freia`) left over from the wider research environment. If you're setting up on Linux/macOS or want a lean install, the packages that actually matter are:

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib scikit-learn pillow
```

Then add [`NX_easy`](https://github.com/hariravi15/NX_easy) to your path so `Automation/NX.py` can import it:

```bash
git clone https://github.com/hariravi15/NX_easy.git
# then either drop it inside Automation/, or add its parent folder to PYTHONPATH
```

## Usage

Every script in this repo currently hardcodes its input/output folders near the top of the file as absolute Windows paths (e.g. `D:\Dataset\...`, `D:\Automtion\...`) — a holdover from the original research setup rather than a configured default. **Before running anything, open the script and update those paths for your machine.** Where a script also accepts CLI flags, they're listed below and override the folder scan behavior.

### 1. Train the model

```bash
python ML/MV_unified_3.py --mode train
```

Reads `Config.DATA_ROOT_DIR` (a folder of `<model_id>/{top,bottom,left,right,front,back}.png`), `Config.JSON_DIR` (ground-truth procedural JSON per model), and `Config.DATASET_SPLIT_JSON_PATH` (a `{"train_ids": [...], "val_ids": [...]}` file). Builds a token vocabulary from the training JSONs, then trains the MVCNN + Transformer with early stopping. Saves the vocabulary (`VOCAB_SAVE_PATH`), best checkpoint (`MODEL_SAVE_PATH`), and a training/validation loss curve (`PLOT_SAVE_PATH`). All of these paths, plus batch size, learning rate, epoch count, and early-stopping patience, are set on the `Config` class at the top of the file.

### 2. Run inference on new views

```bash
python ML/MV_unified_3.py --mode test --test_dir path/to/six_views_folder
```

`test_dir` should contain `top.png`, `bottom.png`, `left.png`, `right.png`, `front.png`, `back.png` for a single part. Loads the saved vocabulary and checkpoint, runs beam search (width 5) to generate a token sequence, and writes `generated_output_for_<folder_name>.json` — the placeholder-valued command sequence that Stage 3 will later fill with real dimensions.

> **Note:** `tokens_to_json_script()` in `MV_unified_3.py` is currently a stub — it returns the raw token list rather than a fully structured JSON. If you're using the automation scripts downstream (`merge_script.py` expects a `generated_tokens` key), that part already works; if you need a richer structured JSON straight out of inference, you'll need to flesh out this function.

### 3. Calibrate the measurement script

Run once per camera setup, whenever your scanning station's camera positions or focal distance change:

```bash
python Measurment/calibiration.py
```

Photograph a reference sheet of paper (used to calibrate the top-view pixels-per-mm) and a sticker of known height (used to calibrate the side-view pixels-per-mm) in the same imaging conditions as your real captures, set their paths and known dimensions (`PAPER_WIDTH_MM`, `STICKER_HEIGHT_MM`) at the top of the script, and run it. Copy the two printed `PPM_TOP` / `PPM_SIDE` values into `Measurment/measurment.py`.

### 4. Measure the physical part

```bash
python Measurment/measurment.py --top path/to/top.png --side path/to/side.png --output path/to/output_folder
```

Classifies the part's top-view silhouette as `cylinder`, `cuboid`, or `l_clamp` from its contour, extracts dimensions (outer/inner diameter and bolt-hole positions for cylinders; length/width for cuboids; length/width/thickness for L-clamps), reads height from the side view, and writes both a dimension JSON and an annotated debug image to `--output`.

Omit the CLI flags and it falls back to batch mode: it scans `INPUT_FOLDER` (edit at the top of the script) for any file with `top` in its name, looks for a matching `side`/`left` file, and processes every pair it finds.

### 5. Merge sequence + dimensions

```bash
python Automation/merge_script.py
```

Takes the most recently modified JSON from `ML_SEQUENCE_FOLDER` (Step 2's output) and from `DIMENSION_FOLDER` (Step 4's output), substitutes the measured values into the placeholder token sequence, and writes the merged result to `OUTPUT_FOLDER`. With `TIME_INTERVAL = 0` (the default) it runs once and exits; set it above `0` to have it poll continuously as a watch-folder service instead.

### 6. Generate and run the NX journal

```bash
python Automation/NX.py
```

Converts the merged token sequence into a Python journal that calls into [`nx_easy`](https://github.com/hariravi15/NX_easy) (`create_plane`, `sketch_circle`, `sketch_line`, `extrude`, `export`). Requires `NX_easy` to be importable — see [Related repositories](#related-repositories). Run the generated `*_nx_journal.py` file inside Siemens NX (`run_journal.exe path\to\generated_script.py`) to produce the final `.prt`.

### End-to-end orchestration

`Automation/full.py` runs all of the above automatically: it watches an input directory for new folders of scanned views, and for each one, generates and launches a `.bat` file that runs the ML prediction → measurement → merge → NX journal generation → NX execution steps in sequence (with optional Onshape/SolidWorks export stages appended). Update every path constant at the top of `full.py` (script locations, working folders, your `run_journal.exe` path, conda environment name) before use — like the other scripts, these are hardcoded to the original development machine.

## Command vocabulary

The token sequence the decoder generates follows the grammar below (Table 1 in the paper):

| Command | Parameter |
|---|---|
| `<sos>` | ∅ (start of sequence) |
| `ENTITY_START` | type: `Sketch`, `Extrude`, `Fillet` |
| `plane` | plane_id: `XY` |
| `operation_type` | op_id: `NewBody`, `Join`, `Cut` |
| `Dimension` | placeholder, resolved later from measurement |
| `ENTITY_END` | type: `Sketch`, `Extrude`, `Fillet` |
| `<eos>` | ∅ (end of sequence) |

## Results

Evaluated on a held-out synthetic test set (1,000 models) plus 15 real-world physical components scanned on a custom turntable station with three Intel RealSense cameras, against the single-view [CAD-Coder](https://doi.org/10.48550/arXiv.2505.14646) baseline:

| Method | Input | Structural Accuracy | Dimensional Accuracy |
|---|---|---|---|
| CAD-Coder | Single-view | 65% | 10% |
| **MV2CAD (ours)** | **Multi-view** | **90%** | **85%** |

*Structural accuracy* = percentage of test samples whose generated topology (faces, edges, connectivity) exactly matches the ground-truth command sequence. *Dimensional accuracy* = percentage of dimensions within 5 mm of ground truth. See Section 4 of the paper for full experimental details.

## Known limitations

Carried over from the paper's discussion (Section 5), plus a couple of implementation notes:

- Capturing a true bottom view is often impractical on a physical rig; the current pipeline substitutes the top view for it, which can hurt accuracy on asymmetrical parts.
- The measurement script is sensitive to lighting artifacts and part misalignment relative to the camera; current dimensional accuracy is bounded at roughly ±6 mm.
- Supported geometry is currently limited to sketch/extrude/fillet on the XY plane — no sweep, loft, or multi-plane sketching yet.
- The reconstruction step targets Siemens NX specifically; the Onshape/SolidWorks scripts in `Automation/` are earlier, less-validated alternates rather than part of the paper's evaluated pipeline.
- `tokens_to_json_script()` in `ML/MV_unified_3.py` is a stub (see [Usage → step 2](#2-run-inference-on-new-views)).

## Citation

If you use this code, please cite the paper:

```bibtex
@article{ravichandran2026automatic,
  title   = {Automatic Reverse Engineering of Parametric CAD Models from Multi-View 2D Projections Using Deep Learning},
  author  = {Ravichandran, Hariharan and Inkermann, David},
  journal = {Procedia CIRP},
  volume  = {142},
  pages   = {55--60},
  year    = {2026},
  doi     = {10.1016/j.procir.2026.05.223},
  note    = {36th CIRP Design Conference (CIRP Design 2026)}
}
```

## License

Add a license file if you intend for others to reuse or build on this code — the repository doesn't currently include one. The paper itself is CC BY-NC-ND 4.0, but that covers the *article*, not automatically the *code*; pick a license for the software separately (e.g. MIT, Apache-2.0) and add a `LICENSE` file at the repo root.
