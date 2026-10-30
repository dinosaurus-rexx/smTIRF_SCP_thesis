# SCP: single-molecule spot detection for smTIRF movies

Code for the master's thesis of Nahuel Jiménez Pol.

**SCP** finds single molecules in single-molecule TIRF (smTIRF) movies. It detects spots in
every frame, links them over time into tracks, and keeps a track as a molecule only if it is
round, stays in place, flickers in brightness, and is seen in enough frames. This repository
holds the detector, the tool that tunes its parameters on a labelled movie, a web GUI for
labelling movies, and the benchmark against Papylio and DeepFRET.

## Layout

| folder | what it is |
|---|---|
| `SCP/` | the detector, `SCP_KTree_JSON.py` (command line or importable class) |
| `Parameter_opt/` | `SCP_parametrizer.py`: tunes SCP's parameters on the labelled movie with Optuna |
| `GUI2/` | Flask web app for labelling spots in a movie as positive / negative |
| `benchmark/` | scores SCP, Papylio and DeepFRET with one scorer and draws the thesis figures |
| `data/` | the labels of the real movie (the movie itself is not committed, see `data/README.md`) |

## Setup

Tested with Python 3.12.

```bash
pip install -r requirements.txt
```

Then put the labelled movie at `data/vid_ch1.tiff` (see `data/README.md`). The GUI and the
committed benchmark arrays work without it; running SCP, tuning, and regenerating SCP's
benchmark arrays need it.

## How SCP works

1. **Preprocess**: light Gaussian smoothing; subtract a broad background estimate
   (σ = 8 px) and clip at zero.
2. **Find spots**: Laplacian-of-Gaussian blob detection on each frame.
3. **Measure each spot**: brightness, and roundness (4π·area / perimeter² of the spot's outline).
4. **Track**: link spots frame to frame with a KD-tree; a spot joins the nearest free track
   within `max_displacement` px; a track survives gaps of up to 3 frames.
5. **Merge**: drop 1-spot tracks; glue tracks whose average positions are within 1 px
   (friend-of-a-friend); pool their spots and keep one spot per frame.
6. **Classify**: keep a merged track if it has at least `min_pulsation_frames` spots, mean
   roundness ≥ `circularity_threshold`, largest frame-to-frame jump ≤ 2 × `max_displacement`,
   and brightness CV ≥ `intensity_variation_threshold`.
7. **Remove duplicates**: of events within 3 px of each other, keep the longest.

## Usage

All commands run from the repository root.

**Run SCP** on a movie (writes the events to JSON):

```bash
python SCP/SCP_KTree_JSON.py data/vid_ch1.tiff --output_json events.json --no_plot \
    --min_sigma 0.4515 --max_sigma 1.2690 --threshold 0.3234 \
    --circularity_threshold 0.2021 --max_displacement 4.9934 \
    --min_pulsation_frames 20 --intensity_variation_threshold 0.0718 \
    --bandpass_bg_sigma 8 --event_nms_radius 3
```

The command-line defaults are not tuned for this movie, so always pass the parameters.
Note that `--event_nms_radius` defaults to 0 (off).

**Tune the parameters** on the labelled movie:

```bash
python Parameter_opt/SCP_parametrizer.py
```

The movie is cut into three vertical strips: train (x 0–170), validation (x 170–256) and
test (x 256–512). Optuna searches on train, the best 8 trials are re-picked on validation, and
the chosen trial's test score is reported. 60 trials take about 2.5 h; the result goes to
`Parameter_opt/runs/optimization_results.json`.

**Benchmark** against Papylio and DeepFRET, scored on the same held-out test strip
(x 256–512) by default: see `benchmark/README.md`.

```bash
python benchmark/make_scp_arrays.py
python benchmark/unified_benchmark.py
python benchmark/plot_unified.py
```

**Label a movie** in the browser:

```bash
cd GUI2
python smTIRF_tool.py        # then open http://localhost:5000
```

Upload a TIFF, click spots as positive or negative while stepping through frames, and export.
`data/gui_to_labels.py` converts the export into the `coord.npy` / `class.npy` format.

## Data and axis conventions

- Labels (`data/coord.npy`) are `[y, x]`; SCP, its JSON output and the tuner use `(x, y)`.
- Classes (`data/class.npy`): 0 = target molecule (243), 1 = off-target spot (91).
- All results come from one 501-frame recording. The track-length threshold in particular is
  an absolute frame count of this movie and does not transfer to recordings of different
  length or kinetics.
