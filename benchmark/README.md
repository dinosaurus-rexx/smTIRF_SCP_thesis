# Benchmark: SCP vs Papylio vs DeepFRET

Scores the three detectors on the labelled movie with one scorer and draws the comparison
figures used in the thesis.

## Region: the held-out test strip

By default everything is scored on the **test strip, x 256–512** (111 target and 42
off-target annotations), the same strip the tuner holds out. Papylio and DeepFRET ran with
default or pretrained settings.

The whole movie is available as a secondary view (`--band 0 512`, written to `output_x0_512/`).

## Scorer

`unified_benchmark.py` uses the scorer from `Parameter_opt/SCP_parametrizer.py`:

- greedy 1:1 assignment, closest pair first
- a detection matches an annotation when |dx| ≤ 3 **and** |dy| ≤ 3 px
- every detection not matched to a **target** annotation is a false positive, including
  detections on annotated off-target spots
- recall = TP / targets in the region, precision = TP / detections in the region,
  F1 = harmonic mean

Each algorithm is scored at two stages: **detection** (everything it found) and
**selection** (what it kept after its own filtering).

## Inputs

`inputs/{alg}/{detections,selections}.npy`, each (N, 2) in `[y, x]` order, covering the
whole movie (the region is applied at scoring time).

| algorithm | produced by |
|---|---|
| DeepFRET, Papylio | the Snakemake benchmark pipeline, [github.com/Dinand-K/BachelorThesis-FRETAlgorithms](https://github.com/Dinand-K/BachelorThesis-FRETAlgorithms), run on the same movie |
| SCP | the same pipeline's SCP wrapper (detections = mean position of every merged track; selections = SCP's valid events). `make_scp_arrays.py` writes the same two arrays |

Note: the SCP arrays in `inputs/scp/` were made with an older version of the track merging.
Running `make_scp_arrays.py` with the current code gives whole-movie selection F1 0.833
(R 88.1%, P 79.0%) instead of 0.854.

## Reproduce

From the repository root, with `data/vid_ch1.tiff` in place (see `data/README.md`):

```bash
python benchmark/make_scp_arrays.py                       # SCP -> benchmark/inputs/scp/  (~3-5 min)
python benchmark/unified_benchmark.py                     # test strip -> benchmark/output/
python benchmark/plot_unified.py                          # figures -> benchmark/output/figures/

python benchmark/unified_benchmark.py --band 0 512         # whole movie -> benchmark/output_x0_512/
python benchmark/plot_unified.py --out benchmark/output_x0_512
```

Only the first step needs the movie; the others run from the committed arrays.

## Recorded result

**Test strip x 256–512** (`output/summary.csv`), 111 target and 42 off-target annotations:

| algorithm | stage | detections | targets found | off-target | pure noise | recall | precision | F1 |
|---|---|---|---|---|---|---|---|---|
| DeepFRET | detection | 188 | 85 | 25 | 78 | 76.6% | 45.2% | 0.569 |
| DeepFRET | selection | 115 | 72 | 17 | 26 | 64.9% | 62.6% | **0.637** |
| Papylio | detection | 145 | 102 | 32 | 11 | 91.9% | 70.3% | 0.797 |
| Papylio | selection | 79 | 70 | 6 | 3 | 63.1% | 88.6% | **0.737** |
| SCP | detection | 912 | 111 | 38 | 763 | 100.0% | 12.2% | 0.217 |
| SCP | selection | 135 | 104 | 22 | 9 | 93.7% | 77.0% | **0.846** |

Whole movie (`output_x0_512/summary.csv`), 243 target and 91 off-target annotations,
selection stage: SCP **0.854** (R 92.6%, P 79.2%), Papylio **0.803** (R 71.2%, P 92.0%),
DeepFRET **0.692** (R 70.0%, P 68.5%).

Figures used in the thesis: `output/figures/detection_selection_bars_unified_no_noise.png` and
`output/figures/recall_precision_plot_unified.png`.

## Files

| file | what it is |
|---|---|
| `make_scp_arrays.py` | runs SCP on the movie and writes `inputs/scp/*.npy` (+ the parameters used) |
| `unified_benchmark.py` | per-class and global metrics → `output/*.csv`; `--band XMIN XMAX` for another strip |
| `plot_unified.py` | the figures, from those CSVs; `--out DIR` for another strip's directory |
| `output/` | test strip x 256–512: `summary.csv`, per-class and global CSVs, `run_metadata.json`, `figures/` |
| `output_x0_512/` | the same for the whole movie |
