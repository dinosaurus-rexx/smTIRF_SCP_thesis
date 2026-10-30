# Data

| File | In the repo? | What it is |
|---|---|---|
| `coord.npy` | yes | (334, 2) int, annotated spot positions in **[y, x]** order |
| `class.npy` | yes | (334,) int: 0 = target molecule (243), 1 = off-target spot (91) |
| `vid_ch1.tiff` | **no** | the labelled movie, (501, 256, 512) uint16, 131 MB |

The movie is too large for GitHub (100 MB per-file limit), so it is not committed.
Every script reads it from `data/vid_ch1.tiff`.

If you have the raw two-channel recording (`vid.tiff`, shape (501, 2, 256, 512)),
extract the channel the annotations were drawn on:

```bash
python data/prepare_real_movie.py path/to/vid.tiff
```

Note the axis order: the labels are `[y, x]`, while SCP and the tuner work in `(x, y)`.
`Parameter_opt/SCP_parametrizer.py` (`load_test_labels`) does the swap.

To label a movie of your own, annotate it in the GUI (`GUI2/`), export, and convert:

```bash
python data/gui_to_labels.py export.json my_labels/   # -> my_labels/coord.npy, class.npy
```
