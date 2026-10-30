"""
Run SCP on the movie and save the benchmark arrays.

    inputs/scp/detections.npy  mean position of every merged track
    inputs/scp/selections.npy  SCP's valid events

Both are (N, 2) in [y, x].

    python benchmark/make_scp_arrays.py
    python benchmark/make_scp_arrays.py --params Parameter_opt/runs/optimization_results.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# UTF-8 output (SCP prints '→', which fails on Windows otherwise)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "SCP"))
sys.path.insert(0, str(PROJ / "Parameter_opt"))

from SCP_KTree_JSON import smTIRFEventDetector  # noqa: E402
import SCP_parametrizer as pm  # noqa: E402

# Parameters used for the benchmark (tuning trial 3, min_pulsation_frames set to 20)
PARAMS = {
    'min_sigma': 0.4515457399879006,
    'max_sigma': 1.2689848283173364,
    'intensity_threshold': 0.32336429282837625,
    'circular_threshold': 0.20209149318019778,
    'max_displacement': 4.9934159843646295,
    'min_pulsation_frames': 20,
    'intensity_variation_threshold': 0.07179385564931576,
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--params", type=Path, default=None,
                    help="tuning result JSON; its final_parameters replace the defaults")
    ap.add_argument("--out", type=Path, default=HERE / "inputs" / "scp",
                    help="output directory (default: inputs/scp)")
    return ap.parse_args()


def yx(points_xy):
    """(x, y) list -> (N, 2) array in [y, x]"""
    if not points_xy:
        return np.empty((0, 2))
    return np.array([(p[1], p[0]) for p in points_xy])


def main():
    args = parse_args()
    params = dict(PARAMS)
    if args.params:
        params.update(json.loads(args.params.read_text())["final_parameters"])
    print(f"parameters: {params}")

    frames = tifffile.imread(pm.TEST_TIF).astype(np.float32)
    print(f"loaded {pm.TEST_TIF}: {frames.shape}")

    detector = smTIRFEventDetector(
        min_sigma=params['min_sigma'],
        max_sigma=params['max_sigma'],
        threshold=params['intensity_threshold'],
        circularity_threshold=params['circular_threshold'],
        max_displacement=params['max_displacement'],
        min_pulsation_frames=params['min_pulsation_frames'],
        intensity_variation_threshold=params['intensity_variation_threshold'],
        bandpass_bg_sigma=pm.BANDPASS_BG_SIGMA,
        event_nms_radius=pm.EVENT_NMS_RADIUS,
        intensity_aperture=pm.INTENSITY_APERTURE,
    )
    results = detector.process_video(frames)

    # all_tracks is before merging, so merge again here
    merged = detector._merge_overlapping_tracks_optimized(
        results['all_tracks'], position_threshold=detector.track_merge_dist)
    detections = yx([np.mean([s['position'] for s in t['spots'].values()], axis=0)
                     for t in merged.values()])
    selections = yx([e['mean_position'] for e in results['valid_events']])

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "detections.npy", detections)
    np.save(args.out / "selections.npy", selections)
    (args.out / "parameters.json").write_text(json.dumps(params, indent=2))
    print(f"detections (merged tracks): {len(detections)}")
    print(f"selections (valid events):  {len(selections)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
