"""
Score SCP, Papylio and DeepFRET with the same scorer used for tuning.

Writes per-class and global CSVs to output/ (same format as the Snakemake
pipeline) for plot_unified.py. Default region is the test strip x 256-512;
use --band XMIN XMAX for another strip (written to output_x{XMIN}_{XMAX}/).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "Parameter_opt"))

import SCP_parametrizer as pm  # noqa: E402

# inputs/{alg}/{detections,selections}.npy, (N, 2) in [y, x]
RESULTS = HERE / "inputs"

# display_name values from config/realdata.yaml
ALGORITHMS = [("deepfret", "DeepFRET"), ("papylio", "Papylio"), ("scp", "SCP")]
TARGET_NAME = "Target Interactions"
OFFTARGET_NAME = "Off-Target Interactions"
NOISE_NAME = "Pure Noise"

TARGET_CLASS = pm.TEST_POSITIVE_CLASS      # 0
OFFTARGET_CLASS = pm.TEST_NEGATIVE_CLASS   # 1
MARGIN = pm.MATCH_MARGIN                   # 3


def load_xy(path):
    """(N, 2) in [y, x] -> list of (x, y)."""
    arr = np.load(path)
    if arr.size == 0:
        return []
    return [(float(r[1]), float(r[0])) for r in arr]


def load_annotations():
    coord = np.load(pm.TEST_COORD_NPY)
    cls = np.load(pm.TEST_CLASS_NPY)
    spots = [(int(c[1]), int(c[0])) for c in coord]     # [y, x] -> (x, y)
    return spots, cls


def in_band(spot, xband):
    """Same as pm._in_band."""
    return xband[0] <= spot[0] < xband[1]


def split_by_class(spots, cls, detections):
    """Match against all annotations and count target / off-target / noise."""
    pairs, _, unmatched_det = pm.match_detections_to_positives(spots, detections, margin=MARGIN)
    index_of = {tuple(s): i for i, s in enumerate(spots)}
    n_target = n_offtarget = 0
    for p in pairs:
        c = cls[index_of[tuple(p['positive'])]]
        if c == TARGET_CLASS:
            n_target += 1
        elif c == OFFTARGET_CLASS:
            n_offtarget += 1
    return n_target, n_offtarget, len(unmatched_det)


def safe_div(n, d):
    return n / d if d > 0 else 0.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--band", nargs=2, type=int, metavar=("XMIN", "XMAX"),
                    default=None,
                    help="score the half-open x band [XMIN, XMAX) instead. "
                         "Default: the held-out test strip, 256 512.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory. Default: output/ for the test strip, "
                         "output_x{XMIN}_{XMAX}/ for any other band.")
    return ap.parse_args()


def main():
    args = parse_args()
    xband = tuple(args.band) if args.band else tuple(pm.TEST_XBAND)
    test_strip = xband == tuple(pm.TEST_XBAND)
    whole = xband == (0, 512)
    out = args.out or (HERE / ("output" if test_strip else f"output_x{xband[0]}_{xband[1]}"))
    out.mkdir(parents=True, exist_ok=True)

    all_spots, all_cls = load_annotations()
    keep = [i for i, s in enumerate(all_spots) if in_band(s, xband)]
    spots = [all_spots[i] for i in keep]
    cls = np.asarray([all_cls[i] for i in keep])

    positives = [s for s, c in zip(spots, cls) if c == TARGET_CLASS]
    negatives = [s for s, c in zip(spots, cls) if c == OFFTARGET_CLASS]
    n_targets, n_offtargets = len(positives), len(negatives)

    region = ("whole movie, columns 0-512" if whole
              else f"held-out test strip x[{xband[0]},{xband[1]})" if test_strip
              else f"band x[{xband[0]},{xband[1]})")
    print(f"region: {region}")
    print(f"annotations in region: {n_targets} target, {n_offtargets} off-target, "
          f"{len(spots)} total  (of {len(all_spots)} on the whole movie)")
    print(f"scorer: greedy 1:1, per-axis margin {MARGIN} px, F{pm.FBETA:g}, "
          f"every unmatched detection a false positive\n")

    summary_rows = []
    reconciliation = []

    for alg, display in ALGORITHMS:
        det = [d for d in load_xy(RESULTS / alg / "detections.npy")
               if in_band(d, xband)]
        sel = [d for d in load_xy(RESULTS / alg / "selections.npy")
               if in_band(d, xband)]

        # global metrics (same as tuning)
        m_det = pm.compute_metrics(positives, negatives, det)
        m_sel = pm.compute_metrics(positives, negatives, sel)

        # per-class counts for the bar chart
        t_det, o_det, noise_det = split_by_class(spots, cls, det)
        t_sel, o_sel, noise_sel = split_by_class(spots, cls, sel)

        for stage, m, t in (("detection", m_det, t_det), ("selection", m_sel, t_sel)):
            if m['true_positives'] != t:
                reconciliation.append(
                    f"{display} {stage}: targets-only match {m['true_positives']}, "
                    f"all-annotation match {t} (difference {m['true_positives'] - t})")

        df_class = pd.DataFrame([
            {"Class": TARGET_NAME, "Raw_Count": t_det, "Filtered_Count": t_sel,
             "Retention": safe_div(t_sel, t_det),
             "Recall_Raw": safe_div(t_det, n_targets),
             "Recall_Filtered": safe_div(t_sel, n_targets)},
            {"Class": OFFTARGET_NAME, "Raw_Count": o_det, "Filtered_Count": o_sel,
             "Retention": safe_div(o_sel, o_det),
             "Recall_Raw": safe_div(o_det, n_offtargets),
             "Recall_Filtered": safe_div(o_sel, n_offtargets)},
            {"Class": NOISE_NAME, "Raw_Count": noise_det, "Filtered_Count": noise_sel,
             "Retention": safe_div(noise_sel, noise_det),
             "Recall_Raw": 0.0, "Recall_Filtered": 0.0},
        ])
        df_class.to_csv(out / f"class_benchmarks_{alg}.csv", index=False)

        df_global = pd.DataFrame(
            {"Raw": [m_det['precision'], m_det['recall'], m_det['fbeta']],
             "Filtered": [m_sel['precision'], m_sel['recall'], m_sel['fbeta']]},
            index=["Global_Precision", "Global_Recall", "Global_F1_Score"])
        df_global.to_csv(out / f"global_benchmarks_{alg}.csv")

        for stage, m in (("detection", m_det), ("selection", m_sel)):
            summary_rows.append({
                "algorithm": display, "stage": stage,
                "detections": m['detected_count'],
                "targets_found": m['true_positives'],
                "off_target_hit": m['negatives_hit'],
                "false_positives": m['false_positives'],
                "false_negatives": m['false_negatives'],
                "recall": m['recall'], "precision": m['precision'], "f1": m['fbeta'],
            })

        print(f"{display}")
        print(f"  detection  {m_det['detected_count']:>5} dets  "
              f"R {m_det['recall']:>6.1%}  P {m_det['precision']:>6.1%}  "
              f"F1 {m_det['fbeta']:.4f}   "
              f"[target {t_det}, off-target {o_det}, noise {noise_det}]")
        print(f"  selection  {m_sel['detected_count']:>5} dets  "
              f"R {m_sel['recall']:>6.1%}  P {m_sel['precision']:>6.1%}  "
              f"F1 {m_sel['fbeta']:.4f}   "
              f"[target {t_sel}, off-target {o_sel}, noise {noise_sel}]")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out / "summary.csv", index=False)

    meta = {
        "scorer": "greedy 1:1, per-axis margin 3 px (SCP_parametrizer.compute_metrics)",
        "match_margin_px": MARGIN,
        "fbeta": pm.FBETA,
        "region": region,
        "x_band": list(xband),
        "annotations": {"target": n_targets, "off_target": n_offtargets},
        "annotations_whole_movie": len(all_spots),
        "labels": {"coord": Path(pm.TEST_COORD_NPY).relative_to(PROJ).as_posix(),
                   "class": Path(pm.TEST_CLASS_NPY).relative_to(PROJ).as_posix()},
        "detection_arrays": "benchmark/inputs/{alg}/{detections,selections}.npy",
        "false_positive_rule": "every detection not matched to a target annotation",
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print()
    if reconciliation:
        print("target-count reconciliation (targets-only vs all-annotation matching):")
        for line in reconciliation:
            print(f"  {line}")
    else:
        print("target-count reconciliation: the two matchings agree exactly.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
