"""
Tune SCP's parameters on the labelled movie with Optuna.

The movie is split into three x-strips (train / val / test). Optuna maximises F1
on train, the TOP_K best trials are re-ranked on val, and the test score of the
chosen trial is reported. Results go to runs/optimization_results.json.

    python Parameter_opt/SCP_parametrizer.py
"""
import subprocess
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna

REPO = Path(__file__).resolve().parent.parent

# Detector script
scp_script = str(REPO / "SCP" / "SCP_KTree_JSON.py")
PYTHON_CMD = [sys.executable]

# Fixed detector settings (not tuned)
BANDPASS_BG_SIGMA = 8.0     # background subtraction sigma (px)
EVENT_NMS_RADIUS = 3.0      # duplicate event removal radius (px)
INTENSITY_APERTURE = 0.0    # 0 = old intensity measurement

# Labelled movie: coord.npy is (N, 2) in [y, x], class.npy is 0 = target, 1 = off-target
TEST_TIF = str(REPO / "data" / "vid_ch1.tiff")
TEST_COORD_NPY = str(REPO / "data" / "coord.npy")
TEST_CLASS_NPY = str(REPO / "data" / "class.npy")
TEST_POSITIVE_CLASS = 0
TEST_NEGATIVE_CLASS = 1

# x-strips for train / val / test (87 / 45 / 111 targets)
TRAIN_XBAND = (0, 170)
VAL_XBAND = (170, 256)
TEST_XBAND = (256, 512)

# ============================================================================
# OPTIMIZATION CONFIG
# ============================================================================

N_TRIALS = 60                  # number of Optuna trials
FBETA = 1.0                    # beta for the F-beta score (1 = F1)
TOP_K = 8                      # best train trials compared on val
SEED = 42                      # sampler seed
MATCH_MARGIN = 3               # matching margin in pixels

RUN_DIR = Path(__file__).resolve().parent / "runs"
OUTPUT_DIR = RUN_DIR / "detections"
RESULTS_FILE = str(RUN_DIR / "optimization_results.json")

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def calculate_score(true_positives, false_negatives, false_positives, true_negatives,
                   recall_weight=0.6, precision_weight=0.4):
    """Old weighted score (recall/precision), only reported, not optimized."""
    total_positives = true_positives + false_negatives
    total_detections = true_positives + false_positives

    recall = true_positives / total_positives if total_positives > 0 else 0
    precision = true_positives / total_detections if total_detections > 0 else 0

    return recall_weight * recall + precision_weight * precision


def calculate_fbeta(recall, precision, beta=1.0):
    """F-beta score (beta = 1 gives F1)."""
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_coordinates(json_path):
    """Extract x,y coordinates from detected events JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        coordinates = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if 'center' in item:
                        center = item['center']
                        if 'x' in center and 'y' in center:
                            coordinates.append((center['x'], center['y']))
                            continue
                    if 'x' in item and 'y' in item:
                        coordinates.append((item['x'], item['y']))
                    elif 'X' in item and 'Y' in item:
                        coordinates.append((item['X'], item['Y']))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    coordinates.append((item[0], item[1]))

        return coordinates
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

def is_within_margin(coord1, coord2, margin=3):
    """Check if two coordinates are within the specified pixel margin."""
    return abs(coord1[0] - coord2[0]) <= margin and abs(coord1[1] - coord2[1]) <= margin

def find_matches(training_spots, detected_events, margin=3):
    """Find which training spots are detected in the events."""
    matches = []
    unmatched = []

    for train_coord in training_spots:
        found = False
        for detect_coord in detected_events:
            if is_within_margin(train_coord, detect_coord, margin):
                matches.append({
                    'training': train_coord,
                    'detected': detect_coord,
                    'distance': ((train_coord[0] - detect_coord[0])**2 +
                                (train_coord[1] - detect_coord[1])**2)**0.5
                })
                found = True
                break

        if not found:
            unmatched.append(train_coord)

    return matches, unmatched

def match_detections_to_positives(positive_spots, detected_events, margin=3):
    """
    Greedy 1:1 matching of positive spots and detections, closest pairs first.
    Returns (matched_pairs, unmatched_positives, unmatched_detections).
    """
    candidates = []
    for pi, p in enumerate(positive_spots):
        for di, d in enumerate(detected_events):
            if abs(p[0] - d[0]) <= margin and abs(p[1] - d[1]) <= margin:
                dist = ((p[0] - d[0]) ** 2 + (p[1] - d[1]) ** 2) ** 0.5
                candidates.append((dist, pi, di))
    candidates.sort(key=lambda c: c[0])

    matched_pos = set()
    matched_det = set()
    matched_pairs = []
    for dist, pi, di in candidates:
        if pi in matched_pos or di in matched_det:
            continue
        matched_pos.add(pi)
        matched_det.add(di)
        matched_pairs.append({
            'positive': positive_spots[pi],
            'detected': detected_events[di],
            'distance': dist
        })

    unmatched_positives = [p for i, p in enumerate(positive_spots) if i not in matched_pos]
    unmatched_detections = [d for i, d in enumerate(detected_events) if i not in matched_det]
    return matched_pairs, unmatched_positives, unmatched_detections

# ============================================================================
# PARAMETER EVALUATION
# ============================================================================

DETECTION_TIMEOUT = 1200   # seconds per SCP run

def run_detection(params, tif_path, output_json):
    """Run the SCP script on one movie."""
    cmd = [
        *PYTHON_CMD, scp_script,
        tif_path,
        "--no_plot",
        "--output_json", output_json,
        "--min_sigma", str(params['min_sigma']),
        "--max_sigma", str(params['max_sigma']),
        "--threshold", str(params['intensity_threshold']),
        "--circularity_threshold", str(params['circular_threshold']),
        "--max_displacement", str(params['max_displacement']),
        "--min_pulsation_frames", str(params['min_pulsation_frames']),
        "--intensity_variation_threshold", str(params['intensity_variation_threshold'])
    ]
    cmd += ["--bandpass_bg_sigma", str(BANDPASS_BG_SIGMA)]
    if EVENT_NMS_RADIUS > 0:
        cmd += ["--event_nms_radius", str(EVENT_NMS_RADIUS)]
    if INTENSITY_APERTURE > 0:
        cmd += ["--intensity_aperture", str(INTENSITY_APERTURE)]

    # UTF-8 output, otherwise SCP's print of '→' fails on Windows
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }

    # remove old output file
    try:
        os.remove(output_json)
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                                env=env, timeout=DETECTION_TIMEOUT)
        return True, result.stdout
    except subprocess.TimeoutExpired:
        print(f"Timeout: detection exceeded {DETECTION_TIMEOUT}s and was aborted")
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        return False, e.stderr

def _empty_metrics(error=None):
    """Metrics dict with all zeros (used when a run fails)."""
    return {
        'fbeta': 0.0, 'weighted_score': 0.0, 'recall': 0.0, 'precision': 0.0,
        'true_positives': 0, 'false_negatives': 0, 'false_positives': 0,
        'true_negatives': 0, 'negatives_hit': 0, 'detected_count': 0,
        'error': error,
    }


def compute_metrics(p_spots, n_spots, detected_events):
    """Score detections against annotated positives/negatives."""
    # every detection not matched to a positive counts as a false positive
    matched_pairs, unmatched_positives, unmatched_detections = \
        match_detections_to_positives(p_spots, detected_events, margin=MATCH_MARGIN)
    true_positives = len(matched_pairs)
    false_negatives = len(unmatched_positives)
    false_positives = len(unmatched_detections)

    # negatives that were detected (only reported)
    n_matches, n_unmatched = find_matches(n_spots, detected_events, margin=MATCH_MARGIN)
    negatives_hit = len(n_matches)
    true_negatives = len(n_unmatched)

    total_positives = len(p_spots)
    recall = true_positives / total_positives if total_positives > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    return {
        'fbeta': calculate_fbeta(recall, precision, beta=FBETA),
        'weighted_score': calculate_score(true_positives, false_negatives, false_positives, true_negatives),
        'recall': recall,
        'precision': precision,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'negatives_hit': negatives_hit,
        'detected_count': len(detected_events),
        'error': None,
    }

# ============================================================================
# SEARCH SPACE (Optuna / Bayesian)
# ============================================================================

PARAM_BOUNDS = {
    'min_sigma': (0.0, 1.5),
    'max_sigma': (0.5, 3.5),
    # below ~0.2 the detector gets too slow on the real movie
    'intensity_threshold': (0.25, 0.5),
    'circular_threshold': (0.1, 0.6),
    'max_displacement': (0.5, 5.0),
    # minimum track length in frames
    'min_pulsation_frames': (3, 150),
    'intensity_variation_threshold': (0.05, 0.25),
}


def suggest_params(trial):
    """Draw one parameter set from the search space for this trial."""
    bounds = PARAM_BOUNDS
    min_sigma = trial.suggest_float('min_sigma', *bounds['min_sigma'])
    # max_sigma > min_sigma
    ms_lo, ms_hi = bounds['max_sigma']
    max_sigma = trial.suggest_float('max_sigma', max(ms_lo, min_sigma + 0.1), ms_hi)
    return {
        'min_sigma': min_sigma,
        'max_sigma': max_sigma,
        'intensity_threshold': trial.suggest_float('intensity_threshold', *bounds['intensity_threshold']),
        'circular_threshold': trial.suggest_float('circular_threshold', *bounds['circular_threshold']),
        'max_displacement': trial.suggest_float('max_displacement', *bounds['max_displacement']),
        'min_pulsation_frames': trial.suggest_int('min_pulsation_frames', *bounds['min_pulsation_frames']),
        'intensity_variation_threshold': trial.suggest_float('intensity_variation_threshold', *bounds['intensity_variation_threshold']),
    }

# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_labels(coord_npy, class_npy):
    """Load labels and return (x, y) lists of positives and negatives."""
    coord = np.load(coord_npy)          # (N, 2) as [y, x]
    cls = np.load(class_npy)            # (N,)
    xy = [(int(c[1]), int(c[0])) for c in coord]   # [y, x] -> (x, y)
    positives = [xy[i] for i in range(len(xy)) if cls[i] == TEST_POSITIVE_CLASS]
    negatives = [xy[i] for i in range(len(xy)) if cls[i] == TEST_NEGATIVE_CLASS]
    return positives, negatives


def load_test_movie():
    """Movie path and labels, or None if the files are missing."""
    if not (os.path.exists(TEST_TIF) and os.path.exists(TEST_COORD_NPY) and os.path.exists(TEST_CLASS_NPY)):
        return None
    positives, negatives = load_test_labels(TEST_COORD_NPY, TEST_CLASS_NPY)
    return {
        'name': 'real',
        'tif': TEST_TIF,
        'positive_spots': positives,
        'negative_spots': negatives,
    }

# ============================================================================
# STRIP SCORING
# ============================================================================

def _in_band(spot, xband):
    """spot is (x, y); band is (x_min, x_max) half-open on x."""
    return xband[0] <= spot[0] < xband[1]


def evaluate_real_bands(params, positives, negatives, output_json, bands, verbose=False):
    """Run the detector once and score each (name, (x_min, x_max)) band."""
    success, _ = run_detection(params, TEST_TIF, output_json)
    if not success or not os.path.exists(output_json):
        empty = _empty_metrics('real detection failed/empty')
        out = {name: empty for name, _ in bands}
        out.update({'detected_count': 0, 'error': 'real detection failed'})
        return out

    dets = extract_coordinates(output_json)
    out = {'detected_count': len(dets), 'error': None}
    for name, band in bands:
        p = [s for s in positives if _in_band(s, band)]
        n = [s for s in negatives if _in_band(s, band)]
        d = [s for s in dets if _in_band(s, band)]
        out[name] = compute_metrics(p, n, d)
        if verbose:
            m = out[name]
            print(f"    {name} band x{band}: F{FBETA:g}={m['fbeta']:.4f} "
                  f"R={m['recall']:.2%} P={m['precision']:.2%} "
                  f"(pos={len(p)} dets={len(d)})")
    return out

# ============================================================================
# SEARCH
# ============================================================================

# Starting points for the search
WARM_START = (
    {'min_sigma': 0.5, 'max_sigma': 2.5, 'intensity_threshold': 0.30,
     'circular_threshold': 0.10, 'max_displacement': 3.0,
     'min_pulsation_frames': 3, 'intensity_variation_threshold': 0.05},
    {'min_sigma': 0.5, 'max_sigma': 2.5, 'intensity_threshold': 0.35,
     'circular_threshold': 0.30, 'max_displacement': 2.0,
     'min_pulsation_frames': 5, 'intensity_variation_threshold': 0.10},
    {'min_sigma': 0.7, 'max_sigma': 2.0, 'intensity_threshold': 0.40,
     'circular_threshold': 0.25, 'max_displacement': 2.0,
     'min_pulsation_frames': 4, 'intensity_variation_threshold': 0.08},
    {'min_sigma': 0.4515457399879006, 'max_sigma': 1.2689848283173364,
     'intensity_threshold': 0.32336429282837625,
     'circular_threshold': 0.20209149318019778,
     'max_displacement': 4.9934159843646295,
     'min_pulsation_frames': 20,
     'intensity_variation_threshold': 0.07179385564931576},
)


def run_tuning():
    """Optimize on train, pick on val, report test."""
    test_movie = load_test_movie()
    if test_movie is None:
        print(f"ERROR: real movie/labels not found ({TEST_TIF}). See data/README.md.")
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    real_pos, real_neg = test_movie['positive_spots'], test_movie['negative_spots']
    bands = [('train', TRAIN_XBAND), ('val', VAL_XBAND), ('test', TEST_XBAND)]
    for name, b in bands:
        print(f"  {name} band x{b}: {sum(_in_band(s, b) for s in real_pos)} positive spots")

    print("=" * 80)
    print(f"Optuna TPE, {N_TRIALS} trials on the real movie")
    print("(optimize TRAIN strip, select on VAL, report TEST)")
    print("=" * 80)

    def objective(trial):
        params = suggest_params(trial)
        out_json = str(OUTPUT_DIR / f"_scp_detected_t{trial.number}.json")
        print(f"\n[trial {trial.number}] thr={params['intensity_threshold']:.3f} "
              f"sig={params['min_sigma']:.2f}-{params['max_sigma']:.2f} "
              f"circ={params['circular_threshold']:.2f} puls={params['min_pulsation_frames']}")
        res = evaluate_real_bands(params, real_pos, real_neg, out_json, bands=bands, verbose=True)
        for nm in ('train', 'val', 'test'):
            trial.set_user_attr(f'{nm}_fbeta', res[nm]['fbeta'])
            trial.set_user_attr(f'{nm}_recall', res[nm]['recall'])
            trial.set_user_attr(f'{nm}_precision', res[nm]['precision'])
        return res['train']['fbeta']

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for seed_cfg in WARM_START:
        study.enqueue_trial(seed_cfg)

    study.optimize(objective, n_trials=N_TRIALS)

    # best val score among the top train trials
    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    topk = completed[:TOP_K]
    best = max(topk, key=lambda t: t.user_attrs.get('val_fbeta', 0))

    print("\n" + "=" * 80)
    print(f"Selected trial {best.number}")
    print("Final parameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("\nGENERALIZATION (F-beta on the real movie, by strip):")
    for nm in ('train', 'val', 'test'):
        print(f"  {nm:5s}: F{FBETA:g}={best.user_attrs.get(nm+'_fbeta',0):.3f}  "
              f"R={best.user_attrs.get(nm+'_recall',0):.1%}  P={best.user_attrs.get(nm+'_precision',0):.1%}")
    print("=" * 80)

    payload = {
        'method': 'real_movie_strips',
        'fbeta': FBETA,
        'n_trials': len(study.trials),
        'bands': {'train': TRAIN_XBAND, 'val': VAL_XBAND, 'test': TEST_XBAND},
        'warm_start': [dict(c) for c in WARM_START],
        'param_bounds': {k: list(v) for k, v in PARAM_BOUNDS.items()},
        'event_nms_radius': EVENT_NMS_RADIUS,
        'seed': SEED,
        'top_k': TOP_K,
        'final_parameters': best.params,
        'final_trial_number': best.number,
        'train_metrics': {k: best.user_attrs.get('train_'+k) for k in ('fbeta', 'recall', 'precision')},
        'val_metrics': {k: best.user_attrs.get('val_'+k) for k in ('fbeta', 'recall', 'precision')},
        'test_metrics': {k: best.user_attrs.get('test_'+k) for k in ('fbeta', 'recall', 'precision')},
        'all_trials': [
            {'number': t.number, 'params': t.params,
             'train_fbeta': t.user_attrs.get('train_fbeta'),
             'val_fbeta': t.user_attrs.get('val_fbeta'),
             'test_fbeta': t.user_attrs.get('test_fbeta')}
            for t in study.trials
        ],
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    return payload


if __name__ == "__main__":
    run_tuning()
