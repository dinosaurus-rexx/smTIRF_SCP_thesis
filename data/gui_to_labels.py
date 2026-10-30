"""
Convert a GUI export into coord.npy ([y, x]) and class.npy (0 = positive, 1 = negative).

    python data/gui_to_labels.py export.json OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    export = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])

    coord, cls = [], []
    for label, region_type in ((0, 'positive'), (1, 'negative')):
        for region in export.get(region_type, []):
            c = region['center']
            coord.append((int(round(c['y'])), int(round(c['x']))))   # (x, y) -> [y, x]
            cls.append(label)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "coord.npy", np.array(coord, dtype=int).reshape(-1, 2))
    np.save(out / "class.npy", np.array(cls, dtype=int))
    print(f"{cls.count(0)} positive, {cls.count(1)} negative -> {out}")


if __name__ == "__main__":
    main()
