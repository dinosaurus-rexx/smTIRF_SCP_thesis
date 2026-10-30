"""
Extract channel 1 from the raw 2-channel movie (T, C, Y, X) and save it as
data/vid_ch1.tiff. The annotations were made on channel 1.

    python data/prepare_real_movie.py path/to/vid.tiff
"""
import sys
from pathlib import Path

import tifffile

DST = Path(__file__).resolve().parent / "vid_ch1.tiff"
CHANNEL = 1


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    a = tifffile.imread(sys.argv[1])  # (T, C, Y, X)
    print(f"source shape: {a.shape} {a.dtype}")
    if a.ndim != 4:
        raise ValueError(f"expected 4D (T,C,Y,X), got shape {a.shape}")
    single = a[:, CHANNEL, :, :]      # (T, Y, X)
    print(f"extracted channel {CHANNEL}: {single.shape} {single.dtype}")
    tifffile.imwrite(DST, single, imagej=True, metadata={"axes": "TYX"})
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
