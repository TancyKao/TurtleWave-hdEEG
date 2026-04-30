#!/usr/bin/env python3
"""
hdEEG_annotator_GADI.py
"""

import os
import glob
import argparse
from turtlewave_hdEEG import LargeDataset, XLAnnotations

def find_first(patterns):
    """Return the first matching file across a list of glob patterns (sorted within each)."""
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser(description="Generate hdEEG annotations for one subject.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--datafile", default=None)
    ap.add_argument("--memmap", action="store_true")
    args = ap.parse_args()

    subj_dir = os.path.join(args.root, args.subject)
    if not os.path.isdir(subj_dir):
        raise FileNotFoundError(f"Subject directory not found: {subj_dir}")

    if args.datafile:
        datafile = args.datafile
    else:
        # Dataset patterns
        patterns = [
            os.path.join(subj_dir, f"{args.subject}*clean*rebuilt.set"),
            os.path.join(subj_dir, f"{args.subject}_PSG_*_fil_preproc_bclpass_interp_cleanepoch_cleanchans.set"),
            os.path.join(subj_dir, "*.set"),
        ]
        datafile = find_first(patterns)

    if not datafile or not os.path.exists(datafile):
        raise FileNotFoundError(
            "EEG dataset .set not found. Tried patterns:\n  - " +
            "\n  - ".join([
                f"{args.subject}*clean*rebuilt.set",
                f"{args.subject}_PSG_*_fil_preproc_bclpass_interp_cleanepoch_cleanchans.set",
                "*.set",
            ]) + f"\nSearch dir: {subj_dir}"
        )

    print(f"[info] Subject: {args.subject}")
    print(f"[info] Using dataset: {datafile}")

    # Load dataset
    data = LargeDataset(datafile, create_memmap=bool(args.memmap))

    # Output paths
    wonambi_dir = os.path.join(subj_dir, "wonambi")
    os.makedirs(wonambi_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(datafile))[0]
    annot_file = os.path.join(wonambi_dir, f"{base}.xml")

    print(f"[info] Writing annotations to: {annot_file}")

    # Create annotations
    ann = XLAnnotations(data, annot_file)
    ann.process_all()

    print(f"[done] Annotations saved: {annot_file}")

if __name__ == "__main__":
    main()
