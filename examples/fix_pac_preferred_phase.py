"""One-off fix for PAC outputs written before the preferred-phase 180-deg fix.

Older `pacprocessor` runs reported preferred phase 180 deg off because the
bin-centre vector spanned [0, 2*pi) while phase was binned on [-pi, pi].
Only the preferred-phase columns are affected; modulation index, mean vector
length, rho and the Rayleigh stats were already correct, and the
`*_mean_amps.npy` files are untouched.

This rotates the affected columns back by 180 deg:
    degrees -> (x - 180) % 360
    radians -> (x - pi) % (2*pi)

It is non-destructive: every CSV is backed up to `<name>.bak` first, and any
file that already has a `.bak` sibling is skipped, so re-running cannot
double-correct.

Usage:
    .venv/bin/python examples/fix_pac_preferred_phase.py /path/to/pac_results
    .venv/bin/python examples/fix_pac_preferred_phase.py /path/to/pac_results --dry-run
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

DEG_COLS = {"preferred_phase_deg", "PP_degrees",
            "Condition1_PP_deg", "Condition2_PP_deg"}
RAD_COLS = {"preferred_phase_rad", "PP_rad",
            "Condition1_PP_rad", "Condition2_PP_rad"}


def fix_file(path, dry_run=False):
    """Correct one CSV. Returns the list of columns changed (empty if none)."""
    bak = path + ".bak"
    if os.path.exists(bak):
        print(f"  skip (already has .bak): {path}")
        return []

    df = pd.read_csv(path)
    changed = []
    for col in df.columns:
        if col in DEG_COLS:
            df[col] = (df[col] - 180.0) % 360.0
            changed.append(col)
        elif col in RAD_COLS:
            df[col] = (df[col] - np.pi) % (2 * np.pi)
            changed.append(col)

    if not changed:
        return []

    if not dry_run:
        os.replace(path, bak)          # back up the original
        df.to_csv(path, index=False)
    print(f"  {'would fix' if dry_run else 'fixed'} {path}  -> {changed}")
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="Directory containing PAC result CSVs (searched recursively)")
    ap.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", "*.csv"), recursive=True))
    print(f"Scanning {len(files)} CSV file(s) under {args.root}")

    n_fixed = 0
    for f in files:
        if f.endswith(".bak"):
            continue
        if fix_file(f, dry_run=args.dry_run):
            n_fixed += 1

    verb = "would correct" if args.dry_run else "corrected"
    print(f"\nDone: {verb} preferred-phase columns in {n_fixed} file(s).")
    if not args.dry_run and n_fixed:
        print("Originals saved as <name>.bak")


if __name__ == "__main__":
    main()
