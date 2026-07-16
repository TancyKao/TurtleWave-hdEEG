"""
backfill_pac_to_db.py
One-shot driver to back-fill existing phase-amplitude coupling (PAC) result
CSVs into the ``pac_coupling`` table of a ``neural_events.db`` SQLite database.

This walks a subject's PAC results tree (as written by
``ParalPAC.analyze_pac`` / ``export_pac_parameters_to_csv``), reads every
per-channel ``*_pac_parameters.csv``, recovers its event count from the sibling
``*_mean_amps.npy``, and writes one idempotent row per
(subject, channel, event_type, method, stage, phase band, amp band).

Ordering note
-------------
This driver stores the preferred-phase value exactly as it appears in the CSV.
If your CSVs predate the preferred-phase 180-degree fix, run the historical
migration (``examples/fix_pac_preferred_phase.py``) FIRST, then back-fill; the
back-fill itself does not correct polarity.

Usage
-----
    python backfill_pac_to_db.py --root /path/to/SUBJECT/wonambi/pac_results \
        --db /path/to/SUBJECT/wonambi/neural_events.db

    # override the subject id (default: basename of --root's parent chain)
    python backfill_pac_to_db.py --root .../pac_results --db .../neural_events.db \
        --subject SUB001
"""

import os
import sys
import argparse
import logging

from turtlewave_hdEEG import ParalPAC


class _NoDataset:
    """Minimal stand-in so ParalPAC can be built without loading EEG data.

    The back-fill path only touches the filesystem and SQLite; it never reads
    the recording, so a real ``wonambi.Dataset`` is unnecessary here.
    """

    filename = ''


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Back-fill existing PAC result CSVs into the pac_coupling DB table.")
    parser.add_argument('--root', required=True,
                        help="Root of the PAC results tree to walk "
                             "(e.g. .../wonambi/pac_results).")
    parser.add_argument('--db', required=True,
                        help="Path to the target neural_events.db SQLite file.")
    parser.add_argument('--subject', default='folder',
                        help="Subject id. 'folder' (default) uses the basename "
                             "of --root; any other value is used literally.")
    parser.add_argument('--log-file', default=None,
                        help="Optional path to write logs to (in addition to console).")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.root):
        parser.error(f"--root is not a directory: {args.root}")
    if not os.path.exists(args.db):
        parser.error(f"--db does not exist: {args.db}")

    log_level = logging.DEBUG if args.verbose else logging.INFO

    # rootpath is only used by ParalPAC for default output locations, which the
    # back-fill path does not exercise; point it at --root for tidy logging.
    ds = _NoDataset()
    pac = ParalPAC(dataset=ds, annotations=None, rootpath=args.root,
                   log_level=log_level, log_file=args.log_file)

    subject_from = args.subject  # 'folder' triggers basename inference
    totals = pac.backfill_pac_directory(args.root, args.db, subject_from=subject_from)

    print("PAC back-fill summary:")
    for k in ('files', 'added', 'updated', 'skipped', 'n_events_missing'):
        print(f"  {k}: {totals.get(k, 0)}")

    # Non-zero exit if any row was rejected for a missing event count, so a
    # batch harness can surface files that need a re-run.
    return 1 if totals.get('n_events_missing', 0) else 0


if __name__ == '__main__':
    sys.exit(main())
