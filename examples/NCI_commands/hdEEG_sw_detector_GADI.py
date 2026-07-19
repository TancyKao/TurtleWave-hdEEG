#!/usr/bin/env python3
"""
hdEEG_sw_detector.py
Detect slow waves in hdEEG using TurtleWave-hdEEG; export JSON, CSV, and import params to SQLite.
"""

import os
import sys
import argparse
import glob
import logging

from turtlewave_hdEEG.utils import read_channels_from_csv
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalSWA, CustomAnnotations


def find_one(patterns):
    """Return first hit for any glob pattern in patterns, else None."""
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser(description="Run hdEEG slow-wave detection for a single subject.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--subject", required=True)
    # Detection options
    ap.add_argument("--method", default="Staresina2015",
                    choices=["Massimini2004", "AASM/Massimini2004", "Ngo2015", "Staresina2015"],
                    help="Slow-wave detection method.")
    ap.add_argument("--stages", default="NREM2,NREM3")
    ap.add_argument("--freq", default="0.1,4.0")
    ap.add_argument("--trough_duration", default="0.8,2.0")
    ap.add_argument("--neg_peak_thresh", default="-20.0")
    ap.add_argument("--p2p_thresh", default="40.0")
    ap.add_argument("--polar", default="normal", choices=["normal", "opposite"])
    ap.add_argument("--reject_artifacts", action="store_true", default=True)
    ap.add_argument("--reject_arousals", action="store_true", default=False)
    ap.add_argument("--write-db", dest="write_db", action="store_true", default=False,
                    help="write events straight to neural_events.db and skip the "
                         "JSON->CSV->import steps (default off: legacy JSON+CSV path)")
    ap.add_argument("--resume", action="store_true", default=False,
                    help="with --write-db, skip channels already completed for this "
                         "exact method/band/stage scope in the database")
    ap.add_argument("--loglevel", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("hdEEG_sw_detector")

    subj_dir = os.path.join(args.root, args.subject)
    if not os.path.isdir(subj_dir):
        logger.error(f"Subject directory not found: {subj_dir}")
        sys.exit(2)

    # Channels CSV
    channels_csv_path = os.path.join(subj_dir, "channels.csv")
    if not os.path.exists(channels_csv_path):
        logger.error(f"channels.csv not found: {channels_csv_path}")
        sys.exit(3)
    test_channels = read_channels_from_csv(channels_csv_path)
    if not test_channels:
        logger.error("No channels read from channels.csv")
        sys.exit(4)
    logger.info(f"Loaded {len(test_channels)} channel labels")

    set_candidates = [
        os.path.join(subj_dir, f"{args.subject}*clean*rebuilt.set"),
        os.path.join(subj_dir, "*.set"),
    ]
    xml_candidates = [
        os.path.join(subj_dir, "wonambi", f"{args.subject}*clean*rebuilt.xml"),
        os.path.join(subj_dir, "wonambi", "*.xml"),
    ]
    data_file = find_one(set_candidates)
    annot_file = find_one(xml_candidates)
    if not data_file:
        logger.error(f"No dataset .set file found under patterns: {set_candidates}")
        sys.exit(5)
    if not annot_file:
        logger.error(f"No annotation .xml found under patterns: {xml_candidates}")
        sys.exit(6)

    # Output locations
    json_dir = os.path.join(subj_dir, "wonambi", "sw_results")
    os.makedirs(json_dir, exist_ok=True)
    db_path = os.path.join(subj_dir, "wonambi", "neural_events.db")

    logger.info(f"Dataset:    {data_file}")
    logger.info(f"Annotation: {annot_file}")
    logger.info(f"JSON out:   {json_dir}")
    logger.info(f"DB path:    {db_path}")

    # Parse params
    test_method = args.method
    test_method_str = str(test_method).replace("/", "_")
    test_stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    f_lo, f_hi = [float(x) for x in args.freq.split(",")]
    td_lo, td_hi = [float(x) for x in args.trough_duration.split(",")]
    neg_peak_thresh = float(args.neg_peak_thresh)
    p2p_thresh = float(args.p2p_thresh)

    # Load dataset and annotations
    logger.info("Loading dataset and annotations...")
    data = WonambiDataset(data_file)
    annot = CustomAnnotations(annot_file)

    event_processor = ParalSWA(dataset=data, annotations=annot)

    logger.info("Running slow-wave detection...")
    slow_waves = event_processor.detect_slow_waves(
        method=test_method,
        chan=test_channels,
        frequency=(f_lo, f_hi),
        trough_duration=(td_lo, td_hi),
        neg_peak_thresh=neg_peak_thresh,
        p2p_thresh=p2p_thresh,
        polar=args.polar,
        stage=test_stages,
        reject_artifacts=args.reject_artifacts,
        reject_arousals=args.reject_arousals,
        cat=(1, 1, 1, 0),
        save_to_annotations=False,
        json_dir=json_dir,
        create_empty_json=True,
        # Direct-to-DB path (opt-in). Off by default -> legacy behaviour intact.
        write_db=args.write_db,
        db_path=db_path if args.write_db else None,
        resume=args.resume,
    )

    # Filenames
    freq_range = f"{f_lo}-{f_hi}Hz"
    stages_str = "".join(test_stages)
    file_pattern = f"slowwaves_{test_method_str}_{freq_range}_{stages_str}"

    if args.write_db:
        # Slow waves are already in neural_events.db. Skip JSON->CSV->import.
        # A flat CSV can be produced on demand from the DB with
        # export_events_to_csv (pass the exact method= for slash-methods).
        logger.info(f"Direct-to-DB run complete; events written to {db_path}")
        logger.info("All done (direct-to-DB)")
        return

    # Exporting
    logger.info("Exporting parameters CSV...")
    params_csv = os.path.join(json_dir, f"sw_parameters_{test_method_str}_{freq_range}_{stages_str}.csv")
    event_processor.export_slow_wave_parameters_to_csv(
        json_input=json_dir,
        csv_file=params_csv,
        file_pattern=file_pattern
    )

    logger.info("Exporting density CSV...")
    dens_csv = os.path.join(json_dir, f"sw_density_{test_method_str}_{freq_range}_{stages_str}.csv")
    event_processor.export_slow_wave_density_to_csv(
        json_input=json_dir,
        csv_file=dens_csv,
        stage=test_stages,
        file_pattern=file_pattern
    )

    logger.info("Initializing / updating SQLite DB...")
    event_processor.initialize_sqlite_database(db_path)
    event_processor.import_parameters_csv_to_database(
        csv_file=params_csv,
        db_path=db_path
    )

    logger.info("All done ✓")


if __name__ == "__main__":
    main()
