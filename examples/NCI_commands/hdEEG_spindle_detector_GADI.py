#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import logging

from turtlewave_hdEEG.utils import read_channels_from_csv
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalEvents, CustomAnnotations

def find_one(patterns):
    """Return the first existing path matched by any of the glob patterns, else None."""
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser(description="Run hdEEG spindle detection for a single subject.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--method", default="Moelle2011", choices=["Moelle2011", "Ferrarelli2007"])
    ap.add_argument("--stages", default="NREM2,NREM3")
    ap.add_argument("--freq", default="9.0,12.0")
    ap.add_argument("--duration", default="0.5,3")
    ap.add_argument("--reject_artifacts", action="store_true", default=True)
    ap.add_argument("--reject_arousals", action="store_true", default=False)
    ap.add_argument("--loglevel", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    # Logging
    loglevel = getattr(logging, args.loglevel)
    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

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

    # Try to auto-discover dataset (.set) and annotation (.xml)
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
        logger.error(f"No dataset .set file found under: {set_candidates}")
        sys.exit(5)
    if not annot_file:
        logger.error(f"No annotation .xml found under: {xml_candidates}")
        sys.exit(6)

    json_dir = os.path.join(subj_dir, "wonambi", "spindle_results")
    os.makedirs(json_dir, exist_ok=True)
    db_path = os.path.join(subj_dir, "wonambi", "neural_events.db")

    logger.info(f"Dataset:    {data_file}")
    logger.info(f"Annotation: {annot_file}")
    logger.info(f"JSON out:   {json_dir}")
    logger.info(f"DB path:    {db_path}")

    # Parse CLI fields
    test_method = args.method
    test_stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    f_lo, f_hi = [float(x) for x in args.freq.split(",")]
    d_lo, d_hi = [float(x) for x in args.duration.split(",")]

    # Load dataset/annotations
    logger.info("Loading dataset and annotations...")
    data = WonambiDataset(data_file)
    annot = CustomAnnotations(annot_file)

    event_processor = ParalEvents(
        dataset=data,
        annotations=annot
        # You can enable log file if needed:
        # log_level=logging.DEBUG,
        # log_file=os.path.join(subj_dir, "wonambi", "spindle_detection.log"),
    )

    logger.info("Running detect_spindles...")
    spindles = event_processor.detect_spindles(
        method              = test_method,
        chan                = test_channels,
        frequency           = (f_lo, f_hi),
        duration            = (d_lo, d_hi),
        stage               = test_stages,
        reject_artifacts    = args.reject_artifacts,
        reject_arousals     = args.reject_arousals,
        cat                 = (1, 1, 1, 0),  
        save_to_annotations = False,
        json_dir            = json_dir
    )

    freq_range = f"{f_lo:.1f}-{f_hi:.1f}Hz"
    stages_str = "".join(test_stages)
    file_pattern = f"spindles_{test_method}_{freq_range}_{stages_str}"

    logger.info("Initializing SQLite database...")
    event_processor.initialize_sqlite_database(db_path)

    params_csv = os.path.join(json_dir, f"spindle_parameters_{test_method}_{freq_range}_{stages_str}.csv")
    dens_csv   = os.path.join(json_dir, f"spindle_density_{test_method}_{freq_range}_{stages_str}.csv")

    logger.info("Exporting parameters CSV...")
    event_processor.export_spindle_parameters_to_csv(
        json_input   = json_dir,
        csv_file     = params_csv,
        file_pattern = file_pattern
    )

    logger.info("Importing parameters into SQLite...")
    event_processor.import_parameters_csv_to_database(
        csv_file = params_csv,
        db_path  = db_path
    )

    logger.info("Exporting density CSV...")
    event_processor.export_spindle_density_to_csv(
        json_input   = json_dir,
        csv_file     = dens_csv,
        stage        = test_stages,
        file_pattern = file_pattern
    )

    logger.info("All done ✓")

if __name__ == "__main__":
    main()
