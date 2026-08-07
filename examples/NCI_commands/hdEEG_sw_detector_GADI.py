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
from turtlewave_hdEEG import (ParalSWA, CustomAnnotations, fmt_freq_token,
                              join_stage_token)
from turtlewave_hdEEG.dbwrite import verify_channel_coverage
from turtlewave_hdEEG.density import event_density, format_density_table


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
    ap.add_argument("--legacy-json", dest="legacy_json", action="store_true", default=False,
                    help="opt back into the legacy JSON -> CSV -> import pipeline. "
                         "By default events go straight into neural_events.db and "
                         "no per-channel JSON or intermediate CSV is written.")
    # Accepted so PBS scripts that still pass it keep running; it now names the
    # default, so it does nothing.
    ap.add_argument("--write-db", dest="write_db_flag", action="store_true",
                    default=False, help=argparse.SUPPRESS)
    ap.add_argument("--resume", action="store_true", default=False,
                    help="skip channels already completed for this exact "
                         "method/band/stage scope in the database")
    ap.add_argument("--loglevel", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("hdEEG_sw_detector")

    if args.write_db_flag:
        logger.warning(
            "--write-db is now the default and has no effect; pass "
            "--legacy-json to get the old JSON -> CSV -> import pipeline.")

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
    out_dir = os.path.join(subj_dir, "wonambi", "sw_results")
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(subj_dir, "wonambi", "neural_events.db")

    logger.info(f"Dataset:    {data_file}")
    logger.info(f"Annotation: {annot_file}")
    logger.info(f"Results:    {'JSON+CSV in ' + out_dir if args.legacy_json else db_path}")
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
        json_dir=out_dir,
        subject=args.subject,
        # Database is the store of record. --legacy-json opts back into the
        # per-channel JSON the export/import steps below consume.
        write_db=False if args.legacy_json else True,
        db_path=None if args.legacy_json else db_path,
        resume=args.resume,
    )

    freq_range = fmt_freq_token(f_lo, f_hi)
    # The canonical token the detectors stamp on events.stage and
    # processing_status. A raw join of a non-canonically-ordered --stages
    # (e.g. NREM3 NREM2) spells the same scope differently, and the
    # coverage check below would then match no status row and report every
    # channel as missing.
    stages_str = join_stage_token(test_stages)

    def check_coverage_or_exit():
        """Verify the run reached the database, else log and exit non-zero.

        An unconditional success message here hides a run whose events never
        landed (a zero-match ``file_pattern``, a failed import); PBS only sees
        the exit status.

        Both write paths store the unescaped method, so the query does not
        depend on which one ran. ``test_method_str`` is for filenames only.
        """
        coverage = verify_channel_coverage(
            db_path            = db_path,
            event_type         = "slow_wave",
            method             = test_method,
            requested_channels = test_channels,
            freq_lower         = f_lo,
            freq_upper         = f_hi,
            stage_key          = stages_str,
            logger             = logger,
        )
        logger.info(
            f"Database coverage: {coverage['covered']}/{coverage['requested']} "
            f"channels accounted for ({coverage['with_events']} with events) "
            f"for event_type=slow_wave, method={test_method}, band={freq_range}, "
            f"stages={stages_str}"
            f"{'' if coverage['scoped_status'] else ' [unscoped status check]'}")
        if coverage['failed']:
            logger.error(
                f"{len(coverage['failed'])} channel(s) recorded a FAILURE for "
                f"this exact scope: {', '.join(coverage['failed'][:20])}")
        if not coverage['complete']:
            missing = coverage['missing']
            logger.error(
                f"{len(missing)} requested channel(s) have no events and no "
                f"successful processing_status record for this scope in "
                f"{db_path}: {', '.join(missing[:20])}"
                f"{' ...' if len(missing) > 20 else ''}")
            sys.exit(1)
        if coverage['events_only']:
            # events.stage is per-epoch, so these rows cannot be proven to
            # come from THIS run's stage set. Say so rather than claim clean.
            logger.warning(
                f"{len(coverage['events_only'])} channel(s) are accounted for "
                f"by existing event rows only, with no processing_status "
                f"record for this exact scope; their events may predate this "
                f"run: {', '.join(coverage['events_only'][:20])}"
                f"{' ...' if len(coverage['events_only']) > 20 else ''}")
        logger.info("All done: every requested channel is accounted for in "
                    "the database")

    def report_density():
        """Log per-channel slow-wave density derived from the database.

        The denominator is the artefact-free in-stage time the detector
        actually analysed, stored in ``analysed_time`` by the run above. The
        rejection settings are forwarded because they are part of that key: a
        mismatch selects a denominator covering a different amount of time.
        """
        try:
            df = event_density(
                db_path, event_type="slow_wave", method=test_method,
                stage=test_stages, subject=args.subject,
                reject_artifacts=args.reject_artifacts,
                reject_arousals=args.reject_arousals)
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Slow-wave density unavailable: {e}")
            return
        if len(df) == 0:
            logger.warning(
                "No slow-wave rows in the database for this scope, so there "
                "is no density to report. The coverage check below says "
                "whether that is an empty night or a lost run.")
            return
        logger.info("Slow-wave density (events per minute of artefact-free "
                    "in-stage time):\n%s", format_density_table(df))
        if len(test_stages) > 1:
            combined = event_density(
                db_path, event_type="slow_wave", method=test_method,
                stage=test_stages, subject=args.subject,
                reject_artifacts=args.reject_artifacts,
                reject_arousals=args.reject_arousals,
                combine_stages=True)
            logger.info("Slow-wave density, stages pooled:\n%s",
                        format_density_table(combined))

    if not args.legacy_json:
        # Slow waves are already in neural_events.db, with det_* + spectral
        # columns and a detection_runs provenance row. There is no file
        # round-trip to perform. A flat CSV can be produced on demand with
        # turtlewave_hdEEG.export_events_to_csv (pass the exact method= for
        # slash-methods).
        logger.info(f"Detection complete; events written to {db_path}")
        report_density()
        check_coverage_or_exit()
        return

    # ---- legacy JSON -> CSV -> import path (--legacy-json) ---------------
    logger.info("Legacy path: exporting JSON to CSV and importing to SQLite...")
    # MUST use the same band token the detector used to name its JSON files.
    file_pattern = f"slowwaves_{test_method_str}_{freq_range}_{stages_str}"
    params_csv = os.path.join(out_dir, f"sw_parameters_{test_method_str}_{freq_range}_{stages_str}.csv")
    event_processor.export_slow_wave_parameters_to_csv(
        json_input=out_dir,
        csv_file=params_csv,
        file_pattern=file_pattern
    )

    dens_csv = os.path.join(out_dir, f"sw_density_{test_method_str}_{freq_range}_{stages_str}.csv")
    # Forward the run's own rejection settings. The density denominator is
    # the recording time the detector actually analysed; leaving these to the
    # exporter's assumption (both True) while detection ran with
    # --reject_arousals off subtracts arousal time the detector never
    # excluded, which biases every density downward.
    event_processor.export_slow_wave_density_to_csv(
        json_input=out_dir,
        csv_file=dens_csv,
        stage=test_stages,
        file_pattern=file_pattern,
        reject_artifacts=args.reject_artifacts,
        reject_arousals=args.reject_arousals
    )

    event_processor.initialize_sqlite_database(db_path)
    # Pass the UNESCAPED method (e.g. 'AASM/Massimini2004'). Without it the
    # importer falls back to filename.split('_')[2], which stores a bare
    # 'AASM' and makes the run indistinguishable from a different method.
    import_stats = event_processor.import_parameters_csv_to_database(
        csv_file=params_csv,
        db_path=db_path,
        event_type="slow_wave",
        method=test_method
    )
    logger.info(f"Import stats: {import_stats}")

    check_coverage_or_exit()


if __name__ == "__main__":
    main()
