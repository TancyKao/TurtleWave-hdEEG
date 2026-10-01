"""rerun_detection.py -- scoped re-detection of reviewer-selected bad channels (P3).

Consumes the review-GUI QC hand-off (a ``rerun_sidecar.xml`` carrying the base
staging plus the reviewer's marked artefact epochs, and the list of SELECTED
channels queued for re-detection) and re-runs detection on ONLY those channels,
replacing their events in ``neural_events.db`` while every other channel is left
untouched. This is NOT a whole-montage re-detection.

What it does, in order
----------------------
1. Load the sidecar annotation and the EEG file.
2. Resolve the invariant parameters (``ref_chan`` / ``polar`` / ``cat``) from the
   original run's provenance, refusing if they can't be recovered and were not
   supplied (``resolve_rerun_params``).
3. Take the reject set from that same provenance, so the re-detected channels
   are searched over the same time as the channels left alone.
   ``--reject-types`` overrides it and warns.
4. Guard: verify the rater the detector will read carries BOTH the staging AND
   the sidecar artefacts (``verify_rater_match``) -- else the re-run would
   silently reject nothing.
5. Per selected channel, apply the clean-time gate (``channel_clean_gate``):
   channels with too little / too-fragmented artefact-free in-stage time are
   FORCED-DROP, not re-detected.
6. Re-detect the surviving channels with ``write_db=True,
   replace_channels=<survivors>``: artefact epochs are excluded at ESTIMATION
   time by the detector's ``fetch`` (no detect-then-delete), and each channel's
   old rows are DELETE-then-INSERT replaced in one transaction.
7. Record a ``rerun_log`` provenance row (selected/redetected/dropped channels,
   sidecar + snapshot paths) so the run is self-describing and rollback-able.

Export / import are intentionally skipped; the DB is the source of truth.

Usage
-----
::

    python examples/rerun_detection.py \
        --annot  /path/qc_backup/<ts>/rerun_sidecar.xml \
        --eeg    /path/sub-XXX_eeg.set \
        --db     /path/wonambi/neural_events.db \
        --channels /path/selected_channels.csv \
        --event-type spindle --method Wamsley2012 \
        --freq 11 16 --stages NREM2 NREM3

``--ref-chan`` / ``--polar`` / ``--cat`` are optional; when omitted they are
recovered from the original run recorded in ``detection_runs``. The re-run
refuses if they can be recovered from neither source.

Note on the GUI hand-off
------------------------
The current ``eeg_review_gui`` writes ``channels.csv`` as the *kept* channels
(whole montage minus dropped) and ``redetect_request.json``'s
``exclude_channels`` as the re-detect queue unioned with dropped channels;
neither emits the SELECTED re-detect channels as a clean, distinct list. Until a
GUI change adds one, pass the selected channels here explicitly via
``--channels`` (one channel per row, no header).
"""

import argparse
import logging
import os
import sys

from turtlewave_hdEEG import open_dataset  # reads both EEGLAB layouts

from turtlewave_hdEEG import (CustomAnnotations, ParalEvents, ParalSWA, ParalKC,
                              dbwrite)
from turtlewave_hdEEG.rerun import (RerunGuardError, verify_rater_match,
                                    channel_clean_gate, resolve_rerun_params,
                                    resolve_sw_amplitude_thresholds)
from turtlewave_hdEEG.utils import (read_channels_from_csv,
                                   resolve_reject_types,
                                   KNOWN_REJECT_TYPES)


LOG = logging.getLogger('turtlewave_hdEEG.rerun_driver')


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description='Scoped re-detection of reviewer-selected bad channels.')
    p.add_argument('--annot', required=True,
                   help='rerun_sidecar.xml (base staging + reviewer artefacts)')
    p.add_argument('--eeg', required=True, help='EEG data file (.set/.edf/...)')
    p.add_argument('--db', required=True, help='neural_events.db to update')
    p.add_argument('--channels', required=True,
                   help='CSV of SELECTED re-detect channels (no header, one '
                        'per row) -- the replace scope')
    p.add_argument('--event-type', required=True,
                   choices=['spindle', 'slow_wave', 'k_complex'])
    p.add_argument('--method', required=True,
                   help='detection method (e.g. Wamsley2012, Staresina2015, '
                        'AASM/Massimini2004)')
    p.add_argument('--freq', nargs=2, type=float, required=True,
                   metavar=('LO', 'HI'), help='detection band in Hz')
    p.add_argument('--stages', nargs='+', required=True,
                   help='detection stages (e.g. NREM2 NREM3)')
    p.add_argument('--ref-chan', nargs='*', default=None,
                   help='reference channel(s); recovered from provenance if '
                        'omitted')
    p.add_argument('--polar', default=None, choices=['normal', 'opposite'],
                   help="polarity; recovered from provenance if omitted")
    p.add_argument('--cat', nargs=4, type=int, default=None,
                   metavar=('C0', 'C1', 'C2', 'C3'),
                   help='fetch cat tuple (e.g. 1 1 1 0). Recovered from '
                        'provenance for P3+ runs; REQUIRED for a re-run against '
                        'a pre-P3 (P2) database, which never recorded cat -- the '
                        're-run refuses rather than assume a default.')
    p.add_argument('--grp-name', default='eeg')
    p.add_argument('--n-min-min', type=float, default=5.0,
                   help='minimum artefact-free in-stage MINUTES to re-detect a '
                        'channel (else forced-drop). Default 5.')
    p.add_argument('--max-excluded-frac', type=float, default=0.5,
                   help='max fraction of in-stage time excluded before a '
                        'channel is forced-drop. Default 0.5.')
    p.add_argument('--reject-types', default=None,
                   help='comma-separated annotation event types to exclude '
                        'from detection AND from the density denominator '
                        '(e.g. "Artefact,Arousal,Move"). Default: the set '
                        'the ORIGINAL run recorded, so the re-detected '
                        'channels match the ones left alone; the library '
                        'default only when that run recorded none. Passing '
                        'this overrides the original and warns. Pass an empty '
                        'string to reject nothing. Known types: '
                        + ', '.join(KNOWN_REJECT_TYPES) + '.')
    p.add_argument('--no-reject-artifacts', dest='reject_artifacts',
                   action='store_false',
                   help='deprecated: removes Artefact from --reject-types')
    p.add_argument('--no-reject-arousals', dest='reject_arousals',
                   action='store_false',
                   help='deprecated: removes Arousal from --reject-types')
    p.add_argument('--backup', default=None,
                   help='qc_backup snapshot dir, recorded in rerun_log as the '
                        'rollback point (defaults to the sidecar directory)')
    p.add_argument('--requested-by', default=None,
                   help='reviewer name, recorded in rerun_log')
    # None, not True: "not given" must stay distinguishable from "asked for",
    # so --reject-types alone decides the set and the flags only subtract.
    p.set_defaults(reject_artifacts=None, reject_arousals=None)
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    db_path = args.db
    if os.path.isdir(db_path):
        db_path = os.path.join(db_path, 'neural_events.db')

    selected = read_channels_from_csv(args.channels) or []
    selected = [str(c) for c in selected]
    if not selected:
        LOG.error("No channels in %s; nothing to re-detect.", args.channels)
        return 2
    LOG.info("Selected %d channel(s) for re-detection: %s",
             len(selected), selected)

    # 1. Load sidecar annotation + EEG.
    annot = CustomAnnotations(args.annot)
    data = open_dataset(args.eeg)
    try:
        s_freq = data.header['s_freq']
    except Exception:
        s_freq = None

    # 2. Resolve invariant parameters from the original run (or explicit args).
    # Runs BEFORE the reject set is fixed: a re-run must repeat what the
    # original run excluded, not what today's library default excludes. The
    # re-detected channels are compared against channels this run does not
    # touch, so a different reject set means the two halves of the montage were
    # searched over different time -- and the stored density denominator is
    # keyed on the original set, so it would no longer describe these channels.
    try:
        resolved = resolve_rerun_params(
            db_path, args.event_type, args.method,
            freq_lower=args.freq[0], freq_upper=args.freq[1],
            ref_chan=args.ref_chan, polar=args.polar,
            cat=(tuple(args.cat) if args.cat is not None else None),
            logger=LOG)
    except RerunGuardError as e:
        LOG.error("Parameter-invariance guard failed: %s", e)
        return 4
    ref_chan = resolved['ref_chan']
    polar = resolved['polar']
    cat = resolved['cat']
    recovered = resolved['recovered'] or {}
    orig_params = recovered.get('params', {}) or {}
    recorded_reject = recovered.get('reject_types') or None

    # 3. Reject set: the ORIGINAL run's by default.
    requested = (None if args.reject_types is None
                 else [t for t in args.reject_types.split(',') if t.strip()])
    overridden = (requested is not None or args.reject_artifacts is not None
                  or args.reject_arousals is not None)
    if not overridden and recorded_reject:
        reject_types = list(resolve_reject_types(recorded_reject, logger_=LOG))
        LOG.info("Reject set recovered from the original run: %s. Pass "
                 "--reject-types to override.",
                 ", ".join(reject_types) or "nothing rejected")
    else:
        reject_types = list(resolve_reject_types(
            requested, args.reject_artifacts, args.reject_arousals,
            logger_=LOG))
        if not overridden:
            LOG.warning(
                "The original run recorded no reject set (a pre-4.4 run whose "
                "booleans were not stored either), so this re-run uses the "
                "library default: %s. If that run excluded something else, "
                "the re-detected channels are searched over different time "
                "from the channels left alone. Pass --reject-types to state "
                "what it used.", ", ".join(reject_types) or "nothing")
        elif recorded_reject and set(reject_types) != set(recorded_reject):
            LOG.warning(
                "--reject-types OVERRIDES the original run: that run excluded "
                "%s, this re-run excludes %s. The re-detected channels will "
                "have been searched over different time from every channel "
                "this run leaves alone, and the stored density denominator "
                "for this scope is keyed on the ORIGINAL set, so it does not "
                "describe the new rows. Re-detect the whole montage instead "
                "if you mean to change the reject set.",
                ", ".join(recorded_reject) or "nothing",
                ", ".join(reject_types) or "nothing")

    # 4. Rater-match guard -- fail loudly rather than silently un-rejected.
    try:
        verify_rater_match(annot, reject_types, logger=LOG)
    except RerunGuardError as e:
        LOG.error("Rater-match guard failed: %s", e)
        return 3

    # 5. Clean-time gate. It is whole-montage (channel-global): it reuses
    # compute_analysed_seconds(chan=None), the only artefact subtraction Wonambi
    # supports faithfully, and the review-GUI sidecar marks artefacts
    # whole-montage. So it is computed ONCE and is all-or-nothing for this run --
    # every selected channel is re-detected, or every one is forced-drop. A true
    # per-channel gate (needing per-channel artefact marking + per-channel fetch)
    # is a documented follow-up.
    gate = channel_clean_gate(
        annot, args.stages, s_freq=s_freq,
        n_min_sec=args.n_min_min * 60.0,
        max_excluded_frac=args.max_excluded_frac,
        reject_types=reject_types)
    if gate['ok']:
        redetect, dropped = list(selected), []
        LOG.info("Whole-montage clean-time gate PASSED "
                 "(clean=%.2f min, excluded=%.1f%%); re-detecting all %d "
                 "selected channel(s).",
                 gate['clean_sec'] / 60.0, gate['excluded_frac'] * 100,
                 len(selected))
    else:
        redetect, dropped = [], list(selected)
        LOG.warning("Whole-montage clean-time gate FAILED (channel-global, "
                    "all-or-nothing): %s (clean=%.2f min, excluded=%.1f%%). "
                    "FORCED-DROP all %d selected channel(s); nothing "
                    "re-detected.", gate['reason'], gate['clean_sec'] / 60.0,
                    gate['excluded_frac'] * 100, len(selected))
        return 5

    # 6. Re-detect survivors with scoped replace.
    freq = (args.freq[0], args.freq[1])
    common = dict(chan=redetect, ref_chan=ref_chan, grp_name=args.grp_name,
                  frequency=freq, polar=polar, stage=args.stages, cat=cat,
                  reject_types=reject_types,
                  save_to_annotations=False, json_dir=None,
                  write_db=True, db_path=db_path, resume=False,
                  replace_channels=redetect)

    if args.event_type == 'spindle':
        proc = ParalEvents(dataset=data, annotations=annot)
        # Per-method first: on a mixed-method run the scalar 'duration' is
        # None (the methods resolved to different bounds), and replaying it
        # would silently impose one method's bound on another's re-detection.
        by_method = orig_params.get('duration_by_method') or {}
        duration = by_method.get(args.method, orig_params.get('duration'))
        if duration is not None:
            common['duration'] = tuple(duration)
            LOG.info("Re-using the original run's duration bound for %s: %s",
                     args.method, tuple(duration))
        else:
            LOG.info("The original run recorded no duration bound for %s, so "
                     "the method's published default is used.", args.method)
        proc.detect_spindles(method=args.method, **common)
    elif args.event_type == 'slow_wave':
        proc = ParalSWA(dataset=data, annotations=annot)
        sw_kwargs = dict(common)
        # Reuse the original detector thresholds/durations when recorded.
        for key in ('trough_duration', 'min_dur', 'max_dur', 'detrend',
                    'peak_thresh_sigma', 'ptp_thresh_sigma'):
            if orig_params.get(key) is not None:
                sw_kwargs[key] = (tuple(orig_params[key])
                                  if key == 'trough_duration'
                                  else orig_params[key])
        # The two amplitude thresholds go through the version-aware resolver:
        # for Ngo2015/Staresina2015 a pre-4.3.0 run recorded a value that was
        # compared against a sample count, so replaying it verbatim would
        # impose a microvolt floor the original run never applied.
        neg_thr, p2p_thr = resolve_sw_amplitude_thresholds(
            args.method,
            orig_params.get('neg_peak_thresh'),
            orig_params.get('p2p_thresh'),
            recorded_version=(resolved['recovered'] or {}).get(
                'turtlewave_version'),
            logger=LOG)
        if neg_thr is not None:
            sw_kwargs['neg_peak_thresh'] = neg_thr
        if p2p_thr is not None:
            sw_kwargs['p2p_thresh'] = p2p_thr
        proc.detect_slow_waves(method=args.method, **sw_kwargs)
    else:  # k_complex
        proc = ParalKC(dataset=data, annotations=annot)
        kc_kwargs = dict(common)
        for key in ('trough_duration', 'neg_peak_thresh', 'p2p_thresh',
                    'min_isolation', 'detrend'):
            if orig_params.get(key) is not None:
                kc_kwargs[key] = (tuple(orig_params[key])
                                  if key == 'trough_duration'
                                  else orig_params[key])
        proc.detect_kcomplexes(method=args.method, **kc_kwargs)

    # 7. Record the re-run for provenance / rollback.
    import uuid as _uuid
    conn = dbwrite.open_write_connection(db_path)
    try:
        dbwrite.ensure_direct_write_schema(conn, LOG)
        # The most recent matching run is the one detection just wrote.
        new_run = dbwrite.recover_run_scope(
            db_path, args.event_type, args.method, freq[0], freq[1])
        new_run_id = new_run['run_id'] if new_run else None
        backup = args.backup or os.path.dirname(os.path.abspath(args.annot))
        dbwrite.record_rerun(
            conn, str(_uuid.uuid4()), new_run_id, args.event_type, args.method,
            # The same canonical token the detectors stamp on events.stage and
            # processing_status, so this provenance row names the scope in the
            # one spelling everything else uses.
            freq[0], freq[1], dbwrite.join_stage_token(args.stages),
            selected, redetect,
            dropped, os.path.abspath(args.annot), backup, args.requested_by)
    finally:
        conn.close()

    LOG.info("Re-run complete. Re-detected %d channel(s): %s",
             len(redetect), redetect)
    if dropped:
        LOG.info("Forced-drop %d channel(s) (insufficient clean data): %s",
                 len(dropped), dropped)
    LOG.info("Every other channel in %s was left untouched.", db_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
