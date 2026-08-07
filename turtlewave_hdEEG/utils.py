import numpy as np
from concurrent.futures import ProcessPoolExecutor
import json
import csv
import logging
import os
import re

logger = logging.getLogger('turtlewave_hdEEG.utils')

#: Matches a BIDS-style subject token anywhere in a filename stem, e.g.
#: ``sub-10sd`` in ``sub-10sd_ses-1_task-psg_run-1_desc-inspect_eeg.xml``.
SUBJECT_RE = re.compile(r"sub-[A-Za-z0-9]+")


def missing_json_message(json_dir, file_pattern, max_listed=25):
    """Build the error text for a ``file_pattern`` that matched no JSON file.

    A zero-match export is nearly always a filename round-trip bug: the band
    or method token in the pattern does not match what the detector wrote.
    Diagnosing it needs the pattern, the directory, and what is actually in
    that directory, so all three go in the message.

    Parameters
    ----------
    json_dir : str
        Directory that was searched.
    file_pattern : str
        The pattern that matched nothing.
    max_listed : int, optional
        Maximum number of filenames to list before truncating.

    Returns
    -------
    str
        Multi-line, human-readable diagnostic message.
    """
    import glob as _glob

    try:
        present = sorted(os.path.basename(p)
                         for p in _glob.glob(os.path.join(str(json_dir), "*.json")))
    except Exception:
        present = []

    if present:
        shown = present[:max_listed]
        listing = "\n  ".join(shown)
        if len(present) > max_listed:
            listing += f"\n  ... and {len(present) - max_listed} more"
        found = (f"{len(present)} JSON file(s) are present:\n  {listing}")
    else:
        found = "the directory contains no .json files at all."

    return (
        f"No JSON files matched file_pattern {file_pattern!r} in {json_dir!r}; "
        f"{found}\n"
        "The pattern must reproduce exactly what the detector wrote: "
        "{event_type}_{method}_{freq_lo}-{freq_hi}Hz_{stages_joined}. "
        "Build the band token with turtlewave_hdEEG.fmt_freq_token(lo, hi) on "
        "both sides, and escape '/' in method names as '_'. "
        "Pass strict=False to fall back to the old placeholder-CSV behaviour.")


#: Canonical column order of a parameters CSV, matching what the export path
#: produces (Wonambi's ``export_event_params`` columns, plus the clock-time and
#: UUID columns the processors add). Used to write a header-only CSV when a run
#: detected no events, so the file is still a valid, machine-readable
#: parameters CSV with zero data rows.
PARAMS_CSV_COLUMNS = [
    'Start time',
    'Start time (HH:MM:SS)',
    'End time',
    'Stage',
    'Cycle',
    'Event type',
    'Channel',
    'Duration (s)',
    'Min. amplitude (uV)',
    'Max. amplitude (uV)',
    'Peak-to-peak amplitude (uV)',
    'RMS (uV)',
    'Power (uV^2)',
    'Peak power frequency (Hz)',
    'Energy (uV^2s)',
    'Peak energy frequency (Hz)',
    'UUID',
]


def write_empty_params_csv(csv_file, event_type, channels=None, logger=None):
    """Write a header-only parameters CSV for a run that detected no events.

    A detector legitimately finding zero events is a valid result, not a
    failure, but it used to leave no CSV at all: the import step was then
    handed a path that did not exist and raised ``FileNotFoundError``, which is
    indistinguishable from the export having failed. Writing the header row
    keeps the file present and parseable, so the importer can read it, see zero
    data rows and report a clean no-op. It also makes the parameters export
    behave like the density export, which already writes a file in this case.

    The file is deliberately a real CSV table rather than a prose placeholder:
    ``pandas.read_csv`` on it yields an empty DataFrame with the expected
    columns, so every downstream consumer keeps working.

    Parameters
    ----------
    csv_file : str
        Path of the CSV to write. Overwritten if it exists.
    event_type : str
        Event type the run was detecting, e.g. ``'spindle'``, ``'slow_wave'``
        or ``'k_complex'``. Used only in the log message.
    channels : list of str or None, optional
        Channels whose JSON held no events, for the log message. Default
        ``None``.
    logger : logging.Logger or None, optional
        Logger for the confirmation message. ``None`` writes the file silently.
        Default ``None``.

    Returns
    -------
    bool
        True if the file was written, False if writing failed (the failure is
        logged, never raised, so a zero-event run cannot turn into a crash).
    """
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(PARAMS_CSV_COLUMNS)
    except Exception as e:
        if logger is not None:
            logger.error(f"Could not write the empty parameters CSV "
                         f"{csv_file}: {e}")
        return False

    if logger is not None:
        n_chan = len(channels) if channels else 0
        where = f" across {n_chan} channel(s)" if n_chan else ""
        logger.info(
            f"No {event_type} events were detected{where}, so there are no "
            f"parameters to measure. Wrote a header-only CSV at {csv_file}; "
            f"the import step will read it and add no rows.")
        if channels:
            logger.debug(f"Channels with no {event_type} events: "
                         f"{', '.join(str(c) for c in channels)}")
    return True


def normalize_subject(subject):
    """Put a subject identifier into the one canonical ``sub-`` form.

    Every subject-keyed table in ``neural_events.db`` (``analysed_time``,
    ``pac_coupling``, ``sleep_cycles``, ``stage_durations``) is joined on this
    string, so two spellings of one recording are two recordings as far as
    SQL is concerned. The directory-name and annotation-filename branches of
    :func:`derive_subject` have always produced the ``sub-`` form; a caller
    passing ``--subject 10sd`` on the command line produced a bare ``10sd``,
    and the two ended up in the same database as separate subjects. That made
    ``event_density(db)`` (which refuses to pick between subjects) raise
    forever, and silently broke every join to the PAC and cycle tables.

    Normalising here, at the single point every caller already goes through,
    is what keeps them identical.

    Parameters
    ----------
    subject : str or None
        Subject identifier in any spelling.

    Returns
    -------
    str or None
        ``subject`` with a ``sub-`` prefix added when absent, whitespace
        stripped. ``None`` and the empty string pass through unchanged, and
        the ``'unknown_subject'`` placeholder is left alone so it stays
        recognisable.

    Examples
    --------
    >>> normalize_subject('10sd')
    'sub-10sd'
    >>> normalize_subject('sub-10sd')
    'sub-10sd'
    >>> normalize_subject('unknown_subject')
    'unknown_subject'
    """
    if subject is None:
        return None
    s = str(subject).strip()
    if not s or s == 'unknown_subject' or s.startswith('sub-'):
        return s
    logger.info(
        "Subject '%s' normalised to 'sub-%s' so it matches the id every other "
        "entry point writes (derive_subject's directory/annotation branches "
        "and the PAC, cycle and stage-duration tables all use the 'sub-' "
        "form).", s, s)
    return f"sub-{s}"


def derive_subject(annotation_path=None, root_dir=None, explicit=None):
    """Resolve the subject identifier for a recording.

    A single, shared implementation of subject resolution so that every
    caller (detectors, PAC, the back-fill scripts, the GUI) keys the database
    the same way. Resolution order is: an explicit value, then a BIDS
    ``sub-XXXX`` token in the annotation XML filename, then the name of the
    recording's root directory.

    There is deliberately **no** whole-filename-stem fallback. Falling back to
    the stem makes the subject id a function of which XML the caller happened
    to point at, and a subject directory routinely holds more than one --
    e.g. ``MCI042_BL_clean_rebuilt.xml`` and
    ``MCI042_BL_clean_rebuilt_review-qc.xml`` would key one recording as two
    different subjects. Since ``subject`` is the primary key of
    ``sleep_cycles``, ``stage_durations`` and ``pac_coupling``, that silently
    splits one recording's rows in two. The directory name is stable across
    such variants, so a non-BIDS filename falls through to it.

    This function never raises. If nothing resolves it returns
    ``'unknown_subject'``, because losing a run to an exception in an id
    helper is worse than storing a placeholder that a human can spot.

    Parameters
    ----------
    annotation_path : str or None, optional
        Path to the Wonambi annotation XML. Only its basename is inspected,
        and only for a ``sub-XXXX`` token; the file need not exist. A filename
        without that token contributes nothing and resolution moves on to
        ``root_dir``.
    root_dir : str or None, optional
        Recording/subject root directory. Its basename is used when the
        annotation filename carries no ``sub-`` token; a ``sub-`` prefix is
        added when absent.
    explicit : str or None, optional
        Caller-supplied subject id. Any truthy value short-circuits the
        search. It is returned stripped and normalised through
        :func:`normalize_subject`, so ``'10sd'`` and ``'sub-10sd'`` both key
        the database as ``'sub-10sd'`` -- a command-line ``--subject 10sd``
        and the GUI's derived id must not become two subjects.

    Returns
    -------
    str
        The resolved subject identifier, e.g. ``'sub-10sd'``.

    Examples
    --------
    >>> derive_subject(explicit='sub-10sd')
    'sub-10sd'
    >>> derive_subject(explicit='10sd')     # normalised, not two subjects
    'sub-10sd'
    >>> derive_subject(annotation_path='/data/w/sub-10sd_ses-1_eeg.xml')
    'sub-10sd'
    >>> derive_subject(root_dir='/data/10sd/')
    'sub-10sd'
    >>> # Non-BIDS filename: the directory decides, so both XMLs in one
    >>> # subject folder resolve to the same id.
    >>> derive_subject(annotation_path='/d/MCI042_BL/MCI042_BL_review-qc.xml',
    ...                root_dir='/d/MCI042_BL')
    'sub-MCI042_BL'
    """
    try:
        if explicit is not None and str(explicit).strip():
            subject = normalize_subject(str(explicit).strip())
            logger.info(f"Subject '{subject}' resolved from: explicit argument")
            return subject

        if annotation_path:
            stem = os.path.basename(str(annotation_path))
            match = SUBJECT_RE.search(stem)
            if match:
                subject = match.group(0)
                logger.info(
                    f"Subject '{subject}' resolved from: annotation filename "
                    f"({stem})")
                return subject
            logger.debug(
                f"No 'sub-' token in annotation filename ({stem}); falling "
                f"back to the root directory name")

        if root_dir:
            folder = os.path.basename(str(root_dir).rstrip(os.sep))
            if folder:
                subject = folder if folder.startswith("sub-") else f"sub-{folder}"
                logger.info(
                    f"Subject '{subject}' resolved from: root directory name "
                    f"({folder})")
                return subject
    except Exception as e:  # never let an id helper break a detection run
        logger.warning(f"derive_subject failed ({e}); using 'unknown_subject'")
        return 'unknown_subject'

    logger.warning(
        "derive_subject: no explicit id, annotation path or root directory "
        "given; using 'unknown_subject'")
    return 'unknown_subject'



def process_events_parallel(events, data_source, window_size=5, n_workers=4, func=None):
    """
    Process EEG events in parallel
    
    Parameters
    ----------
    events : list of dict
        List of events with at least 'start_time' key
    data_source : LargeDataset or str
        Large dataset object or path to data file
    window_size : float
        Window size around event in seconds
    n_workers : int
        Number of parallel workers
    func : callable or None
        Function to apply to each event data, if None just return the data
        
    Returns
    -------
    results : list
        List of processed event data
    """
    from .dataset import LargeDataset
    
    # Initialize data source if needed
    if isinstance(data_source, str):
        data = LargeDataset(data_source)
    else:
        data = data_source
    
    def process_single_event(event):
        # Load data around event
        start = max(0, event['start_time'] - window_size/2)
        end = start + window_size
        event_data = data.read_data(begtime=start, endtime=end)
        
        # Apply custom function if provided
        result = {
            'event_id': event.get('id', None),
            'start_time': event['start_time'],
            'data': event_data,
        }
        
        if func is not None:
            result['analysis'] = func(event_data, event)
            
        return result
    
    # Process events in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_single_event, events))
    
    return results



def explore_eeglab_structure(filename):
    """
    Utility to explore the structure of an EEGLAB file
    
    Parameters
    ----------
    filename : str
        Path to EEGLAB .set file
    
    Returns
    -------
    structure : dict
        Dictionary representation of EEGLAB file structure
    """
    import scipy.io
    import numpy as np
    
    try:
        # Load the EEGLAB file
        eeglab_data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        
        # Helper function to convert MATLAB structs to dictionaries
        def struct_to_dict(struct):
            if isinstance(struct, np.ndarray):
                return [struct_to_dict(s) for s in struct]
            
            if not hasattr(struct, '_fieldnames'):
                return struct
            
            result = {}
            for field in struct._fieldnames:
                value = getattr(struct, field)
                if hasattr(value, '_fieldnames'):
                    result[field] = struct_to_dict(value)
                elif isinstance(value, np.ndarray) and value.dtype.kind == 'O':
                    result[field] = struct_to_dict(value)
                else:
                    result[field] = value
            return result
        
        # Get the EEG structure
        if 'EEG' in eeglab_data:
            eeg = eeglab_data['EEG']
            eeg_dict = struct_to_dict(eeg)
            return eeg_dict
        else:
            logger.warning("EEG structure not found in file")
            return eeglab_data
    
    except Exception as e:
        logger.error(f"Error exploring EEGLAB file: {e}")
        return None

def _merge_intervals(intervals):
    """Merge overlapping/adjacent ``(start, end)`` spans into disjoint sorted spans.

    Parameters
    ----------
    intervals : iterable of (float, float)
        Spans in seconds. Zero-length or inverted spans (``end <= start``) are
        dropped.

    Returns
    -------
    list of (float, float)
        Sorted, non-overlapping spans. Touching spans (``start == prev_end``)
        are merged so adjacent reviewer marks are never double-counted.
    """
    spans = []
    for s, e in intervals:
        s = float(s)
        e = float(e)
        if e > s:
            spans.append((s, e))
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _subtract_spans(segments, cut_spans, min_dur):
    """Remove ``cut_spans`` from ``segments``, dropping sub-``min_dur`` remnants.

    Interval difference that mirrors :func:`wonambi.trans.reject.remove_artf_evts`:
    a fragment surviving a cut is kept only if it is at least ``min_dur`` long, so
    the extra-interval denominator obeys the same minimum-segment floor as the
    detector's own artefact subtraction.

    Parameters
    ----------
    segments : list of (float, float)
        Clean sub-spans (already in-stage, in-epoch, annotation-artefact-free).
    cut_spans : list of (float, float)
        Merged, sorted spans to remove (from :func:`_merge_intervals`).
    min_dur : float
        Minimum surviving fragment length in seconds; shorter fragments are
        dropped, matching ``remove_artf_evts``.

    Returns
    -------
    list of (float, float)
        Segments with ``cut_spans`` removed. Portions of a cut span lying
        outside every segment (e.g. outside the stage/epoch) subtract nothing,
        because only the overlap with a segment is ever removed.
    """
    if not cut_spans:
        return segments
    result = []
    for s, e in segments:
        s = float(s)
        e = float(e)
        cursor = s
        for cs, ce in cut_spans:
            if ce <= cursor or cs >= e:
                continue
            lo = max(cs, cursor)
            if lo - cursor >= min_dur:
                result.append((cursor, lo))
            cursor = min(ce, e)
            if cursor >= e:
                break
        if e - cursor >= min_dur:
            result.append((cursor, e))
    return result


def _drop_inverted_spans(times):
    """Drop spans that end at or before they start, reporting how many.

    Guards the denominator against Wonambi's
    :func:`wonambi.trans.select.get_times`, which clips every epoch end to
    ``annot.last_second`` (``min(e['end'], last)``). An epoch scored *past* the
    annotation's declared ``last_second`` therefore comes back as
    ``(start, last)`` with ``last < start``: a span with a negative length. Summed
    naively it *subtracts* time from the analysed total, which has no defensible
    interpretation.

    Parameters
    ----------
    times : iterable of (float, float)
        Spans in seconds, as returned in a ``get_times`` bundle's ``'times'``.

    Returns
    -------
    kept : list of (float, float)
        Spans with ``end > start``, order preserved.
    n_inverted : int
        Number of spans dropped with ``end < start`` (genuinely negative
        length; these are the data-integrity signal). Zero-length spans
        (``end == start``) are dropped too but not counted, since they
        contribute nothing either way and are not evidence of a defect.
    worst_gap : float
        Largest ``start - end`` among the inverted spans, in seconds. A lower
        bound on how far the annotation's epochs overrun ``last_second`` (it
        misses the epoch's own length, which the clipping destroyed); used only
        as a fallback when the annotation's epoch list cannot be read.
    """
    kept = []
    n_inverted = 0
    worst_gap = 0.0
    for s, e in times:
        s = float(s)
        e = float(e)
        if e > s:
            kept.append((s, e))
        elif e < s:
            n_inverted += 1
            worst_gap = max(worst_gap, s - e)
    return kept, n_inverted, worst_gap


def _annotation_overrun(annotations, fallback=0.0):
    """How far the scored epochs overrun the annotation's ``last_second``.

    Parameters
    ----------
    annotations : instance of wonambi Annotations (or wrapper)
        Scoring source; must expose ``last_second`` and ``get_epochs``.
    fallback : float, optional
        Overshoot to report when the epoch list or ``last_second`` cannot be
        read (e.g. a stub annotation object). Default ``0.0``.

    Returns
    -------
    last_second : float or None
        The annotation's declared recording end, or ``None`` if unreadable.
    overshoot : float
        ``max(epoch_end) - last_second`` over epochs ending after
        ``last_second``, or ``fallback`` if none do / the lookup fails.
    """
    try:
        last = float(annotations.last_second)
    except Exception:
        return None, fallback
    try:
        ends = [float(ep['end']) for ep in annotations.get_epochs()]
    except Exception:
        return last, fallback
    over = [e - last for e in ends if e > last]
    return last, (max(over) if over else fallback)


def compute_analysed_seconds(annotations, stage, chan=None,
                             reject_types=('Artefact', 'Arousal'),
                             s_freq=None, epoch_len=30,
                             extra_artefact_intervals=None, logger_=None):
    """Compute the artefact-free in-stage time actually fed to a detector.

    This reproduces the segmentation that Wonambi's :func:`wonambi.trans.select.fetch`
    performs during detection, so that an event-density denominator equals the
    exact clean time the detector pooled rather than the (larger) sum of all
    scored epochs of the stage.

    The computation is two steps, mirroring ``fetch`` with
    ``reject_epoch=True, reject_artf=reject_types``:

    1. Select the in-stage epoch spans, dropping epochs whose quality is not
       ``'Good'`` (i.e. ``'Poor'`` / staged ``'Artefact'``). This uses
       :func:`wonambi.trans.select.get_times` with ``exclude=True``.
    2. Subtract, at sample resolution, the seconds overlapped by any event whose
       type is in ``reject_types`` (default ``Artefact``/``Arousal``), using
       :func:`wonambi.trans.reject.remove_artf_evts`. A 2 s artefact inside a
       30 s epoch removes 2 s, not the whole epoch.

    Epoch spans that ``get_times`` clipped into negative length (see Notes) are
    dropped before either step and reported once per call at WARNING level.

    Parameters
    ----------
    annotations : instance of wonambi Annotations (or XLAnnotations/CustomAnnotations)
        Scoring source. Must expose ``get_epochs``, ``get_events`` and
        ``last_second`` (the same object handed to ``fetch`` during detection).
    stage : str
        A single sleep stage label (e.g. ``'NREM2'``). For combined stages,
        call this helper once per stage and sum the returned clean seconds;
        summing per-stage clean time avoids double-counting shared spans.
    chan : str or None, optional
        Channel scope for artefact subtraction, passed straight to
        ``remove_artf_evts``. **Leave this ``None`` (the default) to match the
        detector**, which calls ``fetch`` with no ``chan_full`` and therefore
        subtracts artefacts marked on any channel. Passing a channel name does
        NOT give a clean per-channel denominator with Wonambi's current
        matching (see Notes) and will silently under-subtract, so it is exposed
        only for a future per-channel detection path.
    reject_types : sequence of str or None, optional
        Event types to subtract. Must match what the detection run excluded
        (built from ``reject_artifacts`` / ``reject_arousals``). Default
        ``('Artefact', 'Arousal')``. If ``None`` or empty, no artefact seconds
        are subtracted (only epoch-quality exclusion is applied).
    s_freq : float or None, optional
        Sampling frequency, used only to set the two-sample minimum-segment
        floor (``2 / s_freq``) exactly as ``fetch`` does, so sub-segments too
        short to yield a valid window are dropped identically. If ``None``,
        Wonambi's ``remove_artf_evts`` default floor of ``0.1`` s is used
        (matching the function's own default) rather than 0, so parity holds
        when the sampling rate is unavailable.
    epoch_len : float, optional
        Nominal epoch length in seconds. Accepted for signature symmetry with
        callers/GUI; epoch spans are taken from the annotations directly, so
        this value is not used to fabricate durations. Default ``30``.
    extra_artefact_intervals : iterable of (float, float) or None, optional
        Additional artefact spans in seconds (``(start_sec, end_sec)``) to treat
        as artefact on top of the annotation's ``reject_types`` events. Intended
        for the review GUI, which must subtract the reviewer's LIVE artefact
        marks (held in the ``qc_artefact_intervals`` DB table, not yet exported
        to the annotation) so marking an epoch immediately shrinks that stage's
        analysed time. The spans are unioned with the annotation's reject spans
        before subtraction (overlaps with an existing annotation artefact are NOT
        double-subtracted), clipped to the in-stage, in-epoch time (a span
        straddling an epoch boundary removes only its in-stage portion; a span
        outside every scored epoch of ``stage`` removes nothing), and merged with
        one another so overlapping/adjacent marks are not double-counted. The
        same ``min_dur`` floor as the annotation subtraction is applied to
        surviving fragments. ``None`` or empty reproduces the export-path
        behaviour exactly (no extra subtraction). Default ``None``.
    logger_ : logging.Logger or None, optional
        Logger for the annotation-inconsistency warning (see Notes), so it
        lands in the calling processor's log file alongside the density
        messages. ``None`` (default) uses this module's
        ``turtlewave_hdEEG.utils`` logger. Passing ``None`` never silences the
        warning: a data-integrity condition is always reported somewhere.

    Returns
    -------
    analysed_seconds : float
        Artefact-free in-stage time in seconds (the density denominator, ×60
        for minutes). ``0.0`` when the stage has no Good epochs or is fully
        covered by artefact.
    artefact_seconds_excluded : float
        In-stage Good seconds removed by artefact/arousal subtraction
        (``in_stage_good_seconds - analysed_seconds``). Emitted for provenance.

    Notes
    -----
    **Epochs scored past ``last_second``.** ``get_times`` clips every epoch end
    to ``annot.last_second`` (``min(e['end'], last)``), so an epoch scored
    beyond the annotation's declared recording end comes back as
    ``(start, last)`` with ``last < start`` -- a span of negative length.
    Such spans are dropped here (they contribute zero), and the condition is
    logged once per call at WARNING with the count and the worst overrun in
    seconds, because it means the annotation's ``last_second`` disagrees with
    its own scored epochs.

    Not guarding this is not a small systematic offset. Downstream,
    ``remove_artf_evts`` returns ``times`` untouched when the file contains no
    matching reject event in range, but rebuilds the span list under a
    ``min_dur`` floor (which discards negative-length spans) when it does. So
    the pre-guard denominator was computed by a different rule depending on
    whether the file happened to contain a matching artefact/arousal event --
    silent, per-file, and invisible in the exported numbers.

    With ``chan=None`` this returns exactly the seconds the detector pooled
    (verified against :func:`wonambi.trans.select.fetch` with
    ``reject_epoch=True, reject_artf=reject_types``), because detection also
    passes no ``chan_full`` and so removes artefacts channel-globally. The
    resulting density is per-channel only through its numerator (per-channel
    event counts); the denominator is shared across channels, which is correct
    since the detector fed every channel the same channel-global clean time.

    A true per-channel denominator is not available from Wonambi's
    ``remove_artf_evts`` as-is: it joins ``(chan, '')`` into the string
    ``'chan, '`` and matches only events whose ``event_chan`` equals that
    literal string, so a channel-global (``''``) artefact is skipped and a
    single-channel mark (``'chan'``) does not match either. Making per-channel
    denominators meaningful requires detection to fetch per channel
    (``chan_full=[chan]``) and a matching artefact-marking convention; that is
    out of scope here and tracked as a follow-up.

    This never raises on empty input and never divides; callers guard the
    division (``analysed_seconds == 0`` -> density 0).
    """
    from wonambi.trans.select import get_times
    from wonambi.trans.reject import remove_artf_evts

    # Step 1: in-stage Good epochs (exclude=True mirrors reject_epoch=True).
    try:
        bundles = get_times(annotations, evt_type=None, stage=[stage],
                            cycle=None, chan=None, exclude=True)
    except Exception:
        return 0.0, 0.0

    if not bundles:
        return 0.0, 0.0

    # Two-sample floor, identical to fetch()'s two_sample_dur. When s_freq is
    # unknown, fall back to remove_artf_evts's own default (0.1 s), not 0, so
    # byte-parity with a fetch()/remove_artf_evts run using that default holds.
    if s_freq and s_freq > 0:
        min_dur = 2.0 / float(s_freq)
    else:
        min_dur = 0.1

    reject_list = [str(r) for r in reject_types] if reject_types else None

    # Reviewer's LIVE artefact marks: merge into disjoint spans once, then remove
    # from each bundle's already-artefact-free segments. Subtracting from the
    # post-`remove_artf_evts` segments (rather than the raw epoch times) makes the
    # union with annotation artefacts implicit: a region an annotation artefact
    # already removed is absent from `kept`, so an overlapping extra span removes
    # nothing there -- no double subtraction.
    extra_spans = _merge_intervals(extra_artefact_intervals) \
        if extra_artefact_intervals else []

    in_stage_seconds = 0.0
    clean_seconds = 0.0
    n_inverted = 0
    worst_gap = 0.0
    for bund in bundles:
        times = bund.get('times') or []
        # Drop epoch spans that get_times clipped into negative length before
        # any arithmetic: a negative duration must contribute zero, never
        # subtract from the analysed total. Must precede remove_artf_evts,
        # which drops them only when the file happens to contain a matching
        # reject event (its rebuild path applies a min_dur floor) and passes
        # them straight through when it does not -- so without this guard the
        # denominator is computed by a different rule per file.
        times, n_bad, gap = _drop_inverted_spans(times)
        n_inverted += n_bad
        worst_gap = max(worst_gap, gap)
        if not times:
            continue
        in_stage_seconds += sum(float(e) - float(s) for s, e in times)

        if reject_list:
            kept = remove_artf_evts(times, annotations, chan=chan,
                                    name=reject_list, min_dur=min_dur)
        else:
            kept = times
        if extra_spans:
            kept = _subtract_spans(kept, extra_spans, min_dur)
        clean_seconds += sum(float(e) - float(s) for s, e in kept)

    if n_inverted:
        # One line per call, not per span: the count and the worst overshoot
        # say everything a user needs to go and fix the annotation.
        last_second, overshoot = _annotation_overrun(annotations,
                                                     fallback=worst_gap)
        (logger_ or logger).warning(
            "Annotation is inconsistent: %d scored %s epoch span(s) lie past "
            "the annotation's declared recording end (last_second=%s), which "
            "the scored epochs overrun by up to %.1f s. Those spans are "
            "counted as zero analysed time (they cannot be analysed -- there "
            "is no signal declared there), so this stage's density "
            "denominator covers only the epochs inside last_second. Check "
            "that last_second matches the recording length and that the "
            "trailing epochs are really scored data.",
            n_inverted, stage,
            'unknown' if last_second is None else f"{last_second:g}",
            overshoot)

    artefact_seconds_excluded = max(in_stage_seconds - clean_seconds, 0.0)
    return clean_seconds, artefact_seconds_excluded


class DensityDenominators:
    """Artefact-free density denominators shared by the event-density exporters.

    Bundles everything the spindle and slow-wave density exporters need to turn
    artefact-free time into density, so the computation lives in one place and
    cannot drift between processors:

    * per-stage artefact-free seconds/minutes (cached, channel-global);
    * the set of stages actually detected on (the density time base);
    * the whole-night artefact-free minutes summed over those detected stages
      (Wake excluded unless Wake was itself a detection stage);
    * a per-channel whole-night event count restricted to the detected stages.

    The denominator is channel-global (``chan=None``) on purpose: detection's
    ``fetch`` passes no ``chan_full`` and removes artefacts channel-globally, so
    this equals the exact clean time the detector pooled. Density is per-channel
    only through its numerator (per-channel event counts). See
    :func:`compute_analysed_seconds` for the parity details.

    Prefer :func:`build_density_denominators` to construct instances; it also
    resolves ``s_freq`` and emits the reject-type assumption warning.

    Parameters
    ----------
    annotations : instance of wonambi Annotations (or XLAnnotations/CustomAnnotations)
        Scoring source handed to the detector.
    s_freq : float or None
        Sampling frequency for the two-sample minimum-segment floor.
    reject_types : list of str
        Event types subtracted from the denominator (e.g. ``['Artefact', 'Arousal']``).
        Must mirror what the detection run excluded.
    stage_list : list of str or None
        The requested detection stage(s). When ``None``, the detected stages are
        taken from ``stages_present`` instead.
    stages_present : iterable of str
        Stages present in the loaded events; used as the detected-stage fallback
        when ``stage_list`` is ``None``.
    epoch_len : float, optional
        Nominal epoch length in seconds, forwarded to
        :func:`compute_analysed_seconds`. Default ``30``.
    extra_artefact_intervals : iterable of (float, float) or None, optional
        Extra artefact spans in seconds forwarded unchanged to
        :func:`compute_analysed_seconds` for every stage (per-stage and
        whole-night), so the review GUI's live artefact marks shrink both the
        per-stage and whole-night denominators consistently. ``None`` (the
        export-path default) reproduces the annotation-only denominators.
        Default ``None``.
    logger_ : logging.Logger or None, optional
        Logger forwarded to :func:`compute_analysed_seconds` for its
        annotation-inconsistency warning, so that warning reaches the calling
        processor's log file rather than only stderr. ``None`` (default) uses
        the ``turtlewave_hdEEG.utils`` module logger. Default ``None``.

    Attributes
    ----------
    reject_types : list of str
        The reject types used (for provenance).
    detected_stage_set : set of str
        Individual stages that define the whole-night time base.
    whole_night_analysed_min : float
        Artefact-free minutes summed over ``detected_stage_set``.
    """

    def __init__(self, annotations, s_freq, reject_types, stage_list,
                 stages_present, epoch_len=30, extra_artefact_intervals=None,
                 logger_=None):
        self._annot = annotations
        self._s_freq = s_freq
        self._epoch_len = epoch_len
        self._logger = logger_
        self.reject_types = list(reject_types) if reject_types else []
        # Fixed for this instance, so the per-stage cache stays valid.
        self._extra_artefact_intervals = (list(extra_artefact_intervals)
                                          if extra_artefact_intervals else None)
        self._cache = {}

        # Detected stages define the density time base. Prefer the requested
        # stage(s); else fall back to the stages present in the events. Wake
        # appears here only if it was itself a detection stage, so it is
        # excluded from whole-night unless detected (no special-casing needed).
        if stage_list is None:
            self.detected_stage_set = set(str(s) for s in stages_present)
        else:
            self.detected_stage_set = set(str(s) for s in stage_list)

        wn_sec = 0.0
        for stg in self.detected_stage_set:
            wn_sec += self.analysed_seconds(stg)[0]
        self.whole_night_analysed_min = wn_sec / 60.0

    def analysed_seconds(self, stage):
        """Return ``(clean_seconds, artefact_seconds_excluded)`` for one stage.

        Cached per stage; the denominator is channel-global by design.

        Parameters
        ----------
        stage : str
            A single sleep stage label.

        Returns
        -------
        tuple of float
            ``(clean_seconds, artefact_seconds_excluded)``.
        """
        key = str(stage)
        if key not in self._cache:
            self._cache[key] = compute_analysed_seconds(
                self._annot, key, chan=None, reject_types=self.reject_types,
                s_freq=self._s_freq, epoch_len=self._epoch_len,
                extra_artefact_intervals=self._extra_artefact_intervals,
                logger_=self._logger)
        return self._cache[key]

    def whole_night_count(self, chan_events):
        """Count a channel's events that fall in the detected stages.

        Restricting to the detected stages keeps the whole-night numerator on
        the same time base as :attr:`whole_night_analysed_min`.

        Parameters
        ----------
        chan_events : list of dict
            Detected events for one channel. Each has a ``'stage'``, which may
            be a single label, a list of labels, or -- from 4.3 -- the run's
            joint stage token (``'NREM2NREM3'``). The token is split into its
            components before the comparison; comparing it whole would
            intersect nothing and report every channel's whole-night count as
            zero.

        Returns
        -------
        int
            Number of events whose stage(s) intersect the detected stages.
        """
        from .dbwrite import stage_components
        n = 0
        for ev in chan_events:
            st = ev.get('stage')
            st = st if isinstance(st, (list, tuple)) else [st]
            components = [c for value in st if value is not None
                          for c in stage_components(str(value))]
            if self.detected_stage_set.intersection(components):
                n += 1
        return n


def build_density_denominators(annotations, dataset, reject_artifacts=None,
                               reject_arousals=None, stage_list=None,
                               stages_present=(),
                               logger=None, epoch_len=30,
                               extra_artefact_intervals=None):
    """Build a :class:`DensityDenominators` for the artefact-free denominator.

    Factors the shared setup out of the density exporters: it derives the
    reject-type list from the two flags, resolves ``s_freq`` from the dataset
    header, and records which event types are subtracted from the denominator
    so that choice is never silent.

    ``reject_artifacts`` and ``reject_arousals`` accept ``None`` meaning "not
    specified by the caller". In that case they default to True (the detector
    defaults) and a warning is logged, because an unspecified value that does
    not match the detection run gives a denominator covering the wrong amount
    of time. Passing them explicitly states that they match the run and
    downgrades the message to an informational line.

    Parameters
    ----------
    annotations : instance of wonambi Annotations (or wrapper)
        Scoring source handed to the detector.
    dataset : instance of Dataset
        Used only to read ``header['s_freq']``; missing/failed lookups fall back
        to Wonambi's default minimum-segment floor.
    reject_artifacts : bool or None, optional
        Subtract 'Artefact' time from the denominator. Must match the detection
        run. ``None`` (the default) means the caller did not specify it: True
        is assumed and a warning is logged. Default ``None``.
    reject_arousals : bool or None, optional
        Subtract 'Arousal' time from the denominator. Must match the detection
        run. ``None`` (the default) means the caller did not specify it: True
        is assumed and a warning is logged. Default ``None``.
    stage_list : list of str or None
        Requested detection stage(s); ``None`` means "use stages_present".
    stages_present : iterable of str
        Stages present in the loaded events (detected-stage fallback).
    logger : logging.Logger or None, optional
        Logger for the reject-type message, also forwarded to
        :func:`compute_analysed_seconds` so its annotation-inconsistency
        warning lands in the same log. If ``None``, the reject-type message is
        not emitted (the choice is still recorded in the returned object's
        ``reject_types``); the annotation-inconsistency warning is *not*
        suppressed, it falls back to the ``turtlewave_hdEEG.utils`` logger.
    epoch_len : float, optional
        Nominal epoch length in seconds. Default ``30``.
    extra_artefact_intervals : iterable of (float, float) or None, optional
        Extra artefact spans in seconds forwarded to the
        :class:`DensityDenominators`, so the review GUI's live artefact marks
        subtract from every per-stage and whole-night denominator. ``None`` (the
        export-path default) leaves the denominators annotation-only. Default
        ``None``.

    Returns
    -------
    DensityDenominators
        Configured denominator helper.

    Notes
    -----
    The rejection settings are taken from this call, not read back from the
    detection run's stored settings, so they are only as correct as what the
    caller passes. If the run overrode ``reject_artifacts`` or
    ``reject_arousals``, pass matching values here; otherwise the denominator
    subtracts a different amount of time from the one the detector actually
    analysed, and every density derived from it is biased.
    """
    assumed = reject_artifacts is None or reject_arousals is None
    if reject_artifacts is None:
        reject_artifacts = True
    if reject_arousals is None:
        reject_arousals = True

    reject_types = []
    if reject_artifacts:
        reject_types.append('Artefact')
    if reject_arousals:
        reject_types.append('Arousal')

    s_freq = None
    try:
        s_freq = dataset.header['s_freq']
    except Exception:
        if logger is not None:
            logger.debug("Could not read s_freq from dataset header; artefact-free "
                         "time uses Wonambi's default 0.1 s min-segment floor.")

    if logger is not None:
        subtracted = " and ".join(reject_types) if reject_types else "nothing"
        if assumed:
            logger.warning(
                "Density denominator: assuming the detection run excluded "
                "artefact and arousal epochs (reject_artifacts=True, "
                "reject_arousals=True), so %s time is subtracted from the "
                "recording time each density is divided by. If your detection "
                "run used different settings, pass reject_artifacts= and "
                "reject_arousals= to match it, otherwise the densities will "
                "be biased.", subtracted)
        else:
            logger.info(
                "Density denominator: subtracting %s time from the recording "
                "time (reject_artifacts=%s, reject_arousals=%s, as specified "
                "by the caller).", subtracted, reject_artifacts, reject_arousals)

    return DensityDenominators(annotations, s_freq, reject_types, stage_list,
                               stages_present, epoch_len=epoch_len,
                               extra_artefact_intervals=extra_artefact_intervals,
                               logger_=logger)


def warn_csv_import_deprecated(what, logger_=None):
    """Emit the deprecation notice for the CSV -> database importers.

    Detection writes straight into ``neural_events.db``, so the
    JSON -> CSV -> import round-trip only exists for the legacy
    ``write_db=False`` path and for recovering historical files. Every
    filename round-trip in this pipeline has cost data at least once (the band
    token, the slashed method), which is why the route is going away.

    Both a :class:`DeprecationWarning` **and** a ``logger.warning`` are
    emitted: deprecation warnings are invisible by default and these users run
    scripts, not test suites.

    Parameters
    ----------
    what : str
        Name of the deprecated method, used in the message.
    logger_ : logging.Logger or None, optional
        Logger to warn on. ``None`` (default) uses the
        ``turtlewave_hdEEG.utils`` module logger.

    Returns
    -------
    None
    """
    import warnings
    msg = (
        f"{what} is deprecated and will be removed in 5.0. Detection writes "
        f"events straight into neural_events.db since 4.2, so this CSV import "
        f"step is only needed for the legacy write_db=False path and for "
        f"recovering historical CSV files. Run detection without "
        f"write_db=False (or without --legacy-json) and the events are already "
        f"in the database.")
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    (logger_ or logger).warning(msg)


def warn_density_csv_deprecated(what, logger_=None):
    """Emit the deprecation notice for the JSON-backed density exporters.

    Density now comes from the database
    (:func:`turtlewave_hdEEG.density.event_density`), whose denominator is the
    ``analysed_time`` row the detection run stored. The
    ``export_*_density_to_csv`` methods read per-channel JSON, which detection
    no longer writes unless it is run on the legacy ``write_db=False`` path.

    Both a :class:`DeprecationWarning` **and** a ``logger.warning`` are
    emitted: deprecation warnings are invisible by default, and these users
    run scripts rather than test suites, so a warnings-only notice would never
    be seen.

    Parameters
    ----------
    what : str
        Name of the deprecated method, used in the message.
    logger_ : logging.Logger or None, optional
        Logger to warn on. ``None`` (default) uses the
        ``turtlewave_hdEEG.utils`` module logger.

    Returns
    -------
    None
    """
    import warnings
    msg = (
        f"{what} is deprecated and will be removed in 5.0. It reads the "
        f"per-channel JSON files that detection no longer writes (the "
        f"database is the store of record since 4.2). Use "
        f"turtlewave_hdEEG.density.event_density(db_path, ...) instead, which "
        f"derives density from neural_events.db using the artefact-free "
        f"analysed_time denominator the detection run stored. This method "
        f"still works against a legacy JSON directory produced with "
        f"write_db=False.")
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    (logger_ or logger).warning(msg)


# Function to read channels from CSV file
def read_channels_from_csv(csv_file_path):
    channels = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                # Check if the first cell contains a channel name
                if row and row[0].strip():  # Only add non-empty values
                    channels.append(row[0].strip())
        
        logger.info(f"Found {len(channels)} channels in {csv_file_path}")
        logger.debug(f"Channels read from CSV: {channels}")

        return channels
    except Exception as e:
        logger.error(f"Error reading channel CSV {csv_file_path}: {e}")
        return None
