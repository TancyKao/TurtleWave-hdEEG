"""Correctness guards for scoped channel re-detection (P3).

The review GUI queues bad channels for re-detection with the reviewer's marked
artefact epochs excluded, then a driver re-runs detection on *only* those
channels and replaces their rows in ``neural_events.db`` (see
:func:`turtlewave_hdEEG.dbwrite.write_channel_events` with ``replace=True``).

Artefact-epoch rejection itself needs no new code: detection calls
``fetch(..., reject_epoch=True, reject_artf=['Artefact', 'Arousal'])`` and
Wonambi excises the marked spans from the signal BEFORE the detector's threshold
pooling, so exclusion is correct at ESTIMATION time (re-fetch, not
detect-then-delete). What this module adds are the guards that make a re-run
scientifically defensible rather than merely runnable:

* :func:`verify_rater_match` -- the detector rejects artefacts only from the
  rater it reads (``raters[0]``). If the sidecar's ``Artefact`` events live under
  a different rater than the staging, the detector would reject NOTHING and
  silently emit a still-contaminated re-run. This fails loudly instead.
* :func:`channel_clean_gate` -- a global-threshold detector over too little or
  too-fragmented clean data is as unstable as one over contaminated data, so a
  channel with less than ``n_min_sec`` of artefact-free in-stage time, or more
  than ``max_excluded_frac`` of its in-stage time excluded, is forced to DROP
  rather than re-detected.
* :func:`resolve_rerun_params` -- a re-run MUST reuse the original run's
  ``ref_chan`` / ``polar`` / ``cat`` (a wrong ``polar`` inverts trough polarity,
  the same failure axis as the PAC 180 degree bug). This resolves them from the
  recorded provenance or explicit arguments, and REFUSES when neither is
  available.

None of these guards touch the signal path; they gate whether a channel is
re-detected at all and with which invariant parameters.

Notes
-----
The clean-time gate is channel-GLOBAL, not truly per-channel: it reuses
:func:`turtlewave_hdEEG.utils.compute_analysed_seconds` with ``chan=None`` (the
only artefact subtraction Wonambi supports faithfully; see that function's
Notes). When the reviewer's artefact marks are whole-montage (as the current
review-GUI sidecar writes them, ``chan='(all)'``), the gate therefore evaluates
identically for every re-detect channel -- it protects against a globally
over-contaminated re-run, not against a single channel being individually
fragmented. A per-channel gate needs per-channel artefact marking and a
per-channel fetch, tracked as a follow-up.
"""

import logging

from .utils import compute_analysed_seconds
from . import dbwrite


class RerunGuardError(Exception):
    """A re-run pre-flight guard refused to proceed.

    Raised (not logged-and-continued) so a scoped re-detection never silently
    produces contaminated or polarity-inverted output.
    """


def verify_rater_match(annotations, reject_types, logger=None):
    """Ensure the detector's rater carries both the staging and the artefacts.

    The detector reads a single rater (Wonambi/``CustomAnnotations`` auto-select
    ``raters[0]``) and rejects artefacts only from THAT rater. If the sidecar's
    ``Artefact``/``Arousal`` events were appended under a different rater than
    the one holding the staging, the detector would find no artefacts to reject
    and emit a still-contaminated re-run with no error. This checks, before any
    detection, that the rater the detector will read contains BOTH the staged
    epochs AND every reject-type event that exists anywhere in the file.

    Parameters
    ----------
    annotations : CustomAnnotations or wonambi Annotations
        The annotation object handed to the detector. Its currently selected
        rater is the one the detector will read.
    reject_types : sequence of str
        Event types the run will reject (e.g. ``['Artefact', 'Arousal']``).
        Empty/``None`` means the run rejects nothing, so only the staging
        presence is checked.
    logger : logging.Logger or None, optional
        Logger for a warning when a reject type is absent from EVERY rater (a
        clean file -- not an error, but worth noting for a re-run whose purpose
        is to exclude reviewer artefacts).

    Raises
    ------
    RerunGuardError
        When the detector's rater has no staged epochs, or when a reject-type
        event exists under some OTHER rater but not under the detector's rater
        (the silent-no-rejection trap).
    """
    log = logger or logging.getLogger('turtlewave_hdEEG.rerun')

    raters = list(getattr(annotations, 'raters', []) or [])
    if not raters:
        raise RerunGuardError(
            "Annotation file has no rater; the detector cannot read staging or "
            "artefacts. Refusing to re-detect.")

    # ``annotations.rater`` is the currently-selected rater XML Element (not its
    # name); resolve it to the name string the detector effectively reads.
    rater_obj = getattr(annotations, 'rater', None)
    if rater_obj is None:
        detector_rater = None
    elif isinstance(rater_obj, str):
        detector_rater = rater_obj
    else:
        detector_rater = rater_obj.get('name')
    if not detector_rater:
        # Mirror CustomAnnotations' auto-select so the check matches detection.
        detector_rater = raters[0]
    try:
        annotations.get_rater(detector_rater)
    except Exception:
        pass

    def _events_under(rater, name):
        """Count events of ``name`` under ``rater``.

        A READ FAILURE is raised as a :class:`RerunGuardError`, never swallowed
        into ``0`` -- otherwise a genuine error would be indistinguishable from a
        legitimate empty result and could downgrade the guard to a mere warning,
        letting a re-run that rejects nothing proceed. Wonambi's ``get_events``
        returns ``[]`` (not an error) for an absent event type, so a raised
        exception here really is a read failure.
        """
        try:
            annotations.get_rater(rater)
            events = annotations.get_events(name=name)
        except Exception as e:
            raise RerunGuardError(
                f"Failed to read '{name}' events under rater '{rater}': {e}. "
                f"Refusing to re-detect rather than assume zero artefacts.")
        return len(events or [])

    # (1) staging present under the detector's rater?
    try:
        annotations.get_rater(detector_rater)
        n_epochs = len(annotations.get_epochs() or [])
    except Exception as e:
        raise RerunGuardError(
            f"Could not read epochs under detector rater '{detector_rater}': "
            f"{e}. Refusing to re-detect.")
    if n_epochs == 0:
        raise RerunGuardError(
            f"Detector rater '{detector_rater}' has no staged epochs; the "
            f"re-run would have no stages to detect in. Ensure the staging and "
            f"the artefacts are under the same rater.")

    # (2) every reject type present under the detector's rater if present at all.
    for rt in (reject_types or []):
        n_detector = _events_under(detector_rater, rt)
        n_anywhere = max(_events_under(r, rt) for r in raters)
        if n_detector == 0 and n_anywhere > 0:
            other = [r for r in raters if _events_under(r, rt) > 0]
            # Restore the detector's rater before raising.
            try:
                annotations.get_rater(detector_rater)
            except Exception:
                pass
            raise RerunGuardError(
                f"'{rt}' events exist under rater(s) {other} but NOT under the "
                f"rater the detector reads ('{detector_rater}'). The re-run "
                f"would reject no '{rt}' and silently stay contaminated. Put "
                f"the artefacts under '{detector_rater}' (the sidecar must "
                f"append them to the rater that holds the staging).")
        if n_anywhere == 0:
            log.warning(
                "No '%s' events under any rater; the re-run will reject none. "
                "Confirm the reviewer's artefact marks were written to the "
                "sidecar.", rt)

    # Leave the detector's rater selected.
    try:
        annotations.get_rater(detector_rater)
    except Exception:
        pass


def channel_clean_gate(annotations, stages, s_freq=None, n_min_sec=300.0,
                       max_excluded_frac=0.5, reject_types=('Artefact', 'Arousal'),
                       extra_artefact_intervals=None):
    """Decide whether a channel has enough clean data to re-detect.

    A global-threshold detector estimates its threshold by pooling the
    artefact-free signal; over too little or too-fragmented clean data that
    estimate is unstable, so such a channel should be DROPPED, not re-detected.

    Parameters
    ----------
    annotations : CustomAnnotations or wonambi Annotations
        Scoring source (same object handed to the detector).
    stages : sequence of str
        Detection stages; clean seconds are summed over them (per-stage, to
        avoid double-counting shared spans).
    s_freq : float or None, optional
        Sampling frequency for the two-sample minimum-segment floor, matching
        the detector's ``fetch``. ``None`` uses Wonambi's 0.1 s default floor.
    n_min_sec : float, optional
        Minimum artefact-free in-stage seconds required to re-detect. Default
        ``300`` (5 minutes). Recorded in provenance by the caller.
    max_excluded_frac : float, optional
        Maximum fraction of in-stage Good time that may be excluded as
        artefact. Above this the channel is dropped even if ``n_min_sec`` is
        met. Default ``0.5``.
    reject_types : sequence of str, optional
        Artefact event types subtracted (must match the run). Default
        ``('Artefact', 'Arousal')``.
    extra_artefact_intervals : iterable of (float, float) or None, optional
        Extra artefact spans (seconds) to subtract on top of the annotation's
        reject events, forwarded to :func:`compute_analysed_seconds`. Reserved
        for a future per-channel gate; ``None`` (default) reproduces the
        annotation-only, channel-global behaviour.

    Returns
    -------
    dict
        ``{'ok': bool, 'clean_sec': float, 'in_stage_sec': float,
        'excluded_frac': float, 'reason': str}``. ``ok`` is True only when there
        is at least ``n_min_sec`` clean time AND at most ``max_excluded_frac``
        excluded. ``reason`` is empty when ``ok`` else the drop rationale.
    """
    clean_sec = 0.0
    excluded_sec = 0.0
    for stg in stages:
        c, x = compute_analysed_seconds(
            annotations, stg, chan=None, reject_types=reject_types,
            s_freq=s_freq, extra_artefact_intervals=extra_artefact_intervals)
        clean_sec += c
        excluded_sec += x
    in_stage_sec = clean_sec + excluded_sec
    excluded_frac = (excluded_sec / in_stage_sec) if in_stage_sec > 0 else 1.0

    reasons = []
    if clean_sec < n_min_sec:
        reasons.append(
            f"only {clean_sec / 60.0:.2f} min clean in-stage time "
            f"(< {n_min_sec / 60.0:.2f} min minimum)")
    if excluded_frac > max_excluded_frac:
        reasons.append(
            f"{excluded_frac * 100:.1f}% of in-stage time excluded "
            f"(> {max_excluded_frac * 100:.0f}% maximum)")
    ok = not reasons
    return {
        'ok': ok,
        'clean_sec': clean_sec,
        'in_stage_sec': in_stage_sec,
        'excluded_frac': excluded_frac,
        'reason': '; '.join(reasons),
    }


#: Slow-wave methods whose published criteria contain no absolute microvolt
#: amplitude floor (Ngo thresholds at 1.25x the mean, Staresina at the 75th
#: percentile). Before 4.3.0 a recorded ``neg_peak_thresh``/``p2p_thresh`` for
#: these was compared against a SAMPLE-INDEX distance, never microvolts.
RELATIVE_SW_METHODS = ('Ngo2015', 'Staresina2015')

#: ``(neg_peak_thresh, p2p_thresh)`` pairs that the shipped entry points
#: injected for :data:`RELATIVE_SW_METHODS` before 4.3.0, none of which the
#: caller chose as a microvolt criterion:
#:
#: * ``(-80.0, 140.0)`` -- ``ParalSWA.detect_slow_waves``'s substitution for
#:   ``None``, and the value ``turtlewave_gui`` sent for Ngo2015;
#: * ``(-20.0, 40.0)`` -- ``hdEEG_sw_detector_GADI.py``'s non-Massimini default;
#: * ``(-75.0, 75.0)`` -- ``turtlewave_gui``'s Staresina2015 tab, where the 75
#:   is the PERCENTILE widget's value forwarded into a microvolt argument.
LEGACY_INERT_SW_THRESHOLDS = frozenset({
    (-80.0, 140.0),
    (-20.0, 40.0),
    (-75.0, 75.0),
})

#: First release in which a recorded slow-wave amplitude threshold genuinely
#: means microvolts for :data:`RELATIVE_SW_METHODS`.
FLOOR_FIX_VERSION = (4, 3, 0)


def _version_tuple(version):
    """Parse ``'4.3.0'`` into ``(4, 3, 0)`` for ordering, or ``None``.

    Parameters
    ----------
    version : str or None
        A recorded ``turtlewave_version``.

    Returns
    -------
    tuple of int or None
        The leading dotted integer components, or ``None`` when the string is
        absent or not parseable (a pre-provenance run, or a dev tag).
    """
    if not version:
        return None
    parts = []
    for chunk in str(version).split('.'):
        digits = ''
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


def resolve_sw_amplitude_thresholds(method, neg_peak_thresh, p2p_thresh,
                                    recorded_version=None, logger=None):
    """Read recorded slow-wave amplitude thresholds in their original meaning.

    Before 4.3.0, ``ImprovedDetectSlowWave`` applied these two thresholds to
    Ngo2015/Staresina2015 through a filter that ran BEFORE the event fields
    were converted to microvolts, so ``p2p_thresh`` was compared against
    Wonambi's ``ptp`` -- a sample-index distance -- and ``neg_peak_thresh`` was
    tested as ``abs(trough_val) >= thresh``, which a negative value can never
    fail. Replaying such a recorded value as the microvolt floor it now means
    would impose a criterion the original run never applied.

    The policy, applied only to :data:`RELATIVE_SW_METHODS` and only to runs
    recorded before :data:`FLOOR_FIX_VERSION`:

    * a pair matching a known legacy default
      (:data:`LEGACY_INERT_SW_THRESHOLDS`) resolves to ``(None, None)`` -- the
      published criteria -- and says so;
    * a lone negative ``neg_peak_thresh`` resolves to ``None`` whatever its
      value, because ``abs(x) >= a negative number`` is unconditionally true:
      that arm was provably a no-op, independent of signal and sampling rate;
    * anything else is AMBIGUOUS and is kept, with a loud warning that it will
      now bind in microvolts when it previously did not.

    The Massimini family is returned untouched: its thresholds were always
    enforced inside the detector, in microvolts.

    .. note::

       Dropping the floor does not reproduce the original run bit-for-bit
       where it was binding. The sample-count comparison rejected nothing at
       ~500 Hz but everything below ~200 Hz, so a low-rate recording's
       original event set was smaller than the published criteria give. That
       run was wrong; this returns the criteria the method actually defines.

    Parameters
    ----------
    method : str
        Slow-wave method the re-run will use.
    neg_peak_thresh, p2p_thresh : float or None
        The values recovered from the original run's ``params_json``.
    recorded_version : str or None, optional
        ``turtlewave_version`` of the original run, from
        :func:`turtlewave_hdEEG.dbwrite.recover_run_scope`. ``None`` is treated
        as pre-fix, which is the safe reading: every database that predates the
        column also predates the fix.
    logger : logging.Logger or None, optional
        Logger for the resolution message.

    Returns
    -------
    tuple
        ``(neg_peak_thresh, p2p_thresh)`` to pass to the re-run.
    """
    log = logger or logging.getLogger('turtlewave_hdEEG.rerun')

    if method not in RELATIVE_SW_METHODS:
        return neg_peak_thresh, p2p_thresh
    if neg_peak_thresh is None and p2p_thresh is None:
        return None, None

    version = _version_tuple(recorded_version)
    if version is not None and version >= FLOOR_FIX_VERSION:
        log.info(
            "Re-run %s: recorded neg_peak_thresh=%r / p2p_thresh=%r come from "
            "a %s run, i.e. after the microvolt-floor fix, so they are a "
            "deliberate uV criterion and are replayed unchanged.",
            method, neg_peak_thresh, p2p_thresh, recorded_version)
        return neg_peak_thresh, p2p_thresh

    try:
        pair = (None if neg_peak_thresh is None else float(neg_peak_thresh),
                None if p2p_thresh is None else float(p2p_thresh))
    except (TypeError, ValueError):
        # A params_json that does not hold numbers here is not something to
        # guess at: keep the recorded values and let the detector reject them,
        # rather than crash the re-run inside a provenance helper.
        log.warning(
            "Re-run %s: recorded neg_peak_thresh=%r / p2p_thresh=%r are not "
            "numeric, so they cannot be checked against the pre-4.3.0 "
            "defaults; passing them through unchanged.",
            method, neg_peak_thresh, p2p_thresh)
        return neg_peak_thresh, p2p_thresh

    if pair in LEGACY_INERT_SW_THRESHOLDS:
        log.info(
            "Re-run %s: recorded neg_peak_thresh=%r / p2p_thresh=%r predate "
            "4.3.0's amplitude-floor fix and were never a microvolt criterion "
            "(they were compared against a sample count); replaying with the "
            "published criteria instead, i.e. no absolute uV floor.",
            method, neg_peak_thresh, p2p_thresh)
        return None, None

    neg_out, p2p_out = neg_peak_thresh, p2p_thresh
    if neg_out is not None and float(neg_out) < 0:
        log.info(
            "Re-run %s: recorded neg_peak_thresh=%r was provably inert before "
            "4.3.0 (abs(trough_val) >= a negative number is always true), so "
            "it is dropped rather than replayed as a %.1f uV depth floor.",
            method, neg_out, abs(float(neg_out)))
        neg_out = None

    if neg_out is not None or p2p_out is not None:
        log.warning(
            "Re-run %s: recorded neg_peak_thresh=%r / p2p_thresh=%r is not a "
            "known pre-4.3.0 default, so neg_peak_thresh=%r / p2p_thresh=%r "
            "are KEPT -- but they will now bind as ABSOLUTE MICROVOLT floors, "
            "which they did not in the original run (p2p was compared against "
            "a sample-index distance). Neither Ngo et al. 2015 nor Staresina "
            "et al. 2015 defines a uV amplitude criterion. Pass them "
            "explicitly as None to re-run on the published criteria.",
            method, neg_peak_thresh, p2p_thresh, neg_out, p2p_out)

    return neg_out, p2p_out


def resolve_rerun_params(db_path, event_type, method, freq_lower=None,
                         freq_upper=None, ref_chan=None, polar=None, cat=None,
                         logger=None):
    """Resolve the invariant re-run parameters, reusing the original run's.

    A re-run must reuse the original run's ``ref_chan``, ``polar`` and ``cat``,
    never a driver/GUI default: a different reference makes every amplitude
    threshold incomparable, and a wrong ``polar`` inverts trough polarity. This
    prefers an explicitly supplied value, else the value recorded for the most
    recent matching run in ``detection_runs``; if a parameter is available from
    NEITHER source it REFUSES rather than guess.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database with the ``detection_runs`` provenance.
    event_type, method : str
        Detection scope to recover the original run for.
    freq_lower, freq_upper : float or None, optional
        Band bounds; used to disambiguate a same-method run at a different band.
    ref_chan, polar, cat : optional
        Explicit overrides. When given, they take precedence over the recovered
        values (and a warning is logged if they DIFFER from what was recorded,
        since re-running with a changed reference/polarity/cat is a scientific
        change, not a tuning). ``cat`` is treated strictly: it lives only in
        ``params_json``, so a pre-P3 run never recorded it; when it is neither
        supplied here nor recorded, this REFUSES (raises) rather than assume a
        default, because production runs concatenate (``cat=(1, 1, 1, 0)``) and a
        wrong ``cat`` pools the signal differently and shifts every threshold.
    logger : logging.Logger or None, optional
        Logger for the recovery/override messages.

    Returns
    -------
    dict
        ``{'ref_chan', 'polar', 'cat', 'recovered'}`` -- the effective values
        plus the raw recovered dict (or ``None`` if no run was on record).

    Raises
    ------
    RerunGuardError
        When ``ref_chan`` or ``polar`` can be resolved from neither an explicit
        argument nor the recorded provenance.
    """
    log = logger or logging.getLogger('turtlewave_hdEEG.rerun')
    recovered = dbwrite.recover_run_scope(
        db_path, event_type, method, freq_lower, freq_upper)

    if recovered is None:
        log.warning(
            "No prior %s/%s run found in detection_runs; cannot recover "
            "ref_chan/polar/cat from provenance -- they must be supplied "
            "explicitly.", event_type, method)

    def _pick(name, explicit):
        """Resolve a column-backed invariant (``ref_chan`` / ``polar``).

        These are always present in a recorded run (``record_run`` stores them in
        a column and :func:`recover_run_scope` parses them back to real objects),
        so recovery only fails when there is NO matching run at all.
        """
        rec_present = recovered is not None
        rec_val = recovered.get(name) if recovered else None
        if explicit is not None:
            if rec_present and str(explicit) != str(rec_val):
                log.warning(
                    "Re-run %s=%r DIFFERS from the original run's %r. Changing "
                    "the reference/polarity is a scientific change, not a "
                    "tuning; densities/amplitudes will not be comparable to the "
                    "untouched channels.", name, explicit, rec_val)
            return explicit
        if rec_present:
            log.info("Re-run %s recovered from provenance: %r", name, rec_val)
            return rec_val
        raise RerunGuardError(
            f"Cannot resolve '{name}' for the re-run from either an explicit "
            f"argument or the recorded provenance for {event_type}/{method}. "
            f"Refusing to re-detect -- a wrong {name} would silently corrupt "
            f"amplitude/polarity. Supply it explicitly.")

    # cat is an ESTIMATION invariant (production runs concatenate,
    # cat=(1,1,1,0); a different cat pools differently and shifts every global
    # threshold). It lives only in params_json, so a pre-P3 run never recorded
    # it. Use an explicit value, else the recorded value ONLY when it was truly
    # recorded (cat_recorded) -- distinguishing a genuine cat=None from an
    # un-recorded one -- else REFUSE, consistent with the ref/polar guard.
    cat_recorded = bool(recovered and recovered.get('cat_recorded'))
    rec_cat = recovered.get('cat') if recovered else None
    if cat is not None:
        if cat_recorded and str(cat) != str(rec_cat):
            log.warning(
                "Re-run cat=%r DIFFERS from the original run's %r. Changing the "
                "concatenation pools the signal differently and shifts global "
                "thresholds; re-detected channels will not be comparable to the "
                "untouched ones.", cat, rec_cat)
        cat_eff = cat
    elif cat_recorded:
        log.info("Re-run cat recovered from provenance: %r", rec_cat)
        cat_eff = rec_cat
    else:
        raise RerunGuardError(
            f"Cannot resolve 'cat' for the {event_type}/{method} re-run: the "
            f"original run did not record it (pre-P3 provenance) and none was "
            f"supplied. Refusing to re-detect -- production runs concatenate "
            f"(cat=(1,1,1,0)) and a silently-assumed default would pool the "
            f"signal differently and shift thresholds. Pass the original run's "
            f"cat explicitly (e.g. --cat 1 1 1 0).")

    return {
        'ref_chan': _pick('ref_chan', ref_chan),
        'polar': _pick('polar', polar),
        'cat': cat_eff,
        'recovered': recovered,
    }
