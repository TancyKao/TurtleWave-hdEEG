"""
cycleprocessor.py

Sleep-cycle detection for TurtleWave-hdEEG.

Ports the rule-based NREM-REM cycle detector from the MATLAB
``SpectraDynamic_Analysis`` toolbox (``cal_SleepCycle_HdEEG.m`` and a
REM-closed variant) to Python, and connects it to TurtleWave's existing
persistence layers:

* the ``events.cycle`` column of ``neural_events.db`` (previously always empty),
* a new ``sleep_cycles`` table holding per-cycle boundaries and durations,
* cycle markers written back into the Wonambi annotation XML so the review GUI
  and ``Annotations.get_cycles()`` can see them.

The detector works purely on the hypnogram (per-epoch sleep stages); no spectral
data is required. Two cycle definitions are supported via ``method``:

``'2022'``
    The modified rule. A cycle is one NREM period plus the segment that
    follows it, up to the next NREM period. An NREM period is a run of NREM
    longer than ``nrem_min`` epochs, within which wake bouts of up to
    ``wake_thresh`` epochs are absorbed, starting at its first N2/N3 epoch.
    The segment counts as the REM period whatever it holds; there is no
    minimum REM duration, so every NREM period yields a cycle.
``'1979'``
    Feinberg & Floyd (1979). Sleep onset at the first N2/N3 epoch; REM runs
    separated by less than ``nrem_min`` epochs of NREM sleep form one REM
    episode; a REM episode of at least ``rem_min`` REM epochs (the first is
    exempt) is a REM period; wake is never a boundary. NREM periods that are
    not followed by a REM period are carried forward, as in the paper.

Both rules treat artefact/unscored epochs as wake. Cycle dicts carry
clock-time and sleep-only durations, a ``rem_class`` and a ``complete`` flag;
see :func:`detect_cycles`.
"""

import logging
import os

import numpy as np

# Database writes go through dbwrite.open_write_connection so this module picks
# up the journal-mode override and the busy timeout (it used to call
# sqlite3.connect directly, with neither).
from . import dbwrite
from .utils import normalize_subject

#: Fallback for module-level helpers called without a processor logger.
LOGGER = logging.getLogger('turtlewave_hdEEG.cycleprocessor')


def _subject_spellings(conn, table, subject, logger=None):
    """Every stored spelling of one recording's id in ``table``.

    The idempotency delete in the cycle writers is keyed on ``subject``. Now
    that the writers normalise before inserting, a row written earlier under
    the bare folder name (which the cycle how-to tells users to pass) is not
    matched by a delete on the canonical id, so the insert adds a *second* row
    instead of replacing the first. ``stage_durations`` has
    ``PRIMARY KEY (subject)`` -- one row per recording is its whole contract --
    and the duplicate doubles any total computed from it.

    Matching every spelling that normalises to the same canonical id makes the
    delete do what it always claimed to.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    table : str
        Table holding a ``subject`` column.
    subject : str
        Canonical (already normalised) subject id.
    logger : logging.Logger or None, optional
        Logger for the stale-spelling notice. Default ``None``.

    Returns
    -------
    list of str
        Stored spellings equivalent to ``subject``, canonical first. On a
        failed lookup this degrades to ``[subject]`` -- the pre-fix
        single-spelling delete -- and says so at WARNING, because that
        degradation silently re-introduces the duplicate row this function
        exists to prevent.
    """
    try:
        stored = [r[0] for r in conn.execute(
            f"SELECT DISTINCT subject FROM {table} "
            f"WHERE subject IS NOT NULL AND subject != ''")]
    except Exception as e:
        (logger or LOGGER).warning(
            "Could not read the stored subject spellings from %s (%s), so the "
            "idempotency delete falls back to the canonical id '%s' alone. If "
            "this recording has rows under an older spelling of its id they "
            "will NOT be replaced, and the insert that follows adds a "
            "duplicate instead. Check the table and re-run.", table, e, subject)
        return [subject]
    equivalent = [s for s in stored
                  if str(s) != subject and normalize_subject(str(s)) == subject]
    if equivalent and logger is not None:
        logger.warning(
            "%s holds this recording under %d older spelling(s) of its "
            "subject id (%s); they are being replaced by '%s' so the "
            "recording keeps one row per key instead of gaining a duplicate.",
            table, len(equivalent), ", ".join(repr(s) for s in equivalent),
            subject)
    return [subject] + equivalent


# Numeric hypnogram codes as produced by ``XLAnnotations.get_hypnogram()``:
# Wake=0, NREM1/2/3=1/2/3, REM=4, artefact/movement/undefined=-1.
_NREM_STAGES = (1, 2, 3)
_N23_STAGES = (2, 3)
_REM_STAGE = 4
_SLEEP_STAGES = (1, 2, 3, 4)

# Coarse scores used by the detection rule (mirrors the MATLAB re-mapping).
_WAKE = 0
_NREM = 2
_ABSORBED_WAKE = 99   # short wake bout absorbed into surrounding NREM
_REM = 4


def _bool_runs(mask):
    """Return ``(start, end_inclusive)`` index pairs for each run of True.

    This is the numpy stand-in for the MATLAB ``bwconncomp`` / ``RunLength``
    calls: it labels maximal contiguous blocks of a boolean array.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    # Pad with False on both ends so edge runs are detected by the diff.
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def detect_cycles(hypnogram, epoch_length=30, wake_thresh=10, nrem_min=30,
                  method='2022', rem_min=10, epoch_starts=None,
                  nrem_onset='n2n3', rem_gap=None, completion_min=10):
    """Detect NREM-REM sleep cycles from a per-epoch hypnogram.

    Parameters
    ----------
    hypnogram : sequence of int
        Numeric per-epoch stage codes as returned by
        ``CustomAnnotations.get_hypnogram()`` (Wake=0, NREM1/2/3=1/2/3, REM=4,
        artefact/undefined=-1).
    epoch_length : float, optional
        Epoch duration in seconds (default 30).
    wake_thresh : int, optional
        ``'2022'`` only. Maximum length, in epochs, of a Wake bout that is
        absorbed into the surrounding NREM instead of ending the NREM period
        (default 10, i.e. 5 min at 30-s epochs).
    nrem_min : int, optional
        Minimum NREM period length in epochs (default 30, i.e. 15 min).
        ``'2022'``: an NREM run must be *longer* than this. ``'1979'``: an
        NREM period must hold at least this many epochs of NREM sleep (wake
        subtracted), and REM runs separated by fewer NREM-sleep epochs than
        this belong to one REM period (see ``rem_gap``).
    method : {'2022', '1979'}, optional
        Cycle definition; see Notes. Default ``'2022'``.
    rem_min : int, optional
        ``'1979'`` only. Minimum number of REM epochs for a REM episode to
        count as a REM period (default 10, i.e. 5 min). The first REM period
        of the night is exempt and needs one REM epoch.
    epoch_starts : sequence of float, optional
        Start time in seconds of each epoch, same length as ``hypnogram``. If
        omitted, epoch ``i`` is assumed to start at ``i * epoch_length``.
    nrem_onset : {'n2n3', 'any'}, optional
        ``'2022'`` only. ``'n2n3'`` (default) starts each NREM period at its
        first N2 or N3 epoch, so leading N1 falls into the preceding segment.
        ``'any'`` starts it at the first NREM epoch of any stage, which is the
        pre-4.5 behaviour and matches the MATLAB ``cal_SleepCycle.m``.
        ``'1979'`` always uses stage-2 onset, as Feinberg & Floyd do.
    rem_gap : int or None, optional
        ``'1979'`` only. REM runs separated by fewer than this many epochs of
        NREM sleep are merged into one REM episode. ``None`` (default) uses
        ``nrem_min``, which is the Feinberg & Floyd rule (an interruption of
        less than 15 min of NREM does not end the REM period).
    completion_min : int, optional
        Sleep epochs that must follow the last NREM period (``'2022'``) or the
        last REM period (``'1979'``) for the final cycle to be flagged
        ``complete`` (default 10, i.e. 5 min; Feinberg & Floyd's completion
        rule).

    Returns
    -------
    list of dict
        One dict per cycle, in chronological order, with keys:
        ``cycle_number`` (1-based), ``method``, ``nrem_start_epoch``,
        ``nrem_end_epoch``, ``rem_start_epoch``, ``rem_end_epoch`` (all
        inclusive epoch indices; ``rem_*`` is the segment that follows the
        NREM period, which may be empty -> ``rem_start_epoch >
        rem_end_epoch``), ``nrem_start_sec``, ``nrem_end_sec``,
        ``rem_end_sec`` (cycle end in seconds), the clock-time durations
        ``nrem_dur_min``, ``rem_dur_min``, ``cycle_dur_min``, the sleep-only
        durations ``nrem_n23_dur_min`` (N2+N3 in the NREM period),
        ``nrem_sleep_min`` (N1+N2+N3 in the NREM period), ``rem_sleep_min``
        (REM epochs in the segment), ``rem_in_nremp_min`` (REM epochs inside
        the NREM period; non-zero only under ``'1979'``), ``wake_in_seg_min``
        (wake and unscored epochs in the segment), ``rem_class``
        (``'full'`` when ``rem_sleep_min`` covers at least ``rem_min``
        epochs, ``'short'`` for fewer but at least one, ``'none'``),
        ``complete`` (bool, see ``completion_min``) and ``sorem`` (bool; True
        on cycle 1 under ``'1979'`` when a REM episode occurred before
        ``nrem_min`` epochs of NREM sleep had accumulated and was absorbed).

    Notes
    -----
    ``'2022'`` (modified rule). Every NREM period is a cycle. An NREM period
    is a run of NREM epochs, within which wake bouts of up to ``wake_thresh``
    epochs are absorbed, that is longer than ``nrem_min`` epochs; it starts
    at its first N2/N3 epoch (``nrem_onset='n2n3'``) and ends on its last NREM
    epoch. The REM segment of a cycle is everything from the end of the NREM
    period to the start of the next one, whatever stages it holds, with no
    minimum REM duration; the last segment ends at the last sleep epoch of
    the recording, so trailing wake is outside every cycle.

    ``'1979'`` (Feinberg & Floyd 1979, Psychophysiology 16:283). Sleep onset
    is the first N2/N3 epoch. REM runs separated by fewer than ``rem_gap``
    epochs of NREM sleep form one REM episode. A REM episode is a REM period
    if it holds at least ``rem_min`` REM epochs; the first episode after at
    least ``nrem_min`` epochs of NREM sleep since sleep onset needs only one
    REM epoch, and an episode before that point is a sleep-onset REM episode
    that is absorbed. Shorter later episodes are absorbed into the NREM
    period around them. NREM period *i* runs from the first N2/N3 epoch after
    REM period *i-1* to the epoch before REM period *i*; wake is never a
    boundary and is only subtracted from ``nrem_sleep_min``. The REM segment
    of cycle *i* runs from the first epoch of REM period *i* to the epoch
    before NREM period *i+1* (Feinberg & Floyd's NREM cycle, stage-2 onset to
    stage-2 onset), or to the last sleep epoch for the final cycle. A trailing
    NREM period with at least ``nrem_min`` epochs of NREM sleep and no REM
    period is returned as an incomplete final cycle.

    Artefact/undefined epochs (-1, or NaN) are treated as wake under both
    rules. An empty hypnogram, or one with no sleep, returns ``[]``.
    """
    hyp = np.asarray(list(hypnogram), dtype=float)
    n = hyp.size
    if n == 0:
        return []
    if method not in ('2022', '1979'):
        raise ValueError(f"Unknown method {method!r}; use '2022' or '1979'")
    if nrem_onset not in ('n2n3', 'any'):
        raise ValueError(
            f"Unknown nrem_onset {nrem_onset!r}; use 'n2n3' or 'any'")
    if rem_gap is None:
        rem_gap = nrem_min

    if epoch_starts is not None:
        starts = np.asarray(list(epoch_starts), dtype=float)
        if starts.size != n:
            raise ValueError("epoch_starts must match hypnogram length")
    else:
        starts = np.arange(n, dtype=float) * epoch_length

    def epoch_start_sec(i):
        return float(starts[i])

    def epoch_end_sec(i):
        # Use the next epoch's start when available so gaps are respected;
        # fall back to start + epoch_length for the final epoch.
        if i + 1 < n:
            return float(starts[i + 1])
        return float(starts[i] + epoch_length)

    is_sleep = np.isin(hyp, _SLEEP_STAGES)
    if not is_sleep.any():
        return []
    last_sleep = int(np.where(is_sleep)[0][-1])
    is_nrem = np.isin(hyp, _NREM_STAGES)
    is_n23 = np.isin(hyp, _N23_STAGES)
    is_rem = hyp == _REM_STAGE

    if method == '2022':
        groups = _modified_groups(hyp, is_nrem, is_n23, wake_thresh, nrem_min,
                                  nrem_onset, last_sleep)
    else:
        groups = _feinberg_groups(is_nrem, is_n23, is_rem, nrem_min, rem_min,
                                  rem_gap, last_sleep)
    if not groups:
        return []

    to_min = epoch_length / 60.0
    cycles = []
    n_groups = len(groups)
    for cyc_num, g in enumerate(groups, start=1):
        ns, ne = g['nrem']
        seg_start, seg_end = g['seg']
        has_seg = seg_end >= seg_start
        is_last = cyc_num == n_groups

        nremp = hyp[ns:ne + 1]
        nrem_dur_min = (ne - ns + 1) * to_min
        n23_epochs = int(np.count_nonzero(np.isin(nremp, _N23_STAGES)))
        nrem_sleep_epochs = int(np.count_nonzero(np.isin(nremp, _NREM_STAGES)))
        rem_in_nremp = int(np.count_nonzero(nremp == _REM_STAGE))

        if has_seg:
            seg = hyp[seg_start:seg_end + 1]
            rem_dur_min = (seg_end - seg_start + 1) * to_min
            rem_sleep_epochs = int(np.count_nonzero(seg == _REM_STAGE))
            wake_in_seg = int(np.count_nonzero(~np.isin(seg, _SLEEP_STAGES)))
        else:
            rem_dur_min = 0.0
            rem_sleep_epochs = 0
            wake_in_seg = 0

        if not is_last:
            complete = True
        elif method == '2022':
            complete = (has_seg and int(np.count_nonzero(
                is_sleep[seg_start:seg_end + 1])) >= completion_min)
        else:
            rem_ep = g.get('rem_ep')
            if rem_ep is None:
                complete = False
            else:
                after = is_nrem[rem_ep[1] + 1:seg_end + 1]
                complete = int(np.count_nonzero(after)) >= completion_min

        if rem_sleep_epochs >= rem_min:
            rem_class = 'full'
        elif rem_sleep_epochs > 0:
            rem_class = 'short'
        else:
            rem_class = 'none'

        cycle_end_epoch = seg_end if has_seg else ne
        cycles.append({
            'cycle_number': cyc_num,
            'method': method,
            'nrem_start_epoch': int(ns),
            'nrem_end_epoch': int(ne),
            'rem_start_epoch': int(seg_start if has_seg else ne + 1),
            'rem_end_epoch': int(seg_end if has_seg else ne),
            'nrem_start_sec': epoch_start_sec(ns),
            'nrem_end_sec': epoch_end_sec(ne),
            'rem_end_sec': epoch_end_sec(cycle_end_epoch),
            'nrem_dur_min': round(nrem_dur_min, 3),
            'nrem_n23_dur_min': round(n23_epochs * to_min, 3),
            'nrem_sleep_min': round(nrem_sleep_epochs * to_min, 3),
            'rem_dur_min': round(rem_dur_min, 3),
            'rem_sleep_min': round(rem_sleep_epochs * to_min, 3),
            'rem_in_nremp_min': round(rem_in_nremp * to_min, 3),
            'wake_in_seg_min': round(wake_in_seg * to_min, 3),
            'cycle_dur_min': round(nrem_dur_min + rem_dur_min, 3),
            'rem_class': rem_class,
            'complete': bool(complete),
            'sorem': bool(g.get('sorem', False)),
        })

    return cycles


def _modified_groups(hyp, is_nrem, is_n23, wake_thresh, nrem_min, nrem_onset,
                     last_sleep):
    """NREM periods and segments under the ``'2022'`` rule.

    Returns a list of ``{'nrem': (start, end), 'seg': (start, end)}`` with
    inclusive epoch indices, or ``[]``.
    """
    n = hyp.size
    # Coarse re-map (mirrors the MATLAB re-mapping): NREM -> 2, REM -> 4,
    # everything else (Wake and artefact/undefined) -> 0.
    score = np.full(n, _WAKE, dtype=int)
    score[is_nrem] = _NREM
    score[hyp == _REM_STAGE] = _REM

    # Absorb short Wake bouts into the surrounding NREM.
    for s, e in _bool_runs(score == _WAKE):
        if (e - s + 1) <= wake_thresh:
            score[s:e + 1] = _ABSORBED_WAKE

    # Contiguous NREM (real + absorbed wake), dropping short runs, then trim
    # absorbed wake from both ends so the period starts and ends on NREM.
    nrem_mask = (score == _NREM) | (score == _ABSORBED_WAKE)
    periods = []
    for s, e in _bool_runs(nrem_mask):
        if (e - s + 1) <= nrem_min:
            continue
        real = np.where(score[s:e + 1] == _NREM)[0]
        if real.size == 0:
            continue
        ns = s + int(real[0])
        ne = s + int(real[-1])
        if nrem_onset == 'n2n3':
            n23 = np.where(is_n23[ns:ne + 1])[0]
            if n23.size == 0:
                continue
            ns = ns + int(n23[0])
            # Re-test the length from the N2/N3 onset over the same run
            # (absorbed wake included), as the original test did from the
            # run start.
            if (e - ns + 1) <= nrem_min:
                continue
        periods.append((ns, ne))

    groups = []
    for idx, (ns, ne) in enumerate(periods):
        seg_start = ne + 1
        seg_end = (periods[idx + 1][0] - 1 if idx + 1 < len(periods)
                   else last_sleep)
        groups.append({'nrem': (ns, ne), 'seg': (seg_start, seg_end)})
    return groups


def _feinberg_groups(is_nrem, is_n23, is_rem, nrem_min, rem_min, rem_gap,
                     last_sleep):
    """NREM periods, REM periods and segments under Feinberg & Floyd 1979.

    Returns a list of ``{'nrem': (s, e), 'seg': (s, e), 'rem_ep': (s, e) or
    None, 'sorem': bool}`` with inclusive epoch indices, or ``[]``.
    """
    onset = np.where(is_n23)[0]
    if onset.size == 0:
        return []
    onset = int(onset[0])

    # REM runs after sleep onset, merged across gaps holding fewer than
    # rem_gap epochs of NREM sleep.
    episodes = []
    for s, e in _bool_runs(is_rem):
        if s < onset:
            continue
        if episodes:
            ps, pe = episodes[-1]
            gap_nrem = int(np.count_nonzero(is_nrem[pe + 1:s]))
            if gap_nrem < rem_gap:
                episodes[-1] = (ps, e)
                continue
        episodes.append((s, e))

    cycles = []
    cursor = onset          # start of the NREM period being built
    first_found = False
    sorem = False
    for rs, re_ in episodes:
        rem_epochs = int(np.count_nonzero(is_rem[rs:re_ + 1]))
        if not first_found:
            nrem_before = int(np.count_nonzero(is_nrem[cursor:rs]))
            if nrem_before < nrem_min:
                sorem = True          # sleep-onset REM: absorbed
                continue
            qualifies = rem_epochs >= 1
        else:
            qualifies = rem_epochs >= rem_min
        if not qualifies:
            continue                  # short REM: absorbed into the NREMP
        cycles.append({'nrem': (cursor, rs - 1), 'rem_ep': (rs, re_),
                       'sorem': False})
        first_found = True
        nxt = np.where(is_n23[re_ + 1:])[0]
        if nxt.size == 0:
            cursor = None
            break
        cursor = re_ + 1 + int(nxt[0])

    if cursor is not None:
        trailing_nrem = int(np.count_nonzero(is_nrem[cursor:last_sleep + 1]))
        if trailing_nrem >= nrem_min:
            cycles.append({'nrem': (cursor, last_sleep), 'rem_ep': None,
                           'sorem': False})

    if not cycles:
        return []
    cycles[0]['sorem'] = sorem

    for idx, cyc in enumerate(cycles):
        ns, ne = cyc['nrem']
        if cyc['rem_ep'] is None:
            cyc['seg'] = (ne + 1, ne)          # empty
            continue
        seg_start = cyc['rem_ep'][0]
        seg_end = (cycles[idx + 1]['nrem'][0] - 1 if idx + 1 < len(cycles)
                   else last_sleep)
        cyc['seg'] = (seg_start, seg_end)
    return cycles


def compute_stage_durations(hypnogram, epoch_length=30, durations=None):
    """Sum per-epoch sleep-stage minutes from a hypnogram.

    Counts how many epochs fall in each numeric stage code and converts to
    minutes. Mirrors the free-function pattern of :func:`detect_cycles` so it is
    unit-testable without a database.

    Parameters
    ----------
    hypnogram : sequence of int
        Numeric per-epoch stage codes as returned by
        ``CustomAnnotations.get_hypnogram()`` (Wake=0, NREM1/2/3=1/2/3, REM=4,
        artefact/undefined=-1).
    epoch_length : float, optional
        Epoch duration in seconds (default 30). Ignored when ``durations``
        is given, except as the reported ``epoch_length``.
    durations : sequence of float, optional
        Length in seconds of each epoch, same length as ``hypnogram`` (for
        example ``CustomAnnotations.epoch_durations()`` on a cut recording's
        exact epochs). When given, stage minutes are sums of these durations
        rather than epoch counts times ``epoch_length``.

    Returns
    -------
    dict
        Duration summary with keys ``epoch_length`` (seconds), ``wake_min``,
        ``n1_min``, ``n2_min``, ``n3_min``, ``rem_min``, ``artefact_min`` (all
        non-sleep-stage epochs; see Notes), and ``total_min``. ``total_min`` is
        ``n_epochs * epoch_length / 60`` (or the summed ``durations``: the
        hypnogram span), so the stage
        parts reconcile exactly by construction: ``wake + n1 + n2 + n3 + rem +
        artefact == total``.

    Notes
    -----
    ``artefact_min`` is computed as the remainder ``total - (wake + n1 + n2 +
    n3 + rem)`` rather than by counting a single code. In normal use these are
    the -1 artefact/undefined epochs (``get_hypnogram`` only emits
    ``{0, 1, 2, 3, 4, -1}``), but folding *any* code outside ``{0, 1, 2, 3, 4}``
    into the remainder guarantees the reconciliation invariant can never be
    silently broken by an unexpected code. An empty hypnogram returns all-zero
    durations.
    """
    hyp = np.asarray(list(hypnogram), dtype=float)
    n = hyp.size
    if durations is None:
        per_epoch_min = epoch_length / 60.0

        def stage_min(code):
            return float(np.count_nonzero(hyp == code)) * per_epoch_min

        total_min = float(n) * per_epoch_min
    else:
        minutes = np.asarray(list(durations), dtype=float) / 60.0
        if minutes.size != n:
            raise ValueError(
                f"durations has {minutes.size} entries but the hypnogram has "
                f"{n} epochs")

        def stage_min(code):
            return float(minutes[hyp == code].sum())

        total_min = float(minutes.sum())

    wake_min = stage_min(0)
    n1_min = stage_min(1)
    n2_min = stage_min(2)
    n3_min = stage_min(3)
    rem_min = stage_min(_REM_STAGE)
    # Fold every non-sleep-stage epoch (typically code -1, but also any
    # unexpected code) into the remainder so the parts always sum to total.
    artefact_min = total_min - (wake_min + n1_min + n2_min + n3_min + rem_min)

    return {
        'epoch_length': float(epoch_length),
        'wake_min': wake_min,
        'n1_min': n1_min,
        'n2_min': n2_min,
        'n3_min': n3_min,
        'rem_min': rem_min,
        'artefact_min': artefact_min,
        'total_min': total_min,
    }


#: Stages given an ``analysed_time_cycles`` row per cycle when the caller
#: names none.
DEFAULT_CYCLE_STAGES = ('NREM1', 'NREM2', 'NREM3', 'REM')

#: ``time_base`` values in ``sleep_cycles`` and ``stage_durations``.
#: ``'original'``: seconds and minutes are on the recording's own time base,
#: which is the full night (an uncut file, or a cut file's full-night
#: hypnogram). ``'cut'``: seconds are on the cut file; ``*_orig`` columns hold
#: the original-night values.
TIME_BASE_ORIGINAL = 'original'
TIME_BASE_CUT = 'cut'


def fullnight_hypnogram_codes(timeline):
    """Numeric hypnogram of a timeline's full-night stages.

    Parameters
    ----------
    timeline : RecordingTimeline
        Map whose ``stages`` are the full-night Compumedics codes.

    Returns
    -------
    list of int
        One code per original ``timeline.epoch_length`` epoch, numbered as
        ``CustomAnnotations.get_hypnogram()`` numbers stages.
    """
    from .annotation import HYPNOGRAM_CODES
    from .timeline import stage_name_for_code
    return [HYPNOGRAM_CODES.get(stage_name_for_code(c), -1)
            for c in timeline.fullnight_hypnogram()]


def _cycles_to_cut(cycles, timeline, annotations):
    """Move full-night cycles onto the cut file's time base.

    Parameters
    ----------
    cycles : list of dict
        Cycles from :func:`detect_cycles` on the full-night hypnogram, seconds
        on the original night.
    timeline : RecordingTimeline
        The cut file's time map.
    annotations : object
        The cut file's annotations, exposing ``get_stage_intervals()``.

    Returns
    -------
    list of dict
        Copies of ``cycles``. ``nrem_start_orig``, ``nrem_end_orig`` and
        ``rem_end_orig`` keep the original-night seconds; ``nrem_start_sec``,
        ``nrem_end_sec`` and ``rem_end_sec`` become cut-file seconds, each
        ``original_to_cut(t)`` snapped to the nearest exact-epoch edge so that
        ``get_epochs(time=...)`` containment and the XML cycle markers work.
        ``*_min`` and the ``*_epoch`` indices stay full-night (the indices
        count original 30 s epochs, not XML epochs). ``time_base`` is
        ``'cut'``.
    """
    ivals = annotations.get_stage_intervals()
    if not ivals:
        raise ValueError("the annotation file has no epochs to place cycle "
                         "bounds on")
    edges = np.array(sorted({float(s) for s, _, _ in ivals}
                            | {float(ivals[-1][1])}))

    def snap(t):
        c = float(timeline.original_to_cut(float(t)))
        return float(edges[int(np.argmin(np.abs(edges - c)))])

    out = []
    for cyc in cycles:
        new = dict(cyc)
        for key, okey in (('nrem_start_sec', 'nrem_start_orig'),
                          ('nrem_end_sec', 'nrem_end_orig'),
                          ('rem_end_sec', 'rem_end_orig')):
            new[okey] = float(cyc[key])
            new[key] = snap(cyc[key])
        new['time_base'] = TIME_BASE_CUT
        out.append(new)
    return out


def _require_existing_db(db_path):
    """Refuse to run a backfill against a database that does not exist.

    Cycle detection and stage-duration accounting are *post-detection* steps:
    they annotate an existing ``neural_events.db``, they never originate one.
    Without this check a mistyped path is silently created as an empty database
    (``sqlite3.connect`` creates the file), the run then dies on
    ``no such table: main.events``, and a stray file is left behind -- on a
    network share, in whatever journal mode the creating call chose.

    Parameters
    ----------
    db_path : str
        Path to the ``neural_events.db`` SQLite database.

    Raises
    ------
    FileNotFoundError
        If no file exists at ``db_path``.
    """
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"No database at {db_path}. Sleep-cycle backfill annotates an "
            f"existing neural_events.db and never creates one -- run event "
            f"detection first, or correct the path.")


class ParalCycles:
    """Detect sleep cycles and persist them to the DB and annotation XML.

    Follows the ``Paral*`` convention used by the other processors: it takes a
    ``dataset``/``annotations`` pair, owns a ``logging.Logger``, and exposes a
    single :meth:`run` entry point that both backfills existing databases and
    tags new detection runs.
    """

    def __init__(self, dataset=None, annotations=None, subject=None,
                 log_level=logging.INFO, log_file=None):
        """Initialize the ParalCycles object.

        Parameters
        ----------
        dataset : Dataset, optional
            Dataset object (kept for API symmetry; not required for detection).
        annotations : CustomAnnotations
            Annotations wrapper providing ``get_hypnogram`` / ``epochs`` and,
            for marker writing, the Wonambi cycle-marker methods (delegated).
        subject : str, optional
            Subject identifier stored in the ``sleep_cycles`` table.
        log_level : int
            Logging level (e.g. ``logging.INFO``).
        log_file : str or None
            Path to a log file. If None, logs to console only.
        """
        self.dataset = dataset
        self.annotations = annotations
        self.subject = subject
        self.logger = self._setup_logger(log_level, log_file)
        self._logged = set()

    def _setup_logger(self, log_level, log_file=None):
        """Set up a dedicated logger for this processor."""
        logger = logging.getLogger('turtlewave_hdEEG.cycleprocessor')
        logger.setLevel(log_level)

        # Process-wide singleton: clear stale handlers so batch loops don't
        # duplicate lines or leak file handles.
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def _epoch_starts(self):
        """Return per-epoch start seconds, or None if unavailable."""
        try:
            epochs = self.annotations.epochs
        except Exception as e:
            self.logger.warning(f"Could not read epochs for timing: {e}")
            return None
        if not epochs:
            return None
        try:
            return [float(ep['start']) for ep in epochs]
        except (KeyError, TypeError, ValueError):
            return None

    def _log_once(self, key, level, msg):
        """Log ``msg`` the first time ``key`` is seen by this instance."""
        if key in self._logged:
            return
        self._logged.add(key)
        self.logger.log(level, msg)

    def _resolve_timeline(self, timeline='auto'):
        """The timeline to compute cycles with, or ``None`` for the XML grid.

        Parameters
        ----------
        timeline : {'auto', 'none'} or RecordingTimeline
            ``'auto'`` reads the sidecar beside ``annotations.annot_file``
            when it exists (:func:`~turtlewave_hdEEG.timeline.load_sidecar_for`
            with ``require=False``); ``'none'`` ignores it; a
            ``RecordingTimeline`` is used as given.

        Returns
        -------
        RecordingTimeline or None

        Raises
        ------
        ValueError
            The annotation file has variable-length epochs and no timeline
            was found or ``'none'`` was asked for: cycle detection counts
            30 s epochs of the full night and must never run on exact epochs.
            Also for an unknown ``timeline`` value.
        timeline.SidecarMismatchError
            ``'auto'`` found a sidecar that no longer matches the XML.
        """
        from .timeline import RecordingTimeline, load_sidecar_for, sidecar_path
        ann = self.annotations
        annot_file = getattr(ann, 'annot_file', None)
        if isinstance(timeline, RecordingTimeline):
            tl = timeline
        elif timeline == 'auto':
            tl = (load_sidecar_for(annot_file, ann, require=False)
                  if annot_file else None)
        elif timeline in ('none', None):
            tl = None
        else:
            raise ValueError(f"timeline={timeline!r}; use 'auto', 'none' or a "
                             f"RecordingTimeline")
        variable = (hasattr(ann, 'has_uniform_epochs')
                    and not ann.has_uniform_epochs())
        if tl is None and variable:
            where = (sidecar_path(annot_file).name if annot_file
                     else '<annotation xml stem>_timeline.json')
            why = ("timeline='none' was passed" if timeline in ('none', None)
                   else f"there is no timeline sidecar {where}")
            raise ValueError(
                f"The annotation file has variable-length epochs (a cut "
                f"recording's exact epochs) and {why}, so sleep cycles cannot "
                f"be computed: cycle detection counts 30 s epochs of the full "
                f"night, which only the sidecar holds. Nothing was written. "
                f"Re-run the annotation step "
                f"(XLAnnotations.add_stages_from_header) to write the XML and "
                f"its sidecar together.")
        if tl is not None:
            if tl.stages:
                self._log_once(
                    'timeline', logging.INFO,
                    f"Cycles and stage durations from the full-night "
                    f"hypnogram in the timeline sidecar ({len(tl.stages)} "
                    f"epochs of {tl.epoch_length:g} s, {tl.n_boundaries} "
                    f"splices removing {tl.removed_seconds:.1f} s); cycle "
                    f"bounds are moved onto the cut file's epoch edges.")
            else:
                self._log_once(
                    'no_fullnight', logging.ERROR,
                    "Sleep cycles NOT computed: this recording was staged from "
                    "its stage events, so the full-night hypnogram is "
                    "unavailable (the sidecar's fullnight_stages is empty) and "
                    "cycle detection cannot run on the cut file's variable-"
                    "length epochs. Stage durations are stored from the cut "
                    "file's epoch durations (time_base='cut'); sleep_cycles, "
                    "events.cycle and analysed_time_cycles are left empty.")
        return tl

    def detect(self, method='2022', epoch_length=30, wake_thresh=10,
               nrem_min=30, rem_min=10, hypnogram=None, nrem_onset='n2n3'):
        """Run cycle detection on the annotation hypnogram.

        Parameters
        ----------
        hypnogram : sequence of int, optional
            Pre-read hypnogram to reuse. If ``None`` (default), it is read from
            ``self.annotations``. Allows :meth:`run` to read the hypnogram once
            and share it with stage-duration computation.

        Returns
        -------
        list of dict
            Cycle dicts as produced by :func:`detect_cycles`.
        """
        if self.annotations is None:
            raise ValueError("annotations are required for cycle detection")

        if hypnogram is None:
            hypnogram = self.annotations.get_hypnogram()
        if not hypnogram:
            self.logger.warning("Empty hypnogram; no cycles detected.")
            return []

        epoch_starts = self._epoch_starts()
        cycles = detect_cycles(
            hypnogram, epoch_length=epoch_length, wake_thresh=wake_thresh,
            nrem_min=nrem_min, method=method, rem_min=rem_min,
            epoch_starts=epoch_starts, nrem_onset=nrem_onset)
        self.logger.info(
            f"Detected {len(cycles)} cycle(s) using method '{method}'.")
        return cycles

    def write_cycle_markers(self, cycles):
        """Replace the cycle markers in the Wonambi annotation XML.

        Uses the underlying Wonambi ``clear_cycles`` / ``set_cycle_mrkr`` (via
        ``CustomAnnotations`` delegation). Markers must land exactly on existing
        epoch starts, so boundaries are taken from the epoch grid. Failures are
        logged and swallowed so DB persistence is never blocked by XML issues.

        The clear always runs, including when ``cycles`` is empty: this is a
        *replacement*, not an append. A re-run that finds no cycles (a raised
        ``nrem_min``, an all-Wake night, rescored epochs) must leave the XML
        agreeing with ``sleep_cycles`` and ``events.cycle``, and keeping the
        previous run's markers would leave the annotation file describing
        cycles that no longer exist anywhere else.

        Because the clear always runs, **every** call rewrites the annotation
        XML, including a zero-cycle one: Wonambi's ``clear_cycles()`` saves the
        file, and ``Annotations.save()`` stamps the rater's ``modified``
        attribute with the current time. So a zero-cycle night legitimately
        shows a fresh ``modified`` timestamp and a changed file on disk even
        though the only content change is the *removal* of the previous run's
        markers. That is the intended outcome, not a spurious write -- but it
        does mean file timestamps and version-control diffs cannot be used to
        tell "cycles were written" from "cycles were cleared"; read
        ``get_cycles()`` or the ``sleep_cycles`` table for that. Pass
        ``write_xml=False`` through :meth:`run` /
        :func:`finalize_cycles_and_durations` to leave the XML untouched.

        Parameters
        ----------
        cycles : list of dict
            Detected cycles, as returned by :meth:`detect`. An empty list
            clears the markers and writes none.

        Returns
        -------
        bool
            True when at least one cycle's markers were written. False when
            ``cycles`` was empty (the markers were still cleared) or the epoch
            grid needed to place them was unavailable.
        """
        # Clear before anything can bail out, so an empty cycle list -- or a
        # missing epoch grid -- still leaves the XML free of the previous run's
        # markers rather than silently keeping them.
        try:
            self.annotations.clear_cycles()
        except Exception as e:
            self.logger.warning(f"clear_cycles failed: {e}")

        if not cycles:
            self.logger.info(
                "No cycles to mark; cleared any existing XML cycle markers.")
            return False
        epoch_starts = self._epoch_starts()
        if epoch_starts is None:
            self.logger.warning(
                "No epoch grid available; XML cycle markers were cleared but "
                "none were written.")
            return False
        n = len(epoch_starts)

        if cycles[0].get('time_base') == TIME_BASE_CUT:
            # Cut-file cycles: the epoch indices count full-night epochs, so
            # place the markers from the cut seconds, which _cycles_to_cut
            # snapped onto exact-epoch edges.
            starts = sorted({int(round(t)) for t in epoch_starts})
            start_set = set(starts)
            written = 0
            for cyc in cycles:
                lo = int(round(cyc['nrem_start_sec']))
                hi = int(round(cyc['rem_end_sec']))
                if hi not in start_set:          # the recording end
                    hi = starts[-1]
                if lo not in start_set or hi <= lo:
                    self.logger.warning(
                        f"Cycle {cyc['cycle_number']} has no cut-file span "
                        f"to mark (cut {cyc['nrem_start_sec']:g}-"
                        f"{cyc['rem_end_sec']:g} s; most of it was removed "
                        f"from the recording); no XML markers for it.")
                    continue
                try:
                    self.annotations.set_cycle_mrkr(lo)
                    self.annotations.set_cycle_mrkr(hi, end=True)
                    written += 1
                except Exception as e:
                    self.logger.warning(
                        f"Could not mark cycle {cyc['cycle_number']}: {e}")
            self.logger.info(
                f"Wrote markers for {written} cycle(s) to XML (cut-file "
                f"time base).")
            return written > 0

        written = 0
        for cyc in cycles:
            start_i = cyc['nrem_start_epoch']
            # End marker: start of the epoch after the cycle when it exists,
            # so the boundary matches the next cycle's start; otherwise the
            # last epoch's start.
            end_i = cyc['rem_end_epoch']
            if end_i < cyc['rem_start_epoch']:      # empty REM segment
                end_i = cyc['nrem_end_epoch']
            marker_end_i = end_i + 1 if end_i + 1 < n else end_i
            try:
                self.annotations.set_cycle_mrkr(
                    int(round(epoch_starts[start_i])))
                self.annotations.set_cycle_mrkr(
                    int(round(epoch_starts[marker_end_i])), end=True)
                written += 1
            except Exception as e:
                self.logger.warning(
                    f"Could not mark cycle {cyc['cycle_number']}: {e}")
        self.logger.info(f"Wrote markers for {written} cycle(s) to XML.")
        return written > 0

    @staticmethod
    def _ensure_sleep_cycles_table(conn):
        """Create the sleep_cycles table + events cycle index if missing.

        Also migrates a sleep_cycles table created by an earlier version by
        adding any missing columns, so backfilling an existing DB never fails
        on an outdated schema.
        """
        conn.execute('''
        CREATE TABLE IF NOT EXISTS sleep_cycles (
            subject TEXT,
            method TEXT,
            cycle_number INTEGER,
            nrem_start REAL,
            nrem_end REAL,
            rem_start REAL,
            rem_end REAL,
            nrem_dur_min REAL,
            nrem_n23_dur_min REAL,
            rem_dur_min REAL,
            cycle_dur_min REAL,
            PRIMARY KEY (subject, method, cycle_number)
        )''')
        # Additive migration for tables made before these columns existed.
        # nrem_*_orig / rem_end_orig: the bounds on the original night (equal
        # to nrem_start/nrem_end/rem_end unless time_base is 'cut');
        # time_base: 'original' or 'cut' (see TIME_BASE_*). NULL on rows
        # written before 4.5.
        existing = {r[1] for r in conn.execute(
            'PRAGMA table_info(sleep_cycles)').fetchall()}
        for col, sql_type in (('nrem_n23_dur_min', 'REAL'),
                              ('nrem_start_orig', 'REAL'),
                              ('nrem_end_orig', 'REAL'),
                              ('rem_end_orig', 'REAL'),
                              ('time_base', 'TEXT')):
            if col not in existing:
                conn.execute(
                    f'ALTER TABLE sleep_cycles ADD COLUMN {col} {sql_type}')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_cycle ON events(cycle)')

    def store_cycles_to_database(self, cycles, db_path, subject=None,
                                 method=None, conn=None):
        """Replace this subject's rows in the ``sleep_cycles`` table.

        Existing rows for the same ``(subject, method)`` are deleted, then the
        new cycles are inserted, so reruns stay idempotent. Every
        ``analysed_time_cycles`` row of the same ``(subject, method)`` is
        deleted with them, under every stage and reject set, since those rows
        describe the cycles being replaced.

        The delete runs even when ``cycles`` is empty. That is what makes a
        re-run finding no cycles (a raised ``nrem_min``, an all-Wake night,
        rescored epochs) *replace* the previous run rather than leave its rows
        behind: :meth:`tag_events_with_cycles` clears ``events.cycle`` for the
        same case, so skipping the delete here would leave the table claiming
        cycles that no event is tagged with and no XML marker records.

        Parameters
        ----------
        cycles : list of dict
            Detected cycles, as returned by :meth:`detect`. An empty list
            deletes the subject's rows for ``method`` and inserts nothing.
        db_path : str
            Path to the ``neural_events.db`` SQLite database.
        subject : str, optional
            Subject identifier. Falls back to ``self.subject`` (then ``''``).
        method : str, optional
            Cycle definition naming the rows to replace. Read from the cycles
            themselves whenever there are any, in which case this argument is
            ignored; when ``cycles`` is empty it is the *only* thing naming
            the rows to delete, so ``method=None`` together with an empty
            ``cycles`` raises (see ``Raises``) rather than silently deleting
            nothing.
        conn : sqlite3.Connection, optional
            An already-open connection **on ``db_path``** (not checked at
            runtime; it is a caller contract). When supplied, the caller owns
            closing it and this method neither opens nor closes a connection,
            which is what lets a whole subject share one connection. When
            ``None`` a connection is opened via
            :func:`~turtlewave_hdEEG.dbwrite.open_write_connection` and closed
            here.

        Returns
        -------
        int
            Number of cycles written.

        Raises
        ------
        ValueError
            If nothing names the rows to replace: ``cycles`` is empty and
            ``method`` is ``None``, or a supplied cycle dict carries
            ``method=None``. The delete is keyed on ``(subject, method)`` and
            ``method=NULL`` matches no row, so such a call would delete
            nothing, insert nothing (or insert un-matchable NULL-method rows)
            and still return, leaving the previous run's rows in place while
            :meth:`tag_events_with_cycles` has cleared every ``events.cycle``
            and :meth:`write_cycle_markers` the XML markers -- the three
            stores disagreeing, which is the defect the 4.3.1 zero-cycle fix
            closed. This used to warn and continue; the method is always known
            at the call site (:meth:`run` passes its own), so it is a caller
            bug. An empty ``cycles`` list with a real ``method`` is the valid
            "this run found no cycles, clear the store" call and keeps
            working. Raised before any connection is opened, so a mistaken
            call cannot create ``db_path`` or alter its schema.
        """
        # One canonical spelling, matching analysed_time / pac_coupling and the
        # detectors. The cycle how-to tells users to pass the bare folder name,
        # so without this a recording carries '10sd' here and 'sub-10sd' there
        # -- two subjects to SQL, and the detectors' single-subject guard then
        # refuses the recording's own next run.
        subject = normalize_subject(
            subject if subject is not None else (self.subject or ''))

        # Resolve which (subject, method) rows this call replaces BEFORE
        # opening a connection or creating the table, so a call that cannot
        # name them changes nothing at all -- open_write_connection would
        # otherwise create a missing db_path and _ensure_sleep_cycles_table
        # add tables to it, for a call that goes on to write no rows.
        method_vals = {c['method'] for c in cycles} or {method}
        if None in method_vals:
            raise ValueError(
                f"store_cycles_to_database cannot identify the sleep_cycles "
                f"rows to replace for subject '{subject}': "
                + ("the cycles list is empty and method=None, so the delete "
                   "has no method to key on and would remove nothing, "
                   "leaving a previous run's rows behind as the only record "
                   "of cycles that events.cycle and the XML markers no "
                   "longer describe."
                   if not cycles else
                   "at least one cycle dict carries method=None, so its row "
                   "would be stored with a NULL method that no later "
                   "replacement can match.")
                + " Nothing was written. Pass method= (e.g. '2022' or "
                  "'1979'); an empty cycles list with a real method is the "
                  "valid 'this run found no cycles, clear the store' call.")

        own = conn is None
        if own:
            conn = dbwrite.open_write_connection(db_path)
        try:
            self._ensure_sleep_cycles_table(conn)
            # Delete every stored spelling of this recording's id, not just the
            # canonical one, or a row written under the bare folder name
            # survives and the insert adds a duplicate cycle.
            spellings = _subject_spellings(conn, 'sleep_cycles', subject,
                                           self.logger)
            placeholders = ",".join("?" * len(spellings))
            deleted = 0
            # Coverage rows describe the cycles being replaced, under every
            # stage and reject set: drop them all with the cycles.
            has_cov = bool(conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                "name='analysed_time_cycles'").fetchone())
            for m in method_vals:
                deleted += conn.execute(
                    f'DELETE FROM sleep_cycles WHERE subject IN ({placeholders}) '
                    f'AND method=?', (*spellings, m)).rowcount
                if has_cov:
                    conn.execute(
                        f'DELETE FROM analysed_time_cycles WHERE subject IN '
                        f'({placeholders}) AND method=?', (*spellings, m))
            conn.executemany('''
                INSERT INTO sleep_cycles
                    (subject, method, cycle_number, nrem_start, nrem_end,
                     rem_start, rem_end, nrem_dur_min, nrem_n23_dur_min,
                     rem_dur_min, cycle_dur_min, nrem_start_orig,
                     nrem_end_orig, rem_end_orig, time_base)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                (subject, c['method'], c['cycle_number'],
                 c['nrem_start_sec'], c['nrem_end_sec'],
                 c['nrem_end_sec'], c['rem_end_sec'],
                 c['nrem_dur_min'], c['nrem_n23_dur_min'],
                 c['rem_dur_min'], c['cycle_dur_min'],
                 c.get('nrem_start_orig', c['nrem_start_sec']),
                 c.get('nrem_end_orig', c['nrem_end_sec']),
                 c.get('rem_end_orig', c['rem_end_sec']),
                 c.get('time_base', TIME_BASE_ORIGINAL))
                for c in cycles])
            conn.commit()
            if cycles:
                self.logger.info(
                    f"Stored {len(cycles)} cycle(s) for subject "
                    f"'{subject}' in {db_path} (replacing {deleted} "
                    f"previously stored row(s)).")
            else:
                self.logger.info(
                    "No cycles to store for subject '%s' in %s; removed %d "
                    "previously stored row(s) for method(s) %s so the table "
                    "describes this run.", subject, db_path, deleted,
                    sorted(str(m) for m in method_vals))
        finally:
            if own:
                conn.close()
        return len(cycles)

    @staticmethod
    def _ensure_stage_durations_table(conn):
        """Create the ``stage_durations`` table if it does not exist.

        One row per subject holding per-stage minutes derived from the
        hypnogram, so downstream analysis can rely on ``neural_events.db`` alone
        instead of separate stage-summary CSVs.
        """
        conn.execute('''
        CREATE TABLE IF NOT EXISTS stage_durations (
            subject TEXT,
            epoch_length REAL,
            wake_min REAL,
            n1_min REAL,
            n2_min REAL,
            n3_min REAL,
            rem_min REAL,
            artefact_min REAL,
            total_min REAL,
            PRIMARY KEY (subject)
        )''')
        # time_base: 'original' (full-night hypnogram, or an uncut file) or
        # 'cut' (summed from a cut file's exact epochs because no full-night
        # hypnogram exists). NULL on rows written before 4.5.
        existing = {r[1] for r in conn.execute(
            'PRAGMA table_info(stage_durations)').fetchall()}
        if 'time_base' not in existing:
            conn.execute(
                'ALTER TABLE stage_durations ADD COLUMN time_base TEXT')

    def store_stage_durations(self, stage_durations, db_path, subject=None,
                              conn=None):
        """Upsert per-stage sleep durations into the ``stage_durations`` table.

        The existing row for ``subject`` is deleted then re-inserted so reruns
        stay idempotent (matching :meth:`store_cycles_to_database`).

        Parameters
        ----------
        stage_durations : dict
            Duration summary as returned by :func:`compute_stage_durations`,
            optionally with ``time_base`` (default ``'original'``). Its
            ``epoch_length`` is stored as given: 30 for a grid or a full-night
            hypnogram, the median epoch duration when the minutes were summed
            from a cut file's variable-length epochs. That median describes
            the file; never multiply an epoch count by it.
        db_path : str
            Path to the ``neural_events.db`` SQLite database.
        subject : str, optional
            Subject identifier. Falls back to ``self.subject`` (then ``''``)
            exactly like the cycle-storage methods.
        conn : sqlite3.Connection, optional
            An already-open connection **on ``db_path``** (not checked at
            runtime; it is a caller contract). When supplied, the caller owns
            closing it. When ``None`` a connection is opened via
            :func:`~turtlewave_hdEEG.dbwrite.open_write_connection` and closed
            here.

        Returns
        -------
        int
            Number of rows written (always 1).
        """
        # Same canonical spelling as write_cycles_to_database; see there.
        subject = normalize_subject(
            subject if subject is not None else (self.subject or ''))
        own = conn is None
        if own:
            conn = dbwrite.open_write_connection(db_path)
        try:
            self._ensure_stage_durations_table(conn)
            # Same as write_cycles_to_database: match every stored spelling of
            # this recording's id. stage_durations is PRIMARY KEY (subject), so
            # a missed old-spelling row is a second row for one recording and
            # doubles any SUM over the table.
            spellings = _subject_spellings(conn, 'stage_durations', subject,
                                           self.logger)
            conn.execute(
                'DELETE FROM stage_durations WHERE subject IN (%s)'
                % ",".join("?" * len(spellings)), spellings)
            conn.execute('''
                INSERT INTO stage_durations
                    (subject, epoch_length, wake_min, n1_min, n2_min, n3_min,
                     rem_min, artefact_min, total_min, time_base)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                subject,
                stage_durations['epoch_length'],
                stage_durations['wake_min'],
                stage_durations['n1_min'],
                stage_durations['n2_min'],
                stage_durations['n3_min'],
                stage_durations['rem_min'],
                stage_durations['artefact_min'],
                stage_durations['total_min'],
                stage_durations.get('time_base', TIME_BASE_ORIGINAL)))
            conn.commit()
            self.logger.info(
                f"Stored stage durations for subject '{subject}' in {db_path} "
                f"(total {stage_durations['total_min']:.1f} min).")
        finally:
            if own:
                conn.close()
        return 1

    def tag_events_with_cycles(self, cycles, db_path=None, conn=None,
                               run_id=None):
        """Replace the ``cycle`` column of the ``events`` table in one scope.

        The ``cycle`` column is first cleared across the whole scope, then each
        event is tagged by testing its ``start_time`` against the cycle spans;
        both happen in one transaction. Clearing first is what makes a re-run
        *replace* rather than merge: the per-cycle ``UPDATE``s only touch rows
        that fall inside a new span, so without the clear an event tagged by a
        previous run whose spans have since moved (a different ``wake_thresh``,
        rescored epochs) would keep its old, now-wrong cycle number. Events
        outside every cycle end up ``cycle=NULL``. The full cycle span (NREM
        start .. next cycle start, or recording end) is used so events in the
        inter-NREM segment are tagged too.

        Passing an empty ``cycles`` list is therefore *not* a no-op: it clears
        the scope (every ``cycle`` value in it becomes NULL) and tags nothing,
        which is the right result when a re-run detects no cycles at all and
        the previous run's tags would otherwise survive. The return value is
        still an ``int`` -- 0, the number of rows *tagged* -- so a caller
        cannot tell a clear from a no-op by the return; the cleared count is
        logged at INFO.

        Parameters
        ----------
        cycles : list of dict
            Detected cycles, as returned by :meth:`detect`. Only
            ``cycle_number``, ``nrem_start_sec`` and ``rem_end_sec`` are read,
            so a caller can rebuild them from stored ``sleep_cycles`` rows
            instead of re-detecting (see
            :func:`turtlewave_hdEEG.dbwrite.tag_run_cycles`).
        db_path : str, optional
            Path to the ``neural_events.db`` SQLite database. Only used when
            ``conn`` is ``None``; ignored otherwise.
        conn : sqlite3.Connection, optional
            An already-open connection **on ``db_path``** (not checked at
            runtime; it is a caller contract). When supplied, the caller owns
            closing it. When ``None`` a connection is opened via
            :func:`~turtlewave_hdEEG.dbwrite.open_write_connection` and closed
            here.
        run_id : str, optional
            When given, only rows carrying this ``events.run_id`` are tagged.
            A detection run passes its own id so it annotates the rows it just
            wrote and leaves every other run's ``cycle`` value alone --
            without it, tagging is a table-wide ``UPDATE`` and one detector's
            finalize step silently renumbers every other detector's events
            (harmlessly when the cycles agree, wrongly when they were computed
            from different scoring). ``None`` (the backfill case) tags the
            whole table, which is what a backfill is for.

        Returns
        -------
        int
            Number of event rows tagged. Zero when ``cycles`` is empty, even
            though rows may have been cleared in that call.
        """
        own = conn is None
        if own:
            conn = dbwrite.open_write_connection(db_path)
        try:
            self._ensure_sleep_cycles_table(conn)
            scoped = run_id is not None
            # Clear the scope before re-tagging, in the same transaction as the
            # per-cycle updates below, so the column is never left holding a
            # mix of this run's numbering and a previous run's. Restricted to
            # rows that actually carry a tag, which costs nothing extra and
            # makes rowcount an honest count of previously tagged rows.
            clear_sql = 'UPDATE events SET cycle=NULL WHERE cycle IS NOT NULL'
            clear_params = []
            if scoped:
                clear_sql += ' AND run_id = ?'
                clear_params.append(str(run_id))
            cleared = conn.execute(clear_sql, clear_params).rowcount
            total = 0
            for idx, cyc in enumerate(cycles):
                lo = cyc['nrem_start_sec']
                if idx + 1 < len(cycles):
                    hi = cycles[idx + 1]['nrem_start_sec']
                    where = 'start_time >= ? AND start_time < ?'
                else:
                    hi = cyc['rem_end_sec']
                    where = 'start_time >= ? AND start_time <= ?'
                params = [str(cyc['cycle_number']), lo, hi]
                if scoped:
                    where += ' AND run_id = ?'
                    params.append(str(run_id))
                cur = conn.execute(
                    f'UPDATE events SET cycle=? WHERE {where}', params)
                total += cur.rowcount
            conn.commit()
            self.logger.info(
                "Cleared %d previous cycle tag(s), then tagged %d event(s) "
                "with a cycle number%s.", cleared, total,
                f" (run_id={run_id})" if scoped else " (whole table)")
        finally:
            if own:
                conn.close()
        return total

    def run(self, db_path, method='2022', write_xml=True, subject=None,
            epoch_length=30, wake_thresh=10, nrem_min=30, rem_min=10,
            conn=None, run_id=None, tag_events=True, nrem_onset='n2n3',
            timeline='auto', coverage_floor_min=5.0, stages=None,
            reject_types=None):
        """Detect cycles, then persist to XML and the database.

        This single entry point serves both backfilling an existing
        ``neural_events.db`` and tagging a freshly detected one. Per-stage sleep
        durations are always written (via :meth:`store_stage_durations`), even
        when no cycles are detected, since an all-wake or unscorable night still
        has stage durations.

        Every store is a *replacement* and all of them run whether or not this
        run detected any cycles: the ``sleep_cycles`` rows for
        ``(subject, method)``, the XML cycle markers (when ``write_xml``) and
        ``events.cycle`` (when ``tag_events``) are cleared before the new
        values, if any, go in. A re-run that finds no cycles therefore leaves
        all three empty and agreeing, instead of an emptied ``events.cycle``
        beside a stale ``sleep_cycles`` table and stale XML markers.

        On a cut recording (a timeline sidecar beside the annotation file, see
        ``timeline``) cycles and stage durations come from the full-night
        hypnogram at 30 s: :func:`detect_cycles` and
        :func:`compute_stage_durations` run on ``fullnight_stages``, and the
        cycle bounds are then moved onto the cut file (:func:`_cycles_to_cut`:
        ``nrem_start_sec`` / ``nrem_end_sec`` / ``rem_end_sec`` in cut seconds
        on exact-epoch edges, ``*_orig`` on the original night, ``*_min`` and
        ``*_epoch`` full-night, ``time_base='cut'``). XML markers and
        ``events.cycle`` use the cut seconds, and one ``analysed_time_cycles``
        row per cycle x stage records how much of each cycle's full-night
        stage time survived the cut and the artefact masks
        (:func:`turtlewave_hdEEG.dbwrite.store_cycle_analysed_time`). A cut
        recording staged from its stage events has no full-night hypnogram:
        its stage durations are summed from the cut file's epoch durations
        (``time_base='cut'``, ``epoch_length`` the median epoch duration), no
        cycles are detected (logged at ERROR) and the cycle stores are
        cleared.

        Parameters
        ----------
        conn : sqlite3.Connection, optional
            An already-open connection **on ``db_path``** (not checked at
            runtime; it is a caller contract), passed straight through to the
            three storage methods so one whole run shares one connection. When
            supplied the caller owns closing it; when ``None`` this method
            opens one connection for the three writes and closes it on exit.
        run_id : str, optional
            Passed to :meth:`tag_events_with_cycles` so a detection run tags
            only the rows it wrote. Default ``None`` (tag every row).
        tag_events : bool, optional
            When False, cycles and stage durations are stored but
            ``events.cycle`` is left alone. When True the column is rewritten
            for the scope even if no cycles were detected -- tagging clears
            before it writes, so a previous run's tags never survive. Used by
            :func:`turtlewave_hdEEG.dbwrite.ensure_cycles_populated`, which
            runs BEFORE a detection's channel loop -- there are no rows to tag
            yet, and tagging then would rewrite an earlier run's. Default
            ``True``.

        timeline : {'auto', 'none'} or RecordingTimeline, optional
            ``'auto'`` (default) uses the sidecar beside the annotation file
            when there is one and raises
            :class:`~turtlewave_hdEEG.timeline.SidecarMismatchError` when it
            no longer matches the XML; ``'none'`` ignores any sidecar (only
            valid for uniform epochs); a ``RecordingTimeline`` is used as
            given.
        coverage_floor_min : float, optional
            ``analysed_time_cycles.low_coverage`` is set when a cycle x stage
            has less than this many analysed minutes. Default 5.0.
        stages : sequence of str or None, optional
            Stages given ``analysed_time_cycles`` rows. ``None`` uses
            :data:`DEFAULT_CYCLE_STAGES`.
        reject_types : sequence of str or None, optional
            Reject set for ``analysed_time_cycles`` (part of its key). ``None``
            uses the library default
            (:func:`turtlewave_hdEEG.utils.resolve_reject_types`).

        Returns
        -------
        list of dict
            The detected cycles (also stored in the DB); cut-file seconds and
            ``*_orig`` keys on a cut recording.

        Raises
        ------
        FileNotFoundError
            If ``db_path`` does not exist and no ``conn`` was supplied. This is
            a post-detection step; it annotates an existing database and never
            creates one.
        ValueError
            If ``self.annotations`` is None, or if its hypnogram is
            unscorable: empty, or every epoch ``-1`` (an epoch grid with no
            scoring saved -- ``get_hypnogram`` maps Undefined/Unknown/
            Artefact/Movement to ``-1``). Either shape is a wrong or unscored
            annotation file rather than a night without cycles, so it fails
            loudly and writes nothing -- silently continuing would clear every
            existing ``events.cycle`` tag, store a 100%-artefact
            ``stage_durations`` row, and report success with zero cycles. A
            scored night with no cycles (all Wake) is not refused. Also when
            the epochs vary in length and there is no usable timeline (see
            :meth:`_resolve_timeline`).
        timeline.SidecarMismatchError
            ``timeline='auto'`` found a sidecar that does not match the XML.
        """
        if self.annotations is None:
            raise ValueError("annotations are required for cycle detection")

        tl = self._resolve_timeline(timeline)

        if tl is not None and tl.stages:
            # Full-night path: cycles and durations on the original night.
            hypnogram = fullnight_hypnogram_codes(tl)
            self._refuse_unscorable(hypnogram, 'the full-night hypnogram in '
                                    'the timeline sidecar')
            cycles_full = detect_cycles(
                hypnogram, epoch_length=tl.epoch_length,
                wake_thresh=wake_thresh, nrem_min=nrem_min, method=method,
                rem_min=rem_min, nrem_onset=nrem_onset)
            self.logger.info(
                f"Detected {len(cycles_full)} cycle(s) using method "
                f"'{method}' on the full-night hypnogram.")
            cycles = _cycles_to_cut(cycles_full, tl, self.annotations)
            stage_durations = compute_stage_durations(
                hypnogram, epoch_length=tl.epoch_length)
            stage_durations['time_base'] = TIME_BASE_ORIGINAL
        else:
            # Read the hypnogram once and reuse it for both cycle detection
            # and stage-duration accounting.
            hypnogram = self.annotations.get_hypnogram()
            self._refuse_unscorable(hypnogram, 'the annotation file')
            if tl is None:
                cycles = self.detect(
                    method=method, epoch_length=epoch_length,
                    wake_thresh=wake_thresh, nrem_min=nrem_min,
                    rem_min=rem_min, hypnogram=hypnogram,
                    nrem_onset=nrem_onset)
                stage_durations = compute_stage_durations(
                    hypnogram, epoch_length=epoch_length)
                stage_durations['time_base'] = TIME_BASE_ORIGINAL
            else:
                # Staged from stage events: no full-night hypnogram, so no
                # cycles (logged once by _resolve_timeline) and stage
                # durations from the cut file's own epochs.
                from .timeline import nominal_epoch_length
                cycles = []
                stage_durations = compute_stage_durations(
                    hypnogram,
                    epoch_length=nominal_epoch_length(self.annotations,
                                                      epoch_length),
                    durations=self.annotations.epoch_durations())
                stage_durations['time_base'] = TIME_BASE_CUT

        own = conn is None
        if own:
            # Backfill onto an existing database only; never create one.
            # Skipped when the caller supplied a connection: they already
            # opened the database, so it exists by construction.
            _require_existing_db(db_path)
            conn = dbwrite.open_write_connection(db_path)
        try:
            # Every write is a REPLACEMENT and runs unconditionally, so
            # sleep_cycles, events.cycle, the XML markers and
            # analysed_time_cycles always describe the same run. Gating them
            # on `cycles` was the 4.3.1 defect: a re-run detecting none
            # cleared events.cycle while the previous run's sleep_cycles rows
            # and XML markers stayed in place.
            if write_xml:
                try:
                    self.write_cycle_markers(cycles)
                except Exception as e:
                    self.logger.warning(
                        f"Cycle-marker writing skipped: {e}")
            self.store_cycles_to_database(cycles, db_path, subject=subject,
                                          method=method, conn=conn)
            if not cycles:
                self.logger.info(
                    "No cycles detected for method '%s'; any previously "
                    "stored cycles were removed and stage durations were "
                    "written.", method)

            if tag_events:
                # Called even with no cycles: tagging clears the scope first,
                # so this is also what removes tags left behind by a previous
                # run whose thresholds did find cycles.
                self.tag_events_with_cycles(cycles, db_path, conn=conn,
                                            run_id=run_id)

            self.store_stage_durations(
                stage_durations, db_path, subject=subject, conn=conn)

            if tl is not None:
                dbwrite.store_cycle_analysed_time(
                    conn, normalize_subject(
                        subject if subject is not None
                        else (self.subject or '')),
                    method, cycles, tl, self.annotations,
                    stages=list(stages) if stages else
                    list(DEFAULT_CYCLE_STAGES),
                    reject_types=reject_types,
                    coverage_floor_min=coverage_floor_min,
                    annotation_file=getattr(self.annotations, 'annot_file',
                                            None),
                    logger=self.logger)
        finally:
            if own:
                conn.close()

        return cycles

    @staticmethod
    def _refuse_unscorable(hypnogram, what):
        """Raise on an empty or entirely unscored hypnogram.

        Two shapes mean "the wrong or an unscored annotation file", not "a
        night without cycles": no epochs at all, and epochs that are all -1
        (``get_hypnogram`` maps Undefined, Unknown, Artefact and Movement to
        -1). Proceeding would clear every existing events.cycle tag (tagging
        clears before it writes) and store a 100 % artefact stage_durations
        row while reporting success. A scored night with no cycles (all Wake)
        contains 0s and passes.

        Parameters
        ----------
        hypnogram : sequence of int
        what : str
            Where the hypnogram came from, for the message.

        Raises
        ------
        ValueError
        """
        if not hypnogram:
            raise ValueError(
                f"{what} has an empty hypnogram, so no cycles or stage "
                f"durations can be computed; nothing was written and any "
                f"existing events.cycle tags were left alone. Check that this "
                f"is the right XML and that it has been scored.")
        if all(stage == -1 for stage in hypnogram):
            raise ValueError(
                f"none of the {len(hypnogram)} epochs in {what} carries a "
                f"sleep stage (every epoch reads as Undefined/Unknown/"
                f"Artefact/Movement), so no cycles or stage durations can be "
                f"computed; nothing was written and any existing events.cycle "
                f"tags were left alone. Check that this is the right XML and "
                f"that its scoring has been saved.")

def finalize_cycles_and_durations(
        annotations, db_path, subject=None,
        methods=('2022', '1979'), tag_method='2022',
        write_xml=True, plot=False, plot_path=None,
        epoch_length=30, wake_thresh=10, nrem_min=30, rem_min=10,
        log_level=logging.INFO, conn=None, run_id=None, tag_events=True,
        nrem_onset='n2n3', timeline='auto', coverage_floor_min=5.0,
        stages=None, reject_types=None):
    """Populate ``neural_events.db`` with sleep cycles + stage durations.

    The explicit post-detection finalize step. Run it once after event
    detection: it detects sleep cycles for every method in ``methods``, stores
    them in ``sleep_cycles`` (both definitions coexist, keyed by
    ``(subject, method)``), writes per-stage durations to ``stage_durations``,
    tags ``events.cycle`` and (optionally) writes cycle markers into the
    annotation XML, and can emit the hypnogram/cycle PNG. Every write is a
    replacement of this subject's previous values rather than an addition to
    them -- including when a run detects no cycles, which empties all three
    stores instead of leaving the earlier run's rows and markers behind -- so
    re-running is safe and always leaves the three agreeing.

    Because :meth:`ParalCycles.tag_events_with_cycles` rewrites *all* event rows
    by time window regardless of method ("last run wins"), ``methods`` is
    reordered so ``tag_method`` runs LAST. That makes ``tag_method``'s cycle
    numbering the one that survives in ``events.cycle`` and — when
    ``write_xml`` is True — in the XML markers, deterministically.

    Parameters
    ----------
    annotations : CustomAnnotations
        Annotation wrapper exposing ``get_hypnogram`` / ``epochs``.
    db_path : str
        Path to the ``neural_events.db`` SQLite database.
    subject : str, optional
        Subject identifier stored in ``sleep_cycles`` / ``stage_durations``.
    methods : sequence of str, optional
        Cycle definitions to detect and store (default ``('2022', '1979')``).
    tag_method : str or None, optional
        The method whose cycle numbering owns ``events.cycle`` and the XML
        markers. Forced to run last among ``methods`` (default ``'2022'``).
        **Must be one of ``methods``**, or ``None`` to tag nothing and write
        no markers -- a ``tag_method`` outside ``methods`` raises (see
        ``Raises``).
    write_xml : bool, optional
        Write cycle markers to the annotation XML for ``tag_method`` only
        (default True). The XML is rewritten on every such call, a zero-cycle
        run included: the markers are *cleared* before the new ones (if any)
        are written, and Wonambi saves the file on the clear, stamping a fresh
        ``modified`` timestamp. A night that produced no cycles therefore still
        shows a changed annotation file -- correctly, since its old markers
        were removed. Set False to leave the XML alone entirely.
    plot : bool, optional
        If True, also write the hypnogram/cycle PNG (default False; plotting is
        normally a separate call).
    plot_path : str, optional
        Destination PNG when ``plot`` is True. Defaults to a file next to
        ``db_path`` named
        ``{subject or 'hypnogram'}_hypnogram_cycles_{methods joined by _vs_}.png``.
    epoch_length : float, optional
        Epoch duration in seconds (default 30).
    wake_thresh : int, optional
        Max Wake epochs absorbed into surrounding NREM (default 10).
    nrem_min : int, optional
        Min NREM epochs to count as an NREM period (default 30).
    rem_min : int, optional
        Min REM epochs to close a cycle under method ``'1979'`` (default 10).
    log_level : int, optional
        Logging level for the internal ``ParalCycles`` (default
        ``logging.INFO``).
    conn : sqlite3.Connection, optional
        An already-open write connection **on ``db_path``**. Pass it when a
        detector is holding one: opening a second write connection while the
        first is open is a writer-vs-writer collision under DELETE journal
        mode, which is exactly the network-drive failure 4.0.2 was cut for.
        When supplied the caller owns closing it, and the
        database-exists check is skipped (the caller has it open, so it
        exists). Default ``None`` (open and close one here).
    run_id : str, optional
        Passed through to the event tagging so a detection run tags only its
        own rows. Default ``None`` (tag every row -- the backfill case).
    nrem_onset : {'n2n3', 'any'}, optional
        Where a ``'2022'`` NREM period starts; see :func:`detect_cycles`.
        Default ``'n2n3'``.
    tag_events : bool, optional
        When False, cycles and stage durations are stored but ``events.cycle``
        is not touched. Default ``True``.
    timeline : {'auto', 'none'} or RecordingTimeline, optional
        Resolved once for all methods; see :meth:`ParalCycles.run`. With a
        full-night timeline the plot shows the full-night hypnogram and the
        cycles on the original night. Default ``'auto'``.
    coverage_floor_min : float, optional
        Low-coverage floor for ``analysed_time_cycles``, in minutes. Default
        5.0.
    stages : sequence of str or None, optional
        Stages given ``analysed_time_cycles`` rows; ``None`` uses
        :data:`DEFAULT_CYCLE_STAGES`.
    reject_types : sequence of str or None, optional
        Reject set for ``analysed_time_cycles``. ``None`` uses the library
        default.

    Returns
    -------
    dict
        ``{method: [cycle dicts]}`` for every method in ``methods``, in the
        original ``methods`` order (not the reordered execution order).

    Raises
    ------
    FileNotFoundError
        If ``db_path`` does not exist and no ``conn`` was supplied. This is
        the post-detection finalize step: it annotates an existing
        ``neural_events.db`` and never creates one, so a missing file is a
        wrong path rather than a database to create. Checked before
        connecting, since connecting would create it.
    ValueError
        If ``tag_method`` is neither ``None`` nor one of ``methods``. That
        combination used to be silently self-contradictory: the
        ``m == tag_method`` gate never fires, so NO XML markers are written at
        all, while ``events.cycle`` is still overwritten by whichever method
        happens to run last. The database and the XML then disagree about the
        cycle numbering with nothing recording which is which, and the mistake
        is a single misspelled argument. Pass ``tag_method=None`` if tagging
        nothing is what you meant.

        Also propagated from :meth:`ParalCycles.run` when ``annotations``
        yields an unscorable hypnogram -- empty, or every epoch ``-1`` (an
        epoch grid whose scoring was never saved). Nothing is written and no
        existing ``events.cycle`` tag is cleared, so a batch caller can count
        the subject as failed and move on. A scored night with no cycles (all
        Wake) is not refused: it stores its stage durations and clears its
        stale tags.

    Notes
    -----
    Only ``tag_method`` writes XML markers, so the XML never ends up with two
    conflicting cycle numberings. All methods are stored in ``sleep_cycles``.
    """
    pc = ParalCycles(annotations=annotations, subject=subject,
                     log_level=log_level)

    methods = list(methods)
    if tag_method is not None and tag_method not in methods:
        raise ValueError(
            f"tag_method={tag_method!r} is not one of methods={methods}. It "
            f"names the cycle definition that owns events.cycle and the XML "
            f"markers, so a value outside the list writes NO markers at all "
            f"while events.cycle silently takes whichever method ran last -- "
            f"a database and an annotation file that disagree, from one "
            f"misspelled argument. Use one of {methods}, or tag_method=None "
            f"to store both definitions and tag nothing.")

    # Reorder so tag_method runs LAST (tagging is "last run wins" whenever
    # more than one method tags). With tag_method=None nothing tags.
    if tag_method in methods:
        run_order = [m for m in methods if m != tag_method] + [tag_method]
    else:
        run_order = list(methods)

    # One connection for the whole subject: every method's cycle storage, event
    # tagging and stage-duration write share it. Previously each of the three
    # storage methods opened and closed its own untimed connection per method
    # (six connect/close cycles for the default two methods), and on a WAL
    # database each close deletes and each connect recreates the -wal/-shm
    # sidecars -- the operation that fails on a network share.
    # One timeline for every method, so the sidecar is read and checked once
    # and its notice is logged once per run, not once per method.
    tl = pc._resolve_timeline(timeline)

    cycles_by_method = {}
    own_conn = conn is None
    if own_conn:
        # Fail fast before connecting: connecting would create the file. This
        # is a backfill onto an existing neural_events.db, so a missing one is
        # a wrong path, not a database to create. Skipped when the caller
        # supplied a connection -- they have the database open already.
        _require_existing_db(db_path)
        conn = dbwrite.open_write_connection(db_path, logger=pc.logger)
    try:
        for m in run_order:
            cycles = pc.run(
                db_path, method=m, write_xml=(write_xml and m == tag_method),
                subject=subject, epoch_length=epoch_length,
                wake_thresh=wake_thresh, nrem_min=nrem_min, rem_min=rem_min,
                conn=conn, run_id=run_id,
                tag_events=(tag_events and m == tag_method),
                nrem_onset=nrem_onset,
                timeline=tl if tl is not None else 'none',
                coverage_floor_min=coverage_floor_min, stages=stages,
                reject_types=reject_types)
            cycles_by_method[m] = cycles
    finally:
        if own_conn:
            conn.close()

    # Return in the caller's requested method order for stable plotting.
    cycles_by_method = {m: cycles_by_method[m] for m in methods}

    if plot:
        if plot_path is None:
            import os
            db_dir = os.path.dirname(os.path.abspath(db_path))
            stem = subject if subject else 'hypnogram'
            fname = (f"{stem}_hypnogram_cycles_"
                     f"{'_vs_'.join(str(m) for m in methods)}.png")
            plot_path = os.path.join(db_dir, fname)
        # Imported lazily so cycleprocessor stays matplotlib-free at module
        # load (headless-safe library import).
        from .cycleplot import plot_from_annotations
        if tl is not None and tl.stages:
            # Full-night picture: the hypnogram the cycles were detected on,
            # with the bounds back on the original night.
            orig = {m: [dict(c, nrem_start_sec=c['nrem_start_orig'],
                             nrem_end_sec=c['nrem_end_orig'],
                             rem_end_sec=c['rem_end_orig']) for c in cyc]
                    for m, cyc in cycles_by_method.items()}
            plot_from_annotations(annotations, orig, plot_path,
                                  epoch_length=tl.epoch_length,
                                  subject=subject,
                                  hypnogram=fullnight_hypnogram_codes(tl))
        else:
            plot_from_annotations(annotations, cycles_by_method, plot_path,
                                  epoch_length=epoch_length, subject=subject)
        pc.logger.info("Cycle plot written to %s", plot_path)

    return cycles_by_method
