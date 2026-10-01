"""
Annotations module for turtlewave_hdEEG
Provides tools to create and save annotations using event information from EEGLAB
"""

from pathlib import Path
import datetime
import tempfile
import os
import time
import logging
from collections import namedtuple
from xml.etree.ElementTree import SubElement
import numpy as np
from wonambi.attr import Annotations as WonambiAnnotations
from wonambi.attr.annotations import create_empty_annotations

from .timeline import (RecordingTimeline, stage_events_from_header,
                       stage_event_intervals, exact_cut_epochs,
                       stage_name_for_code, sidecar_path)

logger = logging.getLogger('turtlewave_hdEEG.annotation')

#: Staging-source names used in the log and in the timeline sidecar.
STAGING_SOURCE_HEADER = 'etc.stages (header, as stored)'
STAGING_SOURCE_TIME_MAP = 'etc.stages through the boundary time map'
STAGING_SOURCE_EVENTS = 'stage events'

#: Share of stage events allowed to disagree with the time-mapped
#: ``etc.stages`` before a WARNING is logged.
STAGE_EVENT_DISAGREEMENT_WARN = 0.01

#: Share of stage events disagreeing with the best reading of a cut file's
#: ``etc.stages`` above which that reading is rejected and the stage events
#: themselves become the staging source. Higher than the WARNING threshold on
#: purpose: a few re-scored or mistyped events should not discard the
#: full-night hypnogram, but a boundary table that misstates the removed time
#: shifts every epoch after the error and disagrees wholesale.
STAGE_EVENT_DISAGREEMENT_REJECT = 0.20

#: The staging chosen by :meth:`XLAnnotations._choose_staging_source`.
#: ``codes`` (one per 30 s epoch from time 0) is set for the as-stored header
#: source and goes through Wonambi's grid import; ``epochs`` (exact
#: ``(start, end, code, orig_epoch)`` tuples) is set for the time-map and
#: stage-event sources and goes through :meth:`XLAnnotations._write_exact_epochs`.
#: ``timeline`` is the map written to the sidecar with exact epochs.
StagingChoice = namedtuple('StagingChoice', 'source codes epochs timeline')

#: Numeric hypnogram codes used by :meth:`CustomAnnotations.get_hypnogram`.
HYPNOGRAM_CODES = {'Wake': 0, 'NREM1': 1, 'NREM2': 2, 'NREM3': 3, 'REM': 4,
                   'Artefact': -1, 'Movement': -1, 'Unknown': -1,
                   'Undefined': -1}

class XLAnnotations:
    """Simplified annotations for large datasets"""

    def __init__(self, dataset, annot_file,rater_name="Anon"):
        """
        Initialize annotations object.

        Parameters
        ----------
        dataset : LargeDataset
            Dataset to associate with annotations.
        annot_file : str
            Path to the annotation file.
        """
        self.dataset = dataset
        self.annot_file = annot_file
        self.rater_name = rater_name

        # Create or load annotations
        if not Path(annot_file).exists():
            self.annotations = create_empty_annotations(annot_file, dataset)
            self.annotations = WonambiAnnotations(annot_file)
            self.annotations.add_rater(self.rater_name)
            print(f"Created a new annotation object for {annot_file}")
        else:
            # Load existing annotations
            self.annotations = WonambiAnnotations(annot_file)
            if self.rater_name not in self.annotations.raters:
                self.annotations.add_rater(self.rater_name)

            print(f"Loaded existing annotation file: {annot_file}")


    #: Half-width, in seconds, of the ``Artefact`` window written around an
    #: EEGLAB ``boundary`` (splice) marker. The window is
    #: ``[onset - BOUNDARY_PAD_SECONDS, onset + BOUNDARY_PAD_SECONDS]``.
    BOUNDARY_PAD_SECONDS = 2.0

    def _recording_length_seconds(self):
        """Recording length in seconds, for clipping annotation windows.

        Returns
        -------
        float or None
            ``header['n_samples'] / sampling_rate`` when the header carries a
            sample count, else ``header['recording_duration']``, else ``None``
            when neither is available -- in which case callers must clip at
            zero only and leave the upper bound open.
        """
        header = getattr(self.dataset, 'header', {}) or {}
        s_freq = getattr(self.dataset, 'sampling_rate', None)

        n_samples = header.get('n_samples')
        if n_samples is not None and s_freq:
            try:
                return float(n_samples) / float(s_freq)
            except (TypeError, ValueError):
                pass

        duration = header.get('recording_duration')
        if duration is not None:
            try:
                return float(duration)
            except (TypeError, ValueError):
                pass

        return None

    def add_artefacts_from_events(self):
        """
        Add artefact, arousal and other exclusion annotations from the
        dataset's EEGLAB event information.

        Every event in ``dataset.header['event']`` is matched, case-insensitively,
        against the rules below and written as a Wonambi event on ``'(all)'``
        channels. ``boundary`` is tested first and its events are removed from
        the remaining masks, so a splice marker can never also be counted as,
        say, a movement. The other rules are independent of each other: a type
        that matches two of them is written under both labels.

        | Rule        | Event type matches (lower-cased)                        | Label    |
        |-------------|---------------------------------------------------------|----------|
        | boundary    | equals `boundary`                                       | Artefact |
        | reject      | contains `reject`                                       | Artefact |
        | arousal     | contains `arousal`                                      | Arousal  |
        | respiratory | contains `hypopnea`/`apnea`/`spo2desat`, or equals `rera`, after removing spaces and underscores | Resp |
        | movement    | contains `move`/`leg`, or equals `lklr`/`lkud`          | Move     |
        | snore       | contains `snor`/`jaw`                                   | Snore    |

        ``apnea`` covers ``obstructiveapnea``, ``centralapnea`` and
        ``mixedapnea``; spaces and underscores are ignored for this rule only,
        so ``spo2 desat`` and ``central apnea`` match. ``rera`` is an exact
        match so that ``arousal5rera`` (or ``arousal 4 rera``) is an Arousal
        only. ``unsure respiratory event`` is not matched. ``spo2artifact``, ``slpcycle*`` and ``slpep*`` match
        no rule and are not imported.

        Event timing comes from ``onsets`` and ``durations`` (both in samples;
        divided by ``dataset.sampling_rate``). A missing, ``None`` or
        non-numeric duration falls back to 1.0 s. Boundary events are the one
        exception -- see Notes.

        Returns
        -------
        total_count : int
            Number of annotations written, summed over EVERY rule above (this
            used to count only Artefact + Arousal).
        execution_time : float
            Wall-clock seconds spent.

        Notes
        -----
        **Why boundary events are masked.** EEGLAB writes a ``boundary`` event
        at each point where a segment of data was cut out. The remaining samples
        are spliced together, so the signal steps discontinuously at that
        instant. On sub-11vi (107 splices x 257 channels) the median step across
        a splice was 1.5 uV, but 7 boundaries stepped more than 30 uV on at
        least one channel -- large enough to seed a false slow wave. Masking
        +/- 2.0 s around every splice also guarantees that no detected event,
        and in particular no slow-oscillation/spindle coupling phase estimate,
        spans a discontinuity. The cost measured on that subject was about 1.6%
        of analysable time.

        **Why the boundary's own duration is ignored.** For a ``boundary``,
        EEGLAB's ``duration`` field is the length of the data that was REMOVED,
        expressed in the ORIGINAL time base, not a span in the surviving
        recording. Using ``onset + duration`` as the window end therefore
        rejects perfectly good post-splice data -- about 36 minutes on one
        measured subject. The window is a fixed +/- ``BOUNDARY_PAD_SECONDS``
        around the onset instead.

        The window is clipped to ``[0, recording length]``, which covers the
        boundaries some files place at latency 0 or at the very end of the
        recording; a window that collapses to zero length after clipping is
        skipped rather than written.
        """

        start_time = time.time()

        # Check if event information exists in header
        if 'event' not in self.dataset.header:
            print("No event information found in dataset header.")
            end_time = time.time()
            print(f"Processing time: {end_time - start_time:.4f} seconds")
            return 0, end_time - start_time

        event_info = self.dataset.header['event']
        onsets = np.array(event_info.get('onsets', []))
        types = event_info.get('types', [])
        durations = np.array(event_info.get('durations', []))
        #isreject = [str(t).lower() == 'reject' for t in types] if types else []
        # Check if we have any events
        if len(onsets) == 0:
            print("No events found in dataset.")

            return 0, time.time() - start_time

        s_freq = self.dataset.sampling_rate
        # float64 explicitly: `np.ones_like(onsets)` inherited the onsets dtype,
        # so an integer latency array produced an integer duration array and the
        # assignment below truncated every sub-second duration to 0 -- a 0.4 s
        # arousal became a zero-length annotation.
        onset_seconds = np.asarray(onsets, dtype=np.float64) / float(s_freq)
        n_events = len(onset_seconds)
        duration_seconds = np.ones(n_events, dtype=np.float64)

        # `durations` may be shorter than `onsets` (or absent, or hold None /
        # non-numeric entries). Anything unusable keeps the 1.0 s fallback
        # instead of raising IndexError or TypeError.
        for i, raw in enumerate(durations[:n_events]):
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isnan(value):
                continue
            duration_seconds[i] = value / float(s_freq)

        end_seconds = onset_seconds + duration_seconds

        # Pre-compile type checks
        types_arr = np.array([str(t).lower() if t else '' for t in types[:len(onsets)]])

        # Rule 0, evaluated first: EEGLAB splice markers. Matched on the exact
        # type so an unrelated type merely containing the word is not swept in.
        boundary_mask = np.array([t.strip() == 'boundary' for t in types_arr],
                                 dtype=bool)
        not_boundary = ~boundary_mask

        # Respiratory names come spelled with and without separators
        # ('spo2desat', 'spo2 desat', 'SpO2_Desat'); drop them for that rule.
        resp_arr = np.char.replace(np.char.replace(types_arr, ' ', ''), '_', '')

        event_masks = {
            "Artefact": (np.char.find(types_arr, 'reject') != -1) & not_boundary,
            "Arousal": (np.char.find(types_arr, 'arousal') != -1) & not_boundary,
            # Matched on the type with spaces and underscores removed, so
            # 'spo2 desat' and 'central apnea' match. `rera` is matched
            # exactly: the masks are not mutually exclusive and a substring
            # match would also tag `arousal5rera` as Resp.
            "Resp": np.any([np.char.find(resp_arr, x) != -1 for x in ['hypopnea', 'apnea', 'spo2desat']]
                           + [resp_arr == 'rera'], axis=0) & not_boundary,
            "Move": np.any([np.char.find(types_arr, x) != -1 for x in ['move', 'leg']] + [types_arr == x for x in ['lklr', 'lkud']], axis=0) & not_boundary,
            "Snore": np.any([np.char.find(types_arr, x) != -1 for x in ['snor', 'jaw']], axis=0) & not_boundary
        }

        event_counts = {key: 0 for key in event_masks}
        event_counts["Boundary"] = 0

        # Boundary windows: fixed +/- BOUNDARY_PAD_SECONDS around the onset,
        # ignoring the event's own duration (see Notes), clipped to the
        # recording, and dropped if the clip leaves nothing.
        boundary_indices = np.where(boundary_mask)[0]
        if len(boundary_indices) > 0:
            pad = float(self.BOUNDARY_PAD_SECONDS)
            rec_length = self._recording_length_seconds()

            b_onsets = onset_seconds[boundary_indices]
            b_starts = np.maximum(b_onsets - pad, 0.0)
            b_ends = b_onsets + pad
            if rec_length is not None:
                b_starts = np.minimum(b_starts, rec_length)
                b_ends = np.minimum(b_ends, rec_length)

            keep = b_ends > b_starts
            n_skipped = int(np.count_nonzero(~keep))
            if n_skipped:
                print(f"Skipped {n_skipped} boundary event(s) whose "
                      f"+/-{pad:g}s window fell outside the recording.")
            if np.any(keep):
                success = self.add_annotations_batch(
                    label="Artefact",
                    start_times=b_starts[keep],
                    end_times=b_ends[keep],
                    channels=None
                )
                if success:
                    event_counts["Boundary"] += int(np.count_nonzero(keep))

        # Batch process annotations
        for event_type, mask in event_masks.items():
            indices = np.where(mask)[0]
            if len(indices) > 0:
                # Add annotations for the event type
                success = self.add_annotations_batch(
                    label=event_type,
                    start_times=onset_seconds[indices],
                    end_times=end_seconds[indices],
                    channels=None
                )
                if success:
                    event_counts[event_type] += len(indices)

        # Count EVERY written type: a file holding only Resp/Move/Snore events
        # (or only boundaries) used to be added to the in-memory tree and then
        # never saved, because the save gate summed Artefact + Arousal alone.
        total_count = sum(event_counts.values())

        if total_count > 0:
            self.annotations.save()
            print(
                f"Added {event_counts['Artefact']} artefact annotations and "
                f"{event_counts['Arousal']} arousal annotations from event information. "
                f"{event_counts['Resp']} respiratory events, "
                f"{event_counts['Move']} movement events, "
                f"{event_counts['Snore']} snore events, "
                f"{event_counts['Boundary']} boundary masks "
                f"(+/-{float(self.BOUNDARY_PAD_SECONDS):g}s, written as Artefact)."
            )
        else:
            print("No artefacts, arousals or other exclusions found in event "
                  "information.")

        execution_time = time.time() - start_time
        print(f"Processing time: {execution_time:.4f} seconds")
        return total_count, execution_time



    def add_stages_from_header(self):
        """
        Import sleep stages into the annotations.

        The staging source is chosen from the header and logged at INFO with
        its numbers (``turtlewave_hdEEG.annotation`` logger):

        | Condition | Staging source | Epochs written |
        |---|---|---|
        | no removed time, and ``30 * len(etc.stages) - T <= 30`` s, or ``T`` unknown | ``etc.stages`` as stored (unchanged behaviour) | 30 s grid, Wonambi ``import_staging`` |
        | boundary events removed data (any amount) | full-night reading through the time map, or the as-stored reading, whichever the stage events support (ties and no events: time map when consistent); see :meth:`_choose_reading` | time map: exact epochs; as stored: 30 s grid |
        | stage events present, and no usable map or no ``etc.stages`` | stage events, each ending at the next onset or the next splice | exact epochs, Undefined in gaps (including data after a splice whose stage event was cut) |
        | longer, and neither usable | nothing imported, ERROR logged, returns ``False`` | none |

        ``T`` is the signal length (``n_samples / sampling rate``). Exact
        epochs are whole-second, variable-length epochs tiling
        ``[0, int(T))`` (see :func:`~turtlewave_hdEEG.timeline.exact_cut_epochs`),
        written by :meth:`_write_exact_epochs`. With exact epochs the map and
        the epochs are also written to ``<annotation xml stem>_timeline.json``
        beside the annotation file (see
        :func:`~turtlewave_hdEEG.timeline.load_sidecar_for`) so that sleep
        cycles and stage durations can be computed on the full night.

        The rater name applied to the imported staging is taken from the instance
        attribute ``self.rater_name`` set at construction, not from an argument.

        Returns
        -------
        bool
            True if successful, False otherwise

        Notes
        -----
        A cut recording keeps its full-night ``etc.stages`` while the signal
        loses the removed data, so importing ``etc.stages`` as stored would put
        every stage after the first splice on the wrong signal. The time map
        adds back the data removed at each ``boundary`` event and cuts every
        surviving original epoch at the splices, so each exact epoch is one
        piece of one original epoch with that epoch's stage. Edges are rounded
        to whole seconds (halves up); a piece under about 0.5 s is dropped and
        its time goes to a neighbour. Never multiply an epoch count by 30 on
        such a file: use the epoch durations.
        """
        try:
            # Make sure we have a header with stages (or stage events)
            header = getattr(self.dataset, 'header', None)
            if header is None:
                print("No stages found in header")
                return False
            stage_events = None
            if 'stages' not in header:
                stage_events = self._header_stage_events()
                if not stage_events:
                    print("No stages found in header")
                    return False

            # Make sure we have an annotations object
            if not hasattr(self, 'annotations'):
                print("No annotations object available")
                return False

            choice = self._choose_staging_source(header, stage_events)
            if choice is None:
                return False

            if choice.epochs is not None:
                ok = self._write_exact_epochs(choice.epochs)
                if ok:
                    self._write_timeline_sidecar(choice.timeline, choice.source,
                                                 choice.epochs)
                return ok

            ok = self._import_stage_codes(choice.codes)
            if ok:
                stale = sidecar_path(self.annot_file)
                if stale.exists():
                    logger.warning(
                        f"{stale.name} exists beside the annotation file but "
                        f"this staging came from '{choice.source}' on a 30 s "
                        f"grid; the sidecar is stale. Delete it before "
                        f"computing sleep cycles.")
            return ok

        except Exception as e:
            print(f"Error importing stages from header: {e}")
            return False

    def _sampling_rate(self):
        """``dataset.sampling_rate``, else ``header['s_freq']``, else None."""
        s_freq = getattr(self.dataset, 'sampling_rate', None)
        if not s_freq:
            s_freq = (getattr(self.dataset, 'header', {}) or {}).get('s_freq')
        try:
            return float(s_freq) if s_freq else None
        except (TypeError, ValueError):
            return None

    def _signal_seconds(self):
        """Signal length in seconds and the sample count behind it.

        Returns
        -------
        (float or None, int or None)
            ``n_samples / sampling rate`` and ``n_samples``; falls back to
            ``header['recording_duration']`` (sample count then derived).
        """
        header = getattr(self.dataset, 'header', {}) or {}
        s_freq = self._sampling_rate()
        n_samples = header.get('n_samples')
        if n_samples is not None and s_freq:
            try:
                return float(n_samples) / s_freq, int(n_samples)
            except (TypeError, ValueError):
                pass
        duration = header.get('recording_duration')
        if duration is not None and s_freq:
            try:
                return float(duration), int(round(float(duration) * s_freq))
            except (TypeError, ValueError):
                pass
        return None, None

    def _header_stage_events(self):
        """Stage events (``ns``/``wake``/``n1``/``n2``/``n3``/``rem``) from the header."""
        header = getattr(self.dataset, 'header', {}) or {}
        s_freq = self._sampling_rate()
        if not s_freq or 'event' not in header:
            return []
        return stage_events_from_header(header.get('event'), s_freq)

    def _choose_staging_source(self, header, stage_events=None, epoch_length=30):
        """Pick the staging source for :meth:`add_stages_from_header`.

        Parameters
        ----------
        header : dict
            ``dataset.header``.
        stage_events : list of (float, str) or None
            Pre-computed stage events, or ``None`` to read them when needed.
        epoch_length : float
            Epoch length of ``etc.stages`` (and of the as-stored grid).

        Returns
        -------
        StagingChoice or None
            ``codes`` for the as-stored source, ``epochs`` and ``timeline``
            for the time-map and stage-event sources; ``None`` (ERROR logged)
            when no source gives aligned staging.
        """
        L = float(epoch_length)
        T, n_samples = self._signal_seconds()
        s_freq = self._sampling_rate()
        has_stages = 'stages' in header and header['stages'] is not None
        reason = None

        if has_stages:
            stages = header['stages']
            n_stages = len(stages)
            if T is None:
                logger.info(
                    f"Staging source: {STAGING_SOURCE_HEADER} "
                    f"({n_stages} epochs; signal length unknown)")
                return StagingChoice(STAGING_SOURCE_HEADER, stages, None, None)

            try:
                timeline = RecordingTimeline.from_header(
                    header, s_freq, n_samples, epoch_length=L)
            except Exception as e:
                logger.warning(f"Could not build the boundary time map: {e}")
                timeline = None
            removed = timeline.removed_seconds if timeline is not None else 0.0
            n_b = timeline.n_boundaries if timeline is not None else 0
            fits_signal = L * n_stages - T <= L

            if removed <= 0:
                # No removed time: the file's own time base is the scoring's.
                if fits_signal:
                    logger.info(
                        f"Staging source: {STAGING_SOURCE_HEADER} "
                        f"({n_stages} epochs; signal {T:.1f} s)")
                    if stage_events is None:
                        stage_events = self._header_stage_events()
                    self._warn_if_events_disagree(stages, stage_events, s_freq,
                                                  n_samples, L)
                    return StagingChoice(STAGING_SOURCE_HEADER, stages, None,
                                         None)
            else:
                chosen = self._choose_reading(timeline, stages, fits_signal,
                                              stage_events, s_freq, n_samples,
                                              T, L)
                if chosen is not None:
                    return chosen

            reason = (f"etc.stages has {n_stages} epochs "
                      f"({L * n_stages:.1f} s) but signal plus removed data is "
                      f"{T + removed:.1f} s "
                      f"({T:.1f} s + {removed:.1f} s at {n_b} boundary events)")
        else:
            reason = "the header has no etc.stages"

        if stage_events is None:
            stage_events = self._header_stage_events()
        if stage_events and T is not None:
            # The map is kept for its boundary table only: etc.stages was not
            # used, so it must not reach the sidecar as the full night.
            ev_timeline = RecordingTimeline.from_header(
                {'event': header.get('event')}, s_freq, n_samples,
                epoch_length=L)
            # Each event's interval also ends at the next splice: the data
            # after it belongs to a later part of the night, and when its own
            # stage event went with the cut it has no stage (Undefined).
            plain = stage_event_intervals(stage_events, T, L)
            clipped = stage_event_intervals(stage_events, T, L,
                                            splices=ev_timeline.cut_onsets)
            lost = (sum(e - s for s, e, _ in plain)
                    - sum(e - s for s, e, _ in clipped))
            epochs = exact_cut_epochs(clipped, ev_timeline.last_second)
            n_undef = sum(1 for e in epochs if e[2] == '?')
            logger.info(
                f"Staging source: {STAGING_SOURCE_EVENTS} "
                f"({len(stage_events)} stage events -> {len(epochs)} exact "
                f"epochs over {ev_timeline.last_second} s, {n_undef} of them "
                f"Undefined; {lost:.1f} s after a splice and before the next "
                f"stage event left Undefined; {reason})")
            return StagingChoice(STAGING_SOURCE_EVENTS, None, epochs,
                                 ev_timeline)

        logger.error(
            f"Staging not imported: {reason}, and there are no stage events "
            f"to fall back on. Importing etc.stages as stored would misalign "
            f"every stage after the first cut.")
        return None

    def _choose_reading(self, timeline, stages, fits_signal, stage_events,
                        s_freq, n_samples, T, L):
        """Choose between the two readings of ``etc.stages`` on a cut file.

        When boundary events removed data, ``etc.stages`` is either the full
        night (read through the time map) or already on the cut time base
        (read as stored). Neither known cleaning pipeline rewrites
        ``etc.stages`` after a cut, so the full-night reading is the default.

        Parameters
        ----------
        timeline : RecordingTimeline
            Time map with ``removed_seconds > 0``.
        stages : sequence
            ``etc.stages`` as stored.
        fits_signal : bool
            ``epoch_length * len(stages) - T <= epoch_length``: the as-stored
            reading is plausible.
        stage_events : list of (float, str) or None
            Stage events, read from the header when ``None``.
        s_freq : float
            Sampling frequency.
        n_samples : int
            Samples in the cut signal.
        T : float
            Signal length in seconds.
        L : float
            Epoch length in seconds.

        Returns
        -------
        StagingChoice or None
            Time map: exact ``epochs`` and the ``timeline``; as stored:
            ``codes`` only. ``None`` when neither reading is usable.

        Notes
        -----
        Candidates: the time map when it is consistent, or when
        ``etc.stages`` ends before the night does (a scorer who stopped early)
        and the stage events support it (disagreement at most
        ``STAGE_EVENT_DISAGREEMENT_WARN``); the as-stored reading when
        ``fits_signal``. ``etc.stages`` running past signal plus removed data
        never makes the map a candidate.
        With stage events, the candidate with the smaller share of
        disagreeing events wins, ties to the time map; the choice is logged
        with both scores. If the winner still disagrees on more than
        ``STAGE_EVENT_DISAGREEMENT_REJECT`` it is rejected (WARNING, returns
        ``None`` so the caller stages from the events); above
        ``STAGE_EVENT_DISAGREEMENT_WARN`` it is kept with a WARNING. Without stage events
        the time map wins when consistent; as stored is then used only with
        a WARNING, and a WARNING is also logged when both readings fit and at
        least half an epoch was removed, the range in which they label the
        grid differently.
        """
        if stage_events is None:
            stage_events = self._header_stage_events()
        map_cmp = timeline.compare_stage_events(stage_events) \
            if stage_events else (0, 0)
        stored_cmp = (0, 0)
        if stage_events and fits_signal:
            as_stored = RecordingTimeline([], [], s_freq, n_samples,
                                          stages=[str(x).strip() for x in stages],
                                          epoch_length=L)
            stored_cmp = as_stored.compare_stage_events(stage_events)

        def share_bad(cmp):
            return cmp[1] / cmp[0] if cmp[0] else 1.0

        # The map is a candidate when etc.stages covers the night, or when it
        # ends early (the scorer stopped before the end) and the stage events
        # support it. etc.stages running past signal plus removed data means
        # the boundary table does not account for the length: not a candidate.
        ends_early = (timeline.original_seconds - L * len(stages)
                      >= -timeline.consistency_tolerance - 1e-9)
        candidates = []
        if timeline.is_consistent or (
                ends_early and map_cmp[0]
                and share_bad(map_cmp) <= STAGE_EVENT_DISAGREEMENT_WARN):
            candidates.append('map')
        if fits_signal:
            candidates.append('stored')
        if not candidates:
            return None

        have_evidence = bool(map_cmp[0] or stored_cmp[0])
        if have_evidence:
            chosen = min(candidates,
                         key=lambda c: (share_bad(map_cmp if c == 'map' else stored_cmp),
                                        0 if c == 'map' else 1))
        else:
            chosen = candidates[0]      # 'map' when present

        n_stages = len(stages)
        removed = timeline.removed_seconds
        stored_txt = (f"{stored_cmp[0] - stored_cmp[1]} of {stored_cmp[0]} as stored"
                      if fits_signal else "as stored not possible")
        scores = (f"stage events agree on {map_cmp[0] - map_cmp[1]} of "
                  f"{map_cmp[0]} through the time map and {stored_txt}"
                  if have_evidence else "no stage events to check")
        chosen_cmp = map_cmp if chosen == 'map' else stored_cmp

        if chosen == 'map':
            epochs = timeline.exact_epochs()
            timeline.meta.update({'stage_events_compared': map_cmp[0],
                                  'stage_events_disagreeing': map_cmp[1]})
            logger.info(
                f"Staging source: {STAGING_SOURCE_TIME_MAP} "
                f"({n_stages} full-night epochs, {timeline.n_boundaries} "
                f"boundary events removing {removed:.1f} s, signal {T:.1f} s "
                f"-> {len(epochs)} exact epochs over {timeline.last_second} s; "
                f"{scores})")
            if not have_evidence and fits_signal and removed >= L / 2:
                logger.warning(
                    f"etc.stages fits both the full night and the cut signal "
                    f"({removed:.1f} s removed) and there are no stage events "
                    f"to tell them apart; read as the full night through the "
                    f"time map. Check the staging if etc.stages was rescored "
                    f"after the cut.")
            result = StagingChoice(STAGING_SOURCE_TIME_MAP, None, epochs,
                                   timeline)
        else:
            msg = (f"Staging source: {STAGING_SOURCE_HEADER} ({n_stages} epochs "
                   f"taken to be on the cut time base although "
                   f"{timeline.n_boundaries} boundary events remove "
                   f"{removed:.1f} s; signal {T:.1f} s; {scores})")
            if have_evidence:
                logger.info(msg)
            else:
                logger.warning(msg)
            result = StagingChoice(STAGING_SOURCE_HEADER, stages, None, None)

        if chosen_cmp[0] and share_bad(chosen_cmp) > STAGE_EVENT_DISAGREEMENT_REJECT:
            logger.warning(
                f"{chosen_cmp[1]} of {chosen_cmp[0]} stage events "
                f"({100.0 * share_bad(chosen_cmp):.1f} %) disagree even with the "
                f"better reading of etc.stages ({result[0]}); the boundary "
                f"events probably misstate the removed time. Falling back to "
                f"the stage events.")
            return None
        if chosen_cmp[0] and share_bad(chosen_cmp) > STAGE_EVENT_DISAGREEMENT_WARN:
            logger.warning(
                f"{chosen_cmp[1]} of {chosen_cmp[0]} stage events "
                f"({100.0 * share_bad(chosen_cmp):.1f} %) disagree with the "
                f"chosen staging ({result[0]}); check the boundary events "
                f"before trusting it")
        return result

    def _warn_if_events_disagree(self, stages, stage_events, s_freq,
                                 n_samples, L):
        """WARN when stage events contradict ``etc.stages`` read as stored.

        Used on files with no removed time, where ``etc.stages`` is imported
        unchanged whatever this finds; the check only reports (for example an
        unmarked gap in the signal that shifted the stage events).

        Parameters
        ----------
        stages : sequence
            ``etc.stages`` as stored.
        stage_events : list of (float, str)
            Stage events from the header; nothing is checked when empty.
        s_freq : float
            Sampling frequency.
        n_samples : int
            Samples in the signal.
        L : float
            Epoch length in seconds.

        Returns
        -------
        (int, int)
            ``(n_compared, n_disagree)``.
        """
        if not stage_events:
            return 0, 0
        as_stored = RecordingTimeline([], [], s_freq, n_samples,
                                      stages=[str(x).strip() for x in stages],
                                      epoch_length=L)
        n_cmp, n_bad = as_stored.compare_stage_events(stage_events)
        if n_cmp and n_bad / n_cmp > STAGE_EVENT_DISAGREEMENT_WARN:
            logger.warning(
                f"{n_bad} of {n_cmp} stage events ({100.0 * n_bad / n_cmp:.1f} %) "
                f"disagree with etc.stages as stored, and no boundary events "
                f"record removed data; the signal may have an unmarked gap. "
                f"etc.stages was imported unchanged.")
        return n_cmp, n_bad

    def _import_stage_codes(self, stages, epoch_length=30):
        """Write Compumedics stage codes to a temp file and import them.

        Parameters
        ----------
        stages : sequence
            One stage code per epoch, starting at the recording start.
        epoch_length : int
            Epoch length in seconds.

        Returns
        -------
        bool
            True when Wonambi's ``import_staging`` succeeded.
        """
        # Get recording start time
        if 'start_time' in self.dataset.header:
            rec_start = self.dataset.header['start_time']
        else:
            # Default to current date/time if not available.
            # `datetime` here is the MODULE (see the import at the top), so
            # this needs the class as well; `datetime.now()` raised
            # AttributeError, which the broad except in the caller turned into
            # a silent "return False" -- i.e. a recording that HAS staging
            # but no header start_time lost its stages without a word.
            rec_start = datetime.datetime.now()

        # Create a temporary file with Compumedics format staging
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_filename = temp_file.name

            # Write stages directly in Compumedics format (one stage code per line)
            for stage_code in stages:
                # Convert to string and write to file
                temp_file.write(f"{stage_code}\n")

        try:
            # Import the staging using Wonambi's import_staging method
            self.annotations.import_staging(
                filename=temp_filename,
                source='compumedics',  # Use compumedics format
                rater_name=self.rater_name,
                rec_start=rec_start,
                staging_start=None,  # Use default (no offset)
                epoch_length=epoch_length,
                poor=['Artefact'],  # Default poor quality markers
                as_qual=False  # Don't import as quality markers
            )

            print(f"Successfully imported {len(stages)} stages from header as rater '{self.rater_name}'")
            return True

        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_filename}: {e}")

    def _write_exact_epochs(self, epochs, poor=('Artefact',)):
        """Replace this rater's epochs with exact variable-length epochs.

        Wonambi's ``import_staging`` can only write a fixed grid, so the
        ``<epoch>`` elements are written directly, in the same layout
        ``import_staging`` produces (integer ``epoch_start`` / ``epoch_end``,
        ``stage``, ``quality``).

        Parameters
        ----------
        epochs : sequence of tuple
            ``(start, end, code, orig_epoch)`` from
            :func:`~turtlewave_hdEEG.timeline.exact_cut_epochs`; ``start`` and
            ``end`` are whole seconds, ``code`` a Compumedics stage code
            (named through ``COMPUMEDICS_STAGE_KEY`` exactly as the grid import
            names it; unrecognised codes become ``Unknown``).
        poor : sequence of str
            Stage names whose epochs get quality ``Poor``; all others ``Good``.

        Returns
        -------
        bool
            True once the file is saved.
        """
        if self.rater_name not in self.annotations.raters:
            self.annotations.add_rater(self.rater_name)
        self.annotations.get_rater(self.rater_name)
        stages = self.annotations.rater.find('stages')
        for old in list(stages):
            stages.remove(old)
        for start, end, code, _orig in epochs:
            epoch = SubElement(stages, 'epoch')
            SubElement(epoch, 'epoch_start').text = str(int(start))
            SubElement(epoch, 'epoch_end').text = str(int(end))
            name = stage_name_for_code(code)
            SubElement(epoch, 'stage').text = name
            SubElement(epoch, 'quality').text = 'Poor' if name in poor else 'Good'
        self.annotations.save()
        durations = [int(e[1]) - int(e[0]) for e in epochs]
        logger.info(
            f"Wrote {len(epochs)} exact epochs as rater '{self.rater_name}' "
            f"({sum(d == 30 for d in durations)} of 30 s, "
            f"{sum(d == 1 for d in durations)} of 1 s, "
            f"{sum(durations)} s in total)")
        return True

    def _write_timeline_sidecar(self, timeline, source, epochs):
        """Write ``<annotation xml stem>_timeline.json`` (schema 2).

        Parameters
        ----------
        timeline : RecordingTimeline
            The time map (for the stage-event source, a map without
            full-night stages).
        source : str
            Staging-source name.
        epochs : list of tuple
            The exact epochs written to the XML, stored as ``cut_epochs``.

        Returns
        -------
        Path or None
            The sidecar path, or ``None`` if writing failed (logged).
        """
        try:
            from . import __version__ as tw_version
        except ImportError:
            tw_version = None
        header = getattr(self.dataset, 'header', {}) or {}
        start = header.get('start_time')
        timeline.cut_epochs = [(int(e[0]), int(e[1]), str(e[2]), int(e[3]))
                               for e in epochs]
        if epochs:
            timeline.last_second = int(epochs[-1][1])
        timeline.meta.update({
            'source': source,
            'turtlewave_version': tw_version,
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
            'recording_start': start.isoformat() if hasattr(start, 'isoformat') else None,
            'data_file': str(getattr(self.dataset, 'filename', '') or ''),
            'annotation_file': str(Path(self.annot_file).name),
            'rater': self.rater_name,
        })
        path = sidecar_path(self.annot_file)
        try:
            timeline.to_json(path)
        except Exception as e:
            logger.error(f"Could not write timeline sidecar {path}: {e}")
            return None
        logger.info(f"Timeline sidecar written: {path}")
        return path

    def add_annotations_batch(self, label, start_times, end_times, channels=None):
        """Add multiple annotations of one label at once.

        The event type is created if the annotation file does not already carry
        it. Nothing is written to disk here -- the caller is responsible for
        ``save()``.

        Parameters
        ----------
        label : str
            Event type name, e.g. ``'Artefact'``.
        start_times : sequence of float
            Start times in seconds.
        end_times : sequence of float
            End times in seconds, same length as ``start_times``. Pairs are
            zipped, so a shorter sequence silently truncates the batch.
        channels : sequence of str or None, optional
            Channel for each annotation. ``None`` (default) writes every
            annotation on ``'(all)'``.

        Returns
        -------
        bool
            True when the whole batch was added, False if any ``add_event``
            raised -- in which case annotations added before the failure remain
            in the in-memory tree, so the count is all-or-nothing only from the
            caller's point of view.
        """
        try:
            if label not in self.annotations.event_types:
                self.annotations.add_event_type(label)
                
            if channels is None:
                channels = ['(all)'] * len(start_times)
                
            # Add events in batch
            for start, end, chan in zip(start_times, end_times, channels):
                self.annotations.add_event(
                    name=label,
                    time=(float(start), float(end)),
                    chan=chan
                )
            return True
        except Exception as e:
            print(f"Error adding batch annotations: {e}")
            return False    


    def add_annotation(self, label, start_time, end_time, channel=None):
        """
        Add a single annotation to the annotations object.
        
        Parameters
        ----------
        label : str
            Label for the annotation
        start_time : float
            Start time in seconds
        end_time : float
            End time in seconds
        channel : str, list, or None
            Channel(s) associated with the annotation. 
            If None, uses '(all)' to indicate all channels.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Format the time as a tuple of float values
            time_tuple = (float(start_time), float(end_time))

            
            if channel is None:
                channel = '(all)'  # Wonambi standard for all channels

            # Make sure the event type exists
            if label not in self.annotations.event_types:
                self.annotations.add_event_type(label)

            # Add the event with proper rater specification
            self.annotations.add_event(
                name=label, 
                time=time_tuple,
                chan=channel
            )
            return True

        except Exception as e:
            print(f"Error adding annotation: {e}")
            return False


    def process_all(self):
        """Import artefact/arousal events and header staging in one pass.

        Both steps always run. Only the staging step has a pass/fail answer:
        :meth:`add_artefacts_from_events` returns ``(count, seconds)`` and a
        count of zero is a legitimate outcome (the recording simply carries no
        reject flags), so it cannot be folded into a success flag without
        reporting a clean recording as a failure.

        Returns
        -------
        bool
            The result of :meth:`add_stages_from_header`: ``True`` when header
            staging was imported, ``False`` when the recording header carries
            no staging (or the import failed) -- in which case the annotation
            XML is written WITHOUT sleep stages and every stage-filtered
            detection downstream will find nothing. This used to be hardcoded
            ``True``, so a non-GUI caller had no way to see that outcome;
            ``frontend/turtlewave_gui.py`` works around it by calling the two
            steps directly.
        """
        # Artefacts/arousals first: it reports its own counts and timing.
        self.add_artefacts_from_events()

        return self.add_stages_from_header()

    def save(self, filename=None):
        """Save the annotations as Wonambi XML.

        Uses ``Annotations.save()``, which serialises the whole annotation
        tree (raters, epochs, events) back to XML.

        **This used to call** ``Annotations.export(filename)``. Wonambi
        7.15's ``export`` defaults to ``xformat='csv'`` and writes a four-column
        epoch/stage CSV to the path it is given, so calling ``save()`` with the
        default ``annot_file`` OVERWROTE the annotation XML with a stage CSV --
        losing every event and rater, and leaving a file that
        ``Annotations(annot_file)`` can no longer parse. Nothing in this
        repository called it, which is why the trap survived.

        Parameters
        ----------
        filename : str or None
            Path to write to. ``None`` (default) uses the ``annot_file`` given
            at construction. A different path is written as a copy: the object
            keeps pointing at its original file afterwards, so a "save as" does
            not silently redirect every later write.

        Returns
        -------
        bool
            True on success, False if the write raised.
        """
        target = self.annot_file if filename is None else filename

        # Annotations.save() has no target argument -- it writes to
        # self.xml_file -- so a copy is made by retargeting it for the one
        # call and restoring it afterwards, even on failure.
        original = getattr(self.annotations, 'xml_file', None)
        try:
            if original is not None and str(target) != str(original):
                try:
                    self.annotations.xml_file = target
                    self.annotations.save()
                finally:
                    self.annotations.xml_file = original
            else:
                self.annotations.save()
            print(f"Annotations saved to {target}")
            return True
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False


class CustomAnnotations:
    """Helper class for reading and working with Wonambi annotations"""
    
    def __init__(self, annot_file):
        self.annot_file = annot_file
        self.wonb_annot = WonambiAnnotations(annot_file)
        
        # Try to explicitly select a rater if none is selected
        if self.wonb_annot.rater is None and len(self.wonb_annot.raters) > 0:
            self.wonb_annot.get_rater(self.wonb_annot.raters[0])
    @property
    def last_second(self):
        """Return the last second in the recording"""
        return self.wonb_annot.last_second
    
    @property
    def first_second(self):
        """Return the first second in the recording"""
        return self.wonb_annot.first_second
    
    @property
    def dataset(self):
        """Return the dataset associated with the annotations"""
        return self.wonb_annot.dataset
    
    @property
    def rater(self):
        """Return the current rater"""
        return self.wonb_annot.rater
    
    @property
    def raters(self):
        """Return all raters in the annotation file"""
        return self.wonb_annot.raters

    @property
    def epochs(self):
        """Get all epochs from the annotation file"""
        try:
            return list(self.wonb_annot.epochs)
        except IndexError:
            # If no rater is found, find all raters and use the first one
            if len(self.wonb_annot.raters) > 0:
                self.wonb_annot.get_rater(self.wonb_annot.raters[0])
                return list(self.wonb_annot.epochs)
            return []
        
    def get_epochs(self, *args, **kwargs):
        """
        Get epochs that match the specified criteria.
        This method matches the Wonambi API for compatibility.
        
        Returns
        -------
        list of dict
            list of epochs, which are dict with 'start' and 'end' times, plus
            additional parameters
        """
        # Delegate to the underlying Wonambi annotations object
        return self.wonb_annot.get_epochs(*args, **kwargs)

    def get_rater(self, rater):
        """
        Select one rater.
        
        Parameters
        ----------
        rater : str
            name of the rater
        """
        return self.wonb_annot.get_rater(rater)
    def add_rater(self, rater):
        """
        Add one rater.
        
        Parameters
        ----------
        rater : str
            name of the rater
        """
        return self.wonb_annot.add_rater(rater)
    
    def get_stages(self):
        """Stage name of every epoch, in file order.

        Returns
        -------
        list of str
            One name per epoch. Epochs may differ in length (exact epochs on
            a cut recording), so never turn a count of these into time by
            multiplying by 30; use :meth:`get_stage_intervals` or
            :meth:`epoch_durations`.
        """
        epochs = self.epochs
        if epochs:
            return [epoch['stage'] for epoch in epochs]
        return []

    def get_hypnogram(self):
        """Numeric stage code of every epoch, in file order.

        Returns
        -------
        list of int
            Wake 0, NREM1/2/3 1/2/3, REM 4, anything else -1. Codes only:
            epochs may differ in length, so never multiply a count by 30; use
            :meth:`get_hypnogram_intervals` or :meth:`epoch_durations`.
        """
        stages = self.get_stages()
        return [HYPNOGRAM_CODES.get(stage, -1) for stage in stages]

    def get_stage_intervals(self):
        """Every epoch as ``(start, end, stage)``, the canonical hypnogram.

        Returns
        -------
        list of (float, float, str)
            Seconds from the recording start and the Wonambi stage name,
            sorted by start. Epoch lengths are whatever the file holds: 30 s
            on a grid import, variable on a cut recording's exact epochs.
        """
        return sorted(((float(e['start']), float(e['end']), str(e['stage']))
                       for e in self.epochs), key=lambda x: x[0])

    def get_hypnogram_intervals(self):
        """Every epoch as ``(start, end, code)`` with numeric stage codes.

        Returns
        -------
        list of (float, float, int)
            As :meth:`get_stage_intervals`, with the codes of
            :meth:`get_hypnogram`.
        """
        return [(s, e, HYPNOGRAM_CODES.get(st, -1))
                for s, e, st in self.get_stage_intervals()]

    def epoch_durations(self):
        """Length of every epoch in seconds, in the order of :meth:`get_stage_intervals`.

        Returns
        -------
        list of float
        """
        return [e - s for s, e, _ in self.get_stage_intervals()]

    def has_uniform_epochs(self, tol=1e-6):
        """Whether every epoch has the same length.

        Parameters
        ----------
        tol : float
            Allowed difference in seconds.

        Returns
        -------
        bool
            True for a grid import (and for a file with no epochs); False for
            a cut recording's exact epochs.
        """
        d = self.epoch_durations()
        return not d or (max(d) - min(d)) <= tol

    def recording_seconds(self):
        """Length of the recording in seconds.

        Returns
        -------
        float
            The annotation file's ``last_second`` (``int(n_samples /
            s_freq)`` when Wonambi created it), else the end of the last
            epoch, else 0.0.
        """
        try:
            last = self.wonb_annot.last_second
            if last:
                return float(last)
        except (AttributeError, TypeError, ValueError):
            pass
        ivals = self.get_stage_intervals()
        return float(ivals[-1][1]) if ivals else 0.0

    def save(self, filename=None):
        """Save the annotations as Wonambi XML.

        ``Annotations.save()`` takes no target and always writes to its own
        ``xml_file``, so this used to accept a ``filename``, ignore it, and
        then print "Annotations saved to <filename>" for a file it had not
        touched. The path is now honoured, and written as a copy: the object
        keeps pointing at its original file afterwards.

        Parameters
        ----------
        filename : str or None
            Path to write to. ``None`` (default) uses the ``annot_file`` given
            at construction.

        Returns
        -------
        bool
            True on success, False if the write raised.
        """
        target = self.annot_file if filename is None else filename

        original = getattr(self.wonb_annot, 'xml_file', None)
        try:
            if original is not None and str(target) != str(original):
                try:
                    self.wonb_annot.xml_file = target
                    self.wonb_annot.save()
                finally:
                    self.wonb_annot.xml_file = original
            else:
                self.wonb_annot.save()
            print(f"Annotations saved to {target}")
            return True
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False

    # Special method for fetch compatibility
    def create_epochs(self, times, epoch_length=30):
        """
        Create epochs from a sequence of time points.
        
        Parameters
        ----------
        times : list or ndarray
            List of time points (in seconds)
        epoch_length : float, optional
            Length of each epoch in seconds
        """
        times = np.asarray(times)
        return self.wonb_annot.create_epochs(times, epoch_length)
    
    # Add method to get time points for a specific stage
    def get_times(self, stage=None, cycle=None, exclude=None):
        """
        Return the times (start and end) for all epochs that match the parameters.
        
        Parameters
        ----------
        stage : str or None
            Stage to match with
        cycle : str or None
            Cycle to match with
        exclude : str or None
            Stage to exclude
            
        Returns
        -------
        list of tuple
            Each tuple contains the start and end time of an epoch
        """
        return self.wonb_annot.get_times(stage=stage, cycle=cycle, exclude=exclude)
            
    # Add any other methods you need to access from the original WonambiAnnotations
    def __getattr__(self, name):
        """Delegate any other method calls to the original WonambiAnnotations object"""
        return getattr(self.wonb_annot, name)