#!/usr/bin/env python3
"""``XLAnnotations.add_artefacts_from_events`` must mask splices and keep time.

Three things are asserted here, all on a real Wonambi annotation file written
to disk and read back:

* **EEGLAB ``boundary`` events are masked.** A boundary marks a splice, where
  a segment of data was cut out and the remainder joined; the signal steps
  discontinuously there and can seed a false slow wave, and any event or
  coupling phase estimate spanning the step is meaningless. Boundaries used to
  match none of the type rules and fell through silently. They now write an
  ``Artefact`` over ``onset +/- 2 s``, clipped to the recording. The event's
  own ``duration`` is IGNORED: for a boundary, EEGLAB stores the length of the
  REMOVED data in the original time base, so ``onset + duration`` would reject
  good post-splice data (about 36 min on one measured subject).
* **Sub-second durations survived being truncated to zero.**
  ``duration_seconds = np.ones_like(onsets)`` inherited the onsets dtype, so an
  integer latency array gave an integer duration array and every duration under
  1 s was floored to 0 -- a 0.4 s arousal became a zero-length annotation.
* **Files holding only Resp/Move/Snore events were never written.** The save
  gate summed Artefact + Arousal only, so those annotations were added to the
  in-memory tree and dropped on exit.
* **Compumedics respiratory names.** ``centralapnea``, ``mixedapnea``,
  ``obstructive apnea`` (with a space) and the exact string ``rera`` are
  respiratory. ``arousal5rera`` is an Arousal only, because ``rera`` is matched
  exactly. ``spo2artifact``, ``slpcycle1`` and ``slpepsws`` match no rule.

Run standalone: ``python tests/test_annotation_events.py``. Any failure raises
and the process exits non-zero.
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turtlewave_hdEEG.annotation import XLAnnotations  # noqa: E402

S_FREQ = 128.0
DURATION = 120.0


def _dataset(tmp, events=None, duration=DURATION, s_freq=S_FREQ):
    """Write a small EDF, open it, and attach an EEGLAB-style event dict.

    Built the same way as ``tests/test_annotation_io.py``: a synthetic
    single-channel recording, no real data needed. ``sampling_rate`` is set
    explicitly because ``add_artefacts_from_events`` reads that attribute, and
    a bare ``wonambi.Dataset`` exposes the rate only as ``header['s_freq']``
    (``LargeDataset`` is what normally adds it).

    Parameters
    ----------
    tmp : str
        Directory to write the EDF into.
    events : dict or None, optional
        Value for ``header['event']``, with keys ``onsets`` (samples),
        ``types`` and ``durations`` (samples). ``None`` leaves the header
        without an ``event`` entry.
    duration : float, optional
        Recording length in seconds. Default ``120.0``.
    s_freq : float, optional
        Sampling frequency in Hz. Default ``128.0``.

    Returns
    -------
    instance of wonambi.Dataset
        Dataset with ``sampling_rate`` set and, if given, ``header['event']``.
    """
    from wonambi import Dataset
    from wonambi.ioeeg import write_edf
    from wonambi.utils.simulate import create_data

    data = create_data(datatype='ChanTime', n_trial=1, s_freq=s_freq,
                       chan_name=['Cz'], time=(0, duration))
    n = len(data.axis['time'][0])
    t = np.arange(n) / s_freq
    data.data[0] = np.asarray(20.0 * np.sin(2 * np.pi * 1.0 * t),
                              dtype='f')[None, :]

    edf = os.path.join(tmp, 'sub-B.edf')
    write_edf(data, edf)
    dataset = Dataset(edf)
    dataset.sampling_rate = dataset.header['s_freq']
    if events is not None:
        dataset.header['event'] = events
    return dataset


def _written_events(annot_file, name):
    """Re-read the annotation XML and return ``(start, end)`` for one label.

    Parameters
    ----------
    annot_file : str
        Path to the annotation XML.
    name : str
        Event type name, e.g. ``'Artefact'``.

    Returns
    -------
    list of tuple of float
        Start/end times in seconds, sorted by start.
    """
    from wonambi.attr import Annotations

    events = Annotations(annot_file).get_events(name=name)
    return sorted((float(e['start']), float(e['end'])) for e in events)


def test_boundary_is_masked_not_duration_spanned():
    """A boundary writes a +/-2 s Artefact and ignores its own duration."""
    print("\n1. EEGLAB boundary -> +/-2 s Artefact, duration ignored:")

    tmp = tempfile.mkdtemp(prefix='tw_annot_ev_bnd_')
    try:
        # onset 100 s; duration 3600 samples = 28.125 s of REMOVED data.
        dataset = _dataset(tmp, events={
            'onsets': [100.0 * S_FREQ],
            'types': ['boundary'],
            'durations': [3600],
        })
        annot_file = os.path.join(tmp, 'boundary.xml')
        xl = XLAnnotations(dataset, annot_file, rater_name='tester')
        count, _ = xl.add_artefacts_from_events()

        assert count == 1, f"expected 1 annotation written, got {count}"
        written = _written_events(annot_file, 'Artefact')
        assert len(written) == 1, (
            f"a boundary must produce exactly one Artefact, got {written}")
        start, end = written[0]
        assert abs(start - 98.0) < 1e-6 and abs(end - 102.0) < 1e-6, (
            f"expected (98.0, 102.0), got ({start}, {end})")
        assert end - start < 28.0, (
            f"the window is {end - start:.3f} s long -- the boundary's own "
            f"duration (28.125 s of removed data) was used as the span, "
            f"which rejects good post-splice data")
        print(f"   [ok] boundary at 100.0 s (duration 3600 samples) -> "
              f"Artefact ({start:.1f}, {end:.1f})")

        # Nothing else should have been swept in by the other five rules.
        for label in ('Arousal', 'Resp', 'Move', 'Snore'):
            assert not _written_events(annot_file, label), (
                f"a 'boundary' event also matched the {label} rule")
        print("   [ok] the boundary matched no other rule")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_boundary_windows_are_clipped_to_the_recording():
    """Boundaries at latency 0, at the end, and past the end are handled."""
    print("\n2. Boundary windows clip to [0, recording length]:")

    tmp = tempfile.mkdtemp(prefix='tw_annot_ev_clip_')
    try:
        dataset = _dataset(tmp, events={
            'onsets': [0.5 * S_FREQ,          # clips at zero
                       DURATION * S_FREQ,      # exactly at the end
                       (DURATION + 5.0) * S_FREQ],  # past the end -> skipped
            'types': ['boundary', 'Boundary', 'boundary'],
            'durations': [0, 0, 0],
        })
        annot_file = os.path.join(tmp, 'clip.xml')
        xl = XLAnnotations(dataset, annot_file, rater_name='tester')
        count, _ = xl.add_artefacts_from_events()

        written = _written_events(annot_file, 'Artefact')
        assert count == 2, (
            f"expected 2 annotations (the out-of-range one skipped), got "
            f"{count}: {written}")
        assert len(written) == 2, written

        start, end = written[0]
        assert abs(start - 0.0) < 1e-6 and abs(end - 2.5) < 1e-6, (
            f"a boundary at 0.5 s must clip to (0.0, 2.5), got ({start}, {end})")
        print(f"   [ok] boundary at 0.5 s -> ({start:.1f}, {end:.1f})")

        start, end = written[1]
        assert abs(start - 118.0) < 1e-6 and abs(end - DURATION) < 1e-6, (
            f"a boundary at the recording end must clip to "
            f"(118.0, {DURATION}), got ({start}, {end})")
        print(f"   [ok] boundary at {DURATION:.1f} s (spelled 'Boundary') -> "
              f"({start:.1f}, {end:.1f})")
        print("   [ok] boundary 5 s past the end collapsed to zero length and "
              "was skipped, not written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_integer_latencies_keep_subsecond_durations():
    """Integer sample latencies must not floor a 0.4 s arousal to 0 s."""
    print("\n3. Sub-second durations survive integer latencies:")

    tmp = tempfile.mkdtemp(prefix='tw_annot_ev_dur_')
    try:
        # int64 onsets, as scipy gives for EEGLAB latencies stored as integers.
        # 0.4 s is 51.2 samples at 128 Hz, so the nearest whole sample (51)
        # is 0.3984375 s -- that quantisation is the file's, not the code's.
        arousal_samples = int(round(0.4 * S_FREQ))
        expected = arousal_samples / S_FREQ
        dataset = _dataset(tmp, events={
            'onsets': np.array([int(30.0 * S_FREQ), int(60.0 * S_FREQ)],
                               dtype=np.int64),
            'types': ['arousal', 'reject'],
            'durations': np.array([arousal_samples, int(2.0 * S_FREQ)],
                                  dtype=np.int64),
        })
        annot_file = os.path.join(tmp, 'dur.xml')
        xl = XLAnnotations(dataset, annot_file, rater_name='tester')
        count, _ = xl.add_artefacts_from_events()
        assert count == 2, f"expected 2 annotations, got {count}"

        arousals = _written_events(annot_file, 'Arousal')
        assert len(arousals) == 1, arousals
        start, end = arousals[0]
        length = end - start
        assert abs(start - 30.0) < 1e-6, f"arousal starts at {start}, not 30.0"
        assert abs(length - expected) < 1e-6, (
            f"the arousal is {length} s long, expected {expected} s (~0.4 s) "
            f"-- an integer duration array floored it")
        assert length > 0, (
            "the arousal has zero length: duration_seconds inherited the "
            "integer onsets dtype and truncated 0.4 -> 0")
        print(f"   [ok] ~0.4 s arousal written as ({start:.1f}, {end:.3f}), "
              f"length {length:.4f} s (expected {expected:.4f} s)")

        artefacts = _written_events(annot_file, 'Artefact')
        assert len(artefacts) == 1, artefacts
        assert abs((artefacts[0][1] - artefacts[0][0]) - 2.0) < 1e-6, artefacts
        print(f"   [ok] 2.0 s reject written as "
              f"({artefacts[0][0]:.1f}, {artefacts[0][1]:.1f})")

        # A short/absent duration list must not raise, and falls back to 1.0 s.
        dataset2 = _dataset(tmp, events={
            'onsets': np.array([int(10.0 * S_FREQ), int(20.0 * S_FREQ)],
                               dtype=np.int64),
            'types': ['arousal', 'arousal'],
            'durations': np.array([int(0.5 * S_FREQ)], dtype=np.int64),
        })
        annot_file2 = os.path.join(tmp, 'shortdur.xml')
        xl2 = XLAnnotations(dataset2, annot_file2, rater_name='tester')
        count2, _ = xl2.add_artefacts_from_events()
        assert count2 == 2, f"expected 2 annotations, got {count2}"
        lengths = [round(e - s, 6) for s, e in _written_events(annot_file2,
                                                               'Arousal')]
        assert lengths == [0.5, 1.0], (
            f"a durations list shorter than onsets must fall back to 1.0 s for "
            f"the missing entry, got lengths {lengths}")
        print(f"   [ok] durations shorter than onsets: lengths {lengths} "
              f"(1.0 s fallback, no IndexError)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resp_only_file_is_saved():
    """A file with no Artefact/Arousal still gets written to disk."""
    print("\n4. Resp-only event list reaches the XML:")

    tmp = tempfile.mkdtemp(prefix='tw_annot_ev_resp_')
    try:
        dataset = _dataset(tmp, events={
            'onsets': [40.0 * S_FREQ],
            'types': ['Hypopnea'],
            'durations': [int(12.0 * S_FREQ)],
        })
        annot_file = os.path.join(tmp, 'resp.xml')
        xl = XLAnnotations(dataset, annot_file, rater_name='tester')
        count, _ = xl.add_artefacts_from_events()

        assert count == 1, (
            f"the return counted {count}: it used to sum Artefact + Arousal "
            f"only, so a Resp-only file reported 0 and was never saved")

        # The real contract: it is on disk, not just in the in-memory tree.
        resp = _written_events(annot_file, 'Resp')
        assert len(resp) == 1, (
            f"the Resp annotation never reached {os.path.basename(annot_file)}"
            f": {resp}")
        start, end = resp[0]
        assert abs(start - 40.0) < 1e-6 and abs(end - 52.0) < 1e-6, (start, end)
        print(f"   [ok] Resp ({start:.1f}, {end:.1f}) written and reloaded "
              f"from a file holding no Artefact or Arousal")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_compumedics_respiratory_names():
    """Compumedics scorer event names: which are Resp, which are not."""
    print("\n5. Compumedics respiratory names:")

    resp_names = ['centralapnea', 'mixedapnea', 'obstructive apnea', 'obstructiveapnea',
                  'rera', 'RERA', ' rera ', 'spo2desat', 'Hypopnea',
                  # separators are ignored for the respiratory rule
                  'spo2 desat', 'SpO2_Desat', 'central apnea', 'mixed_apnea']
    arousal_only = ['arousal5rera', 'arousal 4 rera', 'arousal_5_rera']
    ignored = ['spo2artifact', 'spo2 artifact', 'slpcycle1', 'slpcycle12', 'slpepsws',
               'slpepnrem', 'rerax', 'prera', 'rera2',
               # left unmapped pending a research decision
               'unsure respiratory event']
    names = resp_names + arousal_only + ignored

    tmp = tempfile.mkdtemp(prefix='tw_annot_ev_compu_')
    try:
        onset = {name: 5.0 + 10.0 * i for i, name in enumerate(names)}
        length = 10.0 * len(names) + 20.0
        dataset = _dataset(tmp, duration=length, events={
            'onsets': [onset[n] * S_FREQ for n in names],
            'types': names,
            'durations': [int(4.0 * S_FREQ)] * len(names),
        })
        annot_file = os.path.join(tmp, 'compu.xml')
        xl = XLAnnotations(dataset, annot_file, rater_name='tester')
        count, _ = xl.add_artefacts_from_events()

        resp = [s for s, _ in _written_events(annot_file, 'Resp')]
        want = sorted(onset[n] for n in resp_names)
        assert resp == want, (
            f"Resp events at {resp}, expected {want}; missing "
            f"{[n for n in resp_names if onset[n] not in resp]}, unexpected "
            f"{[n for n in names if onset[n] in resp and n not in resp_names]}")
        print(f"   [ok] Resp: {resp_names}")

        arousal = [s for s, _ in _written_events(annot_file, 'Arousal')]
        assert arousal == sorted(onset[n] for n in arousal_only), (
            f"{arousal_only} must be Arousals, got {arousal}")
        for n in arousal_only:
            assert onset[n] not in resp, (
                f"{n} was also tagged Resp: 'rera' must be an exact match")
        print(f"   [ok] {arousal_only} -> Arousal only, not Resp")

        for label in ('Artefact', 'Move', 'Snore'):
            written = _written_events(annot_file, label)
            assert not written, f"{label} received {written}"
        seen = set(resp) | set(arousal)
        for name in ignored:
            assert onset[name] not in seen, f"{name!r} was written to the annotations"
        assert count == len(resp_names) + len(arousal_only), (
            f"returned count {count}, expected {len(resp_names) + len(arousal_only)} "
            f"(the ignored names must not be counted)")
        print(f"   [ok] {ignored} produce nothing; returned count {count}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TESTING XLAnnotations.add_artefacts_from_events")
    print("==============================================")

    test_boundary_is_masked_not_duration_spanned()
    test_boundary_windows_are_clipped_to_the_recording()
    test_integer_latencies_keep_subsecond_durations()
    test_resp_only_file_is_saved()
    test_compumedics_respiratory_names()

    print("\nAll annotation event tests passed.")
