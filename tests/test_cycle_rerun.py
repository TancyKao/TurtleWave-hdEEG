## test_cycle_rerun.py
#
# Regression tests for the 4.3.1 cycle re-run fixes in cycleprocessor.py:
#
#   1. A changed-threshold re-run (finalize_cycles_and_durations / ParalCycles
#      .run called twice with different wake_thresh) must not leave any
#      event.cycle carrying a tag from the first run's now-superseded spans.
#   2. A re-run that finds zero cycles (e.g. nrem_min raised past every NREM
#      run) must REPLACE the previous run's output everywhere -- sleep_cycles
#      rows, events.cycle, and the XML cycle markers -- not just clear
#      events.cycle while leaving the other two stale.
#   3. An unscorable hypnogram (empty, or every epoch unscored) must raise
#      ValueError and leave the database untouched, including any pre-existing
#      events.cycle tag.
#   4. A scored night with no cycles (all Wake) is a normal, accepted result:
#      no error, cycles=[], stage_durations written, stale tags cleared.
#
# No pytest; plain functions with prints + asserts, matching
# tests/test_cycle_connection_reuse.py. Builds real Wonambi annotation XML
# (as tests/test_sw_amplitude_floor.py and tests/test_turtlewave.py do)
# rather than a hand-rolled stub, because cases 2-4 assert on the XML itself
# (Annotations.get_cycles()) or on ParalCycles.run's ValueError path, neither
# of which a stub annotations object without a real epoch grid can exercise
# faithfully.
#
# Run standalone (the environment's site-packages copy of turtlewave_hdEEG
# may be stale, so put the repo first on PYTHONPATH):
#
#     PYTHONPATH=$PWD python tests/test_cycle_rerun.py

import os
import shutil
import sqlite3
import tempfile
from collections import Counter

from turtlewave_hdEEG import cycleprocessor as cp
from turtlewave_hdEEG.annotation import CustomAnnotations

EPOCH_LENGTH = 30
_STAGE_NAME = {0: 'Wake', 1: 'NREM1', 2: 'NREM2', 3: 'NREM3', 4: 'REM'}


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------

def _tmp_dir(tag):
    return tempfile.mkdtemp(prefix=f'twcyclererun_{tag}_')


def _build_annotations(tmp, n_epochs, epoch_length=EPOCH_LENGTH, stem='sub'):
    """A real Wonambi annotation XML with ``n_epochs`` epochs, all 'Unknown'.

    ``n_epochs=0`` produces a genuinely empty epoch grid (an unscored or
    un-epoched file): ``create_epochs()`` always lays down at least one epoch
    for a non-zero-duration dataset, so the empty grid is forced afterwards by
    stripping the ``<epoch>`` elements it created.

    Returns
    -------
    tuple
        ``(CustomAnnotations, xml_path)``.
    """
    from wonambi import Dataset
    from wonambi.attr.annotations import create_empty_annotations
    from wonambi.ioeeg import write_edf
    from wonambi.utils.simulate import create_data

    n_for_edf = max(n_epochs, 1)
    duration = float(epoch_length * n_for_edf)
    data = create_data(datatype='ChanTime', n_trial=1, s_freq=1.0,
                       chan_name=['Cz'], time=(0, duration))
    edf = os.path.join(tmp, f'{stem}.edf')
    write_edf(data, edf)
    dataset = Dataset(edf)

    xml_path = os.path.join(tmp, f'{stem}.xml')
    create_empty_annotations(xml_path, dataset)
    ann = CustomAnnotations(xml_path)
    ann.wonb_annot.add_rater('tester', epoch_length=epoch_length)

    if n_epochs == 0:
        stages_el = ann.wonb_annot.rater.find('stages')
        for ep in list(stages_el):
            stages_el.remove(ep)
        ann.wonb_annot.save()

    return ann, xml_path


def _stage(ann, hypnogram, epoch_length=EPOCH_LENGTH):
    """Write explicit stage names for the numeric codes in ``_STAGE_NAME``.

    Codes with no entry (only -1 is expected) are left at
    ``create_epochs()``'s default 'Unknown', which ``get_hypnogram()`` also
    maps to -1 -- so a caller wanting an all -1 hypnogram simply never calls
    this.
    """
    for i, code in enumerate(hypnogram):
        name = _STAGE_NAME.get(code)
        if name is not None:
            ann.wonb_annot.set_stage_for_epoch(i * epoch_length, name,
                                               save=False)
    ann.wonb_annot.save()


def _threshold_sensitive_hypnogram():
    """Two NREM/REM cycles whose FIRST cycle boundary moves with wake_thresh.

    An 8-epoch wake stretch sits between a 15-epoch NREM stub and the first
    real 40-epoch NREM run. At ``wake_thresh=10`` the gap is absorbed into
    NREM (8 <= 10), so the stub and the main run merge into one NREM period
    starting at the stub -- cycle 1 starts at epoch 0. At ``wake_thresh=5``
    the gap is NOT absorbed (8 > 5); the stub is then a standalone NREM run
    of 15 epochs, too short to count on its own (<= nrem_min=30) and is
    dropped, so cycle 1 starts at the main run instead -- epoch 23.

    This is deliberately NOT just a re-split that keeps the same overall
    coverage: the union of both runs' cycle spans covers [0, 4140] seconds
    only at wake_thresh=10, and [690, 4140] at wake_thresh=5. Events with
    start_time in [0, 690) are inside cycle 1 after run 1 and inside NO
    cycle after run 2 -- exactly the shape that exposes a missing clear:
    such rows are only re-matched, never explicitly nulled, by any
    per-cycle UPDATE. A fixture whose spans merely renumber without ever
    shrinking their union would pass even on the unfixed code, because
    every previously tagged row would always be re-matched by some new
    span.

    Verified directly against ``detect_cycles`` before this fixture was
    written: wake_thresh=10 -> cycle spans [(1, 0, 2340), (2, 2340, 4140)];
    wake_thresh=5 -> [(1, 690, 2340), (2, 2340, 4140)].
    """
    return ([2] * 15       # NREM stub
            + [0] * 8      # wake gap: absorbed at thresh=10, not at thresh=5
            + [2] * 40     # main NREM run
            + [4] * 15     # REM 1
            + [2] * 40     # NREM period 2
            + [4] * 15     # REM 2
            + [0] * 5)     # trailing wake


def _make_events_db(db_path, n_events, spacing=60.0):
    """One events table with a row every ``spacing`` seconds, cycle NULL."""
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE events (uuid TEXT PRIMARY KEY, '
                 'start_time REAL, cycle TEXT)')
    conn.executemany('INSERT INTO events VALUES (?, ?, NULL)',
                     [(f'e{i}', float(i * spacing)) for i in range(n_events)])
    conn.commit()
    conn.close()


def _fetch(db_path, sql, params=()):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _cycle_spans(cycles):
    """[(cycle_number, lo, hi, hi_inclusive)] exactly as tag_events_with_cycles
    applies them: every span but the last is a half-open [lo, hi); the last
    is closed [lo, hi]."""
    spans = []
    for idx, cyc in enumerate(cycles):
        lo = cyc['nrem_start_sec']
        if idx + 1 < len(cycles):
            spans.append((cyc['cycle_number'], lo,
                         cycles[idx + 1]['nrem_start_sec'], False))
        else:
            spans.append((cyc['cycle_number'], lo, cyc['rem_end_sec'], True))
    return spans


def _expected_cycle_for(start_time, spans):
    """The cycle_number (as str) a given start_time falls in, or None."""
    for num, lo, hi, inclusive in spans:
        if inclusive:
            if lo <= start_time <= hi:
                return str(num)
        elif lo <= start_time < hi:
            return str(num)
    return None


# --------------------------------------------------------------------------
# 1. Changed-threshold re-run leaves no stale tags
# --------------------------------------------------------------------------

def test_changed_threshold_rerun_no_stale_tags():
    """Re-running at a different wake_thresh must re-tag, not merge.

    Reviewer's reproduction on master: after the second run, events.cycle
    held a mix of run-1 and run-2 numbering ({'1': 37, '2': 30, None: 3});
    the fix makes every tag agree with run 2's spans ({'1': 28, '2': 30,
    None: 12} on the reviewer's real data). Exact counts depend on the
    fixture, so this asserts span containment directly instead of counts.
    """
    print("\n1. Changed-threshold re-run leaves no stale events.cycle tags:")
    tmp = _tmp_dir('thresh')
    try:
        hyp = _threshold_sensitive_hypnogram()
        ann, _ = _build_annotations(tmp, len(hyp))
        _stage(ann, hyp)

        db = os.path.join(tmp, 'neural_events.db')
        n_events = int(len(hyp) * EPOCH_LENGTH // 60) + 2
        _make_events_db(db, n_events)

        pc = cp.ParalCycles(annotations=ann, subject='sub-thresh')

        cycles1 = pc.run(db, method='2022', write_xml=False,
                         wake_thresh=10, nrem_min=30)
        print(f"   run 1 (wake_thresh=10): {len(cycles1)} cycle(s), "
              f"cycle 1 starts at {cycles1[0]['nrem_start_sec']}s")
        assert len(cycles1) == 2, (
            f"fixture should yield exactly 2 cycles at wake_thresh=10 "
            f"(the wake gap absorbed into the NREM stub), got "
            f"{len(cycles1)}")
        assert cycles1[0]['nrem_start_sec'] == 0.0, (
            "fixture's cycle 1 should start at epoch 0 when the gap is "
            f"absorbed, got {cycles1[0]['nrem_start_sec']}")

        cycles2 = pc.run(db, method='2022', write_xml=False,
                         wake_thresh=5, nrem_min=30)
        print(f"   run 2 (wake_thresh=5): {len(cycles2)} cycle(s), "
              f"cycle 1 starts at {cycles2[0]['nrem_start_sec']}s")
        assert len(cycles2) == 2, (
            f"fixture should still yield 2 cycles at wake_thresh=5, but "
            f"with cycle 1 starting later (the NREM stub drops out once "
            f"the gap stops being absorbed), got {len(cycles2)}")
        assert cycles2[0]['nrem_start_sec'] == 690.0, (
            "fixture's cycle 1 should start later (690s) once the NREM "
            f"stub is dropped, got {cycles2[0]['nrem_start_sec']} -- if "
            f"this still reads 0.0 the fixture no longer moves the cycle "
            f"1 boundary and the test can pass on unfixed code too")

        spans2 = _cycle_spans(cycles2)
        valid_numbers = {str(c['cycle_number']) for c in cycles2}
        rows = _fetch(db, 'SELECT uuid, start_time, cycle FROM events')

        distribution = Counter(cyc for _, _, cyc in rows)
        print(f"   run-2 tag distribution: {dict(distribution)}")

        bad = []
        for uuid, start_time, cyc_val in rows:
            expected = _expected_cycle_for(start_time, spans2)
            if cyc_val != expected:
                bad.append((uuid, start_time, cyc_val, expected))
            elif cyc_val is not None and cyc_val not in valid_numbers:
                bad.append((uuid, start_time, cyc_val, expected))

        assert not bad, (
            f"{len(bad)} event(s) do not match run 2's span containment "
            f"(uuid, start_time, got, expected): {bad[:5]}")
        print("[ok] every events.cycle value after run 2 is NULL or a run-2 "
              "cycle number whose span contains the event -- no run-1-only "
              "tag survives")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# 2. Zero-cycle re-run replaces everything
# --------------------------------------------------------------------------

def test_zero_cycle_rerun_replaces_everything():
    """A re-run finding no cycles must clear sleep_cycles, events.cycle AND
    the XML markers, not just events.cycle.

    EXPECTED TO FAIL at this branch's current HEAD: ParalCycles.run() only
    calls store_cycles_to_database()/write_cycle_markers() `if cycles:`, so a
    zero-cycle re-run leaves the previous run's sleep_cycles rows and XML
    cycle markers in place while events.cycle alone gets cleared. Written to
    the intended (fixed) behaviour; see the report for the current-HEAD
    result.
    """
    print("\n2. Zero-cycle re-run replaces sleep_cycles, events.cycle, and "
          "XML markers:")
    tmp = _tmp_dir('zero')
    try:
        hyp = _threshold_sensitive_hypnogram()
        ann, xml_path = _build_annotations(tmp, len(hyp))
        _stage(ann, hyp)

        db = os.path.join(tmp, 'neural_events.db')
        n_events = int(len(hyp) * EPOCH_LENGTH // 60) + 2
        _make_events_db(db, n_events)

        subject = 'sub-zero'
        cycles_by_method = cp.finalize_cycles_and_durations(
            ann, db, subject=subject, methods=('2022',), tag_method='2022',
            write_xml=True, wake_thresh=10, nrem_min=30)
        n_cycles_run1 = len(cycles_by_method['2022'])
        print(f"   run 1 (nrem_min=30): {n_cycles_run1} cycle(s)")
        assert n_cycles_run1 > 0, "fixture must find cycles in run 1"

        sleep_cycles_run1 = _fetch(
            db, 'SELECT COUNT(*) FROM sleep_cycles WHERE subject=?',
            (subject,))[0][0]
        tagged_run1 = _fetch(
            db, 'SELECT COUNT(*) FROM events WHERE cycle IS NOT NULL')[0][0]
        markers_run1 = CustomAnnotations(xml_path).wonb_annot.get_cycles()
        print(f"   after run 1: sleep_cycles rows={sleep_cycles_run1}, "
              f"tagged events={tagged_run1}, XML markers={markers_run1}")
        assert sleep_cycles_run1 > 0 and tagged_run1 > 0 and markers_run1, (
            "run 1 should have written cycles, tags and XML markers")

        # Re-run on the SAME hypnogram with nrem_min raised past every NREM
        # run in the fixture (all are 40 epochs) -> zero cycles.
        ann2 = CustomAnnotations(xml_path)
        cycles_by_method2 = cp.finalize_cycles_and_durations(
            ann2, db, subject=subject, methods=('2022',), tag_method='2022',
            write_xml=True, wake_thresh=10, nrem_min=200)
        n_cycles_run2 = len(cycles_by_method2['2022'])
        print(f"   run 2 (nrem_min=200): {n_cycles_run2} cycle(s)")
        assert n_cycles_run2 == 0, (
            f"nrem_min=200 should exceed every 40-epoch NREM run in the "
            f"fixture, got {n_cycles_run2} cycle(s)")

        sleep_cycles_run2 = _fetch(
            db, 'SELECT COUNT(*) FROM sleep_cycles WHERE subject=?',
            (subject,))[0][0]
        tagged_run2 = _fetch(
            db, 'SELECT COUNT(*) FROM events WHERE cycle IS NOT NULL')[0][0]
        stage_rows_run2 = _fetch(
            db, 'SELECT COUNT(*) FROM stage_durations WHERE subject=?',
            (subject,))[0][0]
        markers_run2 = CustomAnnotations(xml_path).wonb_annot.get_cycles()
        print(f"   after run 2: sleep_cycles rows={sleep_cycles_run2}, "
              f"tagged events={tagged_run2}, stage_durations rows="
              f"{stage_rows_run2}, XML markers={markers_run2}")

        assert sleep_cycles_run2 == 0, (
            f"sleep_cycles should hold zero rows for '{subject}' (any "
            f"method) after a zero-cycle re-run, found "
            f"{sleep_cycles_run2} stale row(s) from run 1")
        assert tagged_run2 == 0, (
            f"every events.cycle should be NULL after a zero-cycle re-run, "
            f"found {tagged_run2} still tagged")
        assert not markers_run2, (
            f"the XML should hold no cycle markers after a zero-cycle "
            f"re-run, found {markers_run2}")
        assert stage_rows_run2 == 1, (
            "stage_durations should still hold exactly one row for the "
            "subject -- a zero-cycle night still has stage durations")
        print("[ok] a zero-cycle re-run empties sleep_cycles, events.cycle "
              "and the XML markers together, and still writes "
              "stage_durations")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# 3. Unscored hypnogram is refused
# --------------------------------------------------------------------------

def _assert_refused_and_untouched(ann, xml_path, tmp, label, entry_point):
    """Shared body: entry_point(ann, db) must raise ValueError and leave a
    pre-existing events.cycle tag untouched."""
    safe = ''.join(c if c.isalnum() else '_' for c in label)
    db = os.path.join(tmp, f'neural_events_{safe}.db')
    _make_events_db(db, 5)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE events SET cycle='stale' WHERE uuid='e0'")
    conn.commit()
    conn.close()

    raised = None
    try:
        entry_point(ann, db)
    except ValueError as e:
        raised = e
    print(f"   {label}: {'raised ValueError' if raised else 'DID NOT RAISE'}"
          f"{f' ({raised})' if raised else ''}")
    assert raised is not None, (
        f"{label} should raise ValueError on an unscorable hypnogram")

    surviving = _fetch(db, "SELECT cycle FROM events WHERE uuid='e0'")[0][0]
    assert surviving == 'stale', (
        f"{label}: the pre-existing events.cycle tag should survive "
        f"untouched, found {surviving!r}")


def test_unscored_hypnogram_is_refused():
    """Empty and all-unscored hypnograms raise ValueError via both
    ParalCycles.run and finalize_cycles_and_durations, clearing nothing."""
    print("\n3. Unscored hypnogram is refused (ValueError), tags untouched:")
    tmp = _tmp_dir('unscored')
    try:
        def via_run(ann, db):
            cp.ParalCycles(annotations=ann, subject='sub-x').run(db)

        def via_finalize(ann, db):
            cp.finalize_cycles_and_durations(ann, db, subject='sub-x')

        # (a) all -1: an epoch grid with no scoring saved.
        ann_allneg1, xml1 = _build_annotations(tmp, 20, stem='allneg1')
        assert all(s == -1 for s in ann_allneg1.get_hypnogram()), (
            "fixture is not actually all -1")
        _assert_refused_and_untouched(
            CustomAnnotations(xml1), xml1, tmp, 'all -1 / ParalCycles.run',
            via_run)
        _assert_refused_and_untouched(
            CustomAnnotations(xml1), xml1, tmp,
            'all -1 / finalize_cycles_and_durations', via_finalize)

        # (b) empty: no epochs at all.
        ann_empty, xml2 = _build_annotations(tmp, 0, stem='empty')
        assert ann_empty.get_hypnogram() == [], "fixture is not empty"
        _assert_refused_and_untouched(
            CustomAnnotations(xml2), xml2, tmp, 'empty / ParalCycles.run',
            via_run)
        _assert_refused_and_untouched(
            CustomAnnotations(xml2), xml2, tmp,
            'empty / finalize_cycles_and_durations', via_finalize)

        print("[ok] both entry points refuse both unscorable shapes and "
              "leave every existing events.cycle tag alone")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# 4. Scored all-Wake night is accepted
# --------------------------------------------------------------------------

def test_all_wake_night_is_accepted():
    """A genuinely scored night with zero cycles (all Wake) is not refused:
    no ValueError, zero cycles, stage_durations written, stale tags cleared.
    """
    print("\n4. Scored all-Wake night is accepted (no ValueError, tags "
          "cleared):")
    tmp = _tmp_dir('allwake')
    try:
        n_epochs = 20
        ann, _ = _build_annotations(tmp, n_epochs)
        _stage(ann, [0] * n_epochs)
        assert ann.get_hypnogram() == [0] * n_epochs, (
            "fixture is not a scored all-Wake hypnogram")

        db = os.path.join(tmp, 'neural_events.db')
        _make_events_db(db, 5)
        conn = sqlite3.connect(db)
        conn.execute("UPDATE events SET cycle='stale' WHERE uuid='e0'")
        conn.commit()
        conn.close()

        subject = 'sub-wake'
        cycles_by_method = cp.finalize_cycles_and_durations(
            ann, db, subject=subject, methods=('2022',), tag_method='2022')
        n_cycles = len(cycles_by_method['2022'])
        stage_rows = _fetch(
            db, 'SELECT total_min, wake_min FROM stage_durations '
            'WHERE subject=?', (subject,))
        tagged = _fetch(
            db, 'SELECT COUNT(*) FROM events WHERE cycle IS NOT NULL')[0][0]

        print(f"   cycles={n_cycles}, stage_durations rows={len(stage_rows)}"
              f" {stage_rows}, tagged events={tagged}")

        assert n_cycles == 0, f"an all-Wake night should yield 0 cycles, got {n_cycles}"
        assert len(stage_rows) == 1, "stage_durations should hold one row"
        total_min, wake_min = stage_rows[0]
        assert total_min == wake_min, (
            f"an all-Wake night's stage_durations should be 100% wake: "
            f"total={total_min}, wake={wake_min}")
        assert tagged == 0, (
            f"the pre-existing 'stale' tag should be cleared, {tagged} "
            f"event(s) still carry a cycle value")
        print("[ok] all-Wake night: no ValueError, 0 cycles, stage_durations "
              "written, and the pre-existing tag was cleared")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TESTING CYCLE RE-RUN REGRESSIONS (4.3.1)")
    print("=========================================")

    test_changed_threshold_rerun_no_stale_tags()
    test_zero_cycle_rerun_replaces_everything()
    test_unscored_hypnogram_is_refused()
    test_all_wake_night_is_accepted()

    print("\nAll cycle re-run tests completed!")
