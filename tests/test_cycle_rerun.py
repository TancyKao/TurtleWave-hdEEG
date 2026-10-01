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
# Plus the two follow-ups closed after the 4.3.1 quality gate:
#
#   5. store_cycles_to_database's two directions: an empty cycles list WITH a
#      real method is the valid "clear the store" call and must keep working,
#      while a call that cannot name the rows to replace (empty cycles and
#      method=None, or a cycle dict carrying method=None) must raise
#      ValueError and write nothing -- it used to warn and silently leave the
#      previous run's rows behind. Non-empty cycles without method= still
#      work, reading the method off the cycles (as tests/test_turtlewave.py
#      and tests/test_cycle_connection_reuse.py call it).
#   6. examples/backfill_cycles.py counts a subject whose database was written
#      but whose plot failed as a PASS with a warning, and still counts a
#      database-write failure as a FAIL.
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

import contextlib
import importlib.util
import io
import logging
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


# --------------------------------------------------------------------------
# 5. store_cycles_to_database: clearing needs a method, and says so loudly
# --------------------------------------------------------------------------

def _cycle_dict(method, cycle_number=1, nrem_start=0.0):
    """One minimal cycle row, shaped as :meth:`ParalCycles.detect` returns."""
    return {'method': method, 'cycle_number': cycle_number,
            'nrem_start_sec': nrem_start, 'nrem_end_sec': nrem_start + 600.0,
            'rem_end_sec': nrem_start + 900.0, 'nrem_dur_min': 10.0,
            'nrem_n23_dur_min': 8.0, 'rem_dur_min': 5.0,
            'cycle_dur_min': 15.0}


def _n_cycle_rows(db, subject):
    return _fetch(db, 'SELECT COUNT(*) FROM sleep_cycles WHERE subject=?',
                  (subject,))[0][0]


def test_store_cycles_method_required_to_identify_rows():
    """Empty cycles + a real method clears; nothing-to-key-on raises.

    The 4.3.1 gate flagged that ``store_cycles_to_database(method=None,
    cycles=[])`` only warned. The delete is keyed on ``(subject, method)`` and
    ``method=NULL`` matches no row, so the call removed nothing, inserted
    nothing and returned 0 while ``tag_events_with_cycles([])`` had already
    cleared ``events.cycle`` and the XML markers were gone -- the previous
    run's ``sleep_cycles`` rows surviving as the only description of cycles
    that no longer existed anywhere else. It now raises ValueError, matching
    ``ParalCycles.run`` / ``finalize_cycles_and_durations``, which already
    raise on an unscorable hypnogram (case 3).

    Both directions are asserted, because over-raising would be just as wrong:
    ``cycles=[]`` with a real ``method`` is the valid zero-cycle clear that
    case 2 depends on, and non-empty cycles with no ``method=`` argument is
    how tests/test_turtlewave.py and tests/test_cycle_connection_reuse.py
    call it (the method is read off the cycles themselves).
    """
    print("\n5. store_cycles_to_database needs a method to name the rows it "
          "replaces:")
    tmp = _tmp_dir('storemethod')
    try:
        subject = 'sub-m'
        pc = cp.ParalCycles(annotations=None, subject=subject,
                            log_level=logging.CRITICAL)
        # An existing detection DB: _ensure_sleep_cycles_table indexes
        # events.cycle, so the events table has to be there.
        db = os.path.join(tmp, 'neural_events.db')
        _make_events_db(db, 3)

        # (a) Non-empty cycles, no method= -> method read off the cycles.
        n_written = pc.store_cycles_to_database(
            [_cycle_dict('2022')], db, subject=subject)
        rows = _n_cycle_rows(db, subject)
        print(f"   non-empty cycles, method= omitted -> wrote {n_written}, "
              f"sleep_cycles rows={rows}")
        assert n_written == 1 and rows == 1, (
            "non-empty cycles must still store without an explicit method= "
            "(it is read off the cycles); raising here would break "
            "test_turtlewave.py and test_cycle_connection_reuse.py")

        # (b) Empty cycles WITH a method -> the valid clear.
        n_cleared = pc.store_cycles_to_database(
            [], db, subject=subject, method='2022')
        rows = _n_cycle_rows(db, subject)
        print(f"   cycles=[] with method='2022' -> returned {n_cleared}, "
              f"sleep_cycles rows={rows}")
        assert n_cleared == 0 and rows == 0, (
            "cycles=[] with a real method must delete the subject's rows for "
            f"that method (the zero-cycle re-run path), found {rows} row(s)")

        # Re-seed for the raising cases, which must write nothing.
        pc.store_cycles_to_database([_cycle_dict('2022')], db,
                                    subject=subject, method='2022')
        assert _n_cycle_rows(db, subject) == 1

        # (c) Empty cycles AND method=None -> raises, previous rows survive.
        raised = None
        try:
            pc.store_cycles_to_database([], db, subject=subject, method=None)
        except ValueError as e:
            raised = e
        rows = _n_cycle_rows(db, subject)
        print(f"   cycles=[] with method=None -> "
              f"{'ValueError' if raised else 'DID NOT RAISE'}, "
              f"sleep_cycles rows still {rows}")
        assert raised is not None, (
            "cycles=[] with method=None must raise ValueError: nothing names "
            "the rows to replace, so it used to warn and silently leave the "
            "previous run's rows in place")
        assert 'method' in str(raised), (
            f"the error should tell the caller to pass method=, got: {raised}")
        assert rows == 1, (
            f"the refused call must write nothing: the pre-existing row "
            f"should survive untouched, found {rows} row(s)")

        # (d) A cycle dict carrying method=None -> also raises, writes nothing.
        raised = None
        try:
            pc.store_cycles_to_database([_cycle_dict(None)], db,
                                        subject=subject, method='2022')
        except ValueError as e:
            raised = e
        rows = _n_cycle_rows(db, subject)
        print(f"   cycle dict with method=None -> "
              f"{'ValueError' if raised else 'DID NOT RAISE'}, "
              f"sleep_cycles rows still {rows}")
        assert raised is not None, (
            "a cycle dict with method=None would be inserted with a NULL "
            "method that no later replacement can match; it must raise")
        assert rows == 1, (
            f"the refused call must write nothing, found {rows} row(s)")

        # (e) The raise happens before a connection is opened, so a mistaken
        #     call cannot even create the database file.
        missing = os.path.join(tmp, 'not_created.db')
        raised = None
        try:
            pc.store_cycles_to_database([], missing, subject=subject)
        except ValueError as e:
            raised = e
        exists = os.path.exists(missing)
        print(f"   refused call on a missing db path -> "
              f"{'ValueError' if raised else 'DID NOT RAISE'}, "
              f"file created={exists}")
        assert raised is not None and not exists, (
            "the check must run before open_write_connection, which would "
            "create the missing file for a call that then writes nothing")

        print("[ok] cycles=[] with a real method still clears the store; a "
              "call that cannot name the rows to replace raises ValueError "
              "and writes nothing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# 6. examples/backfill_cycles.py: a plot-only failure is not a subject failure
# --------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKFILL_PATH = os.path.join(_REPO_ROOT, 'examples', 'backfill_cycles.py')


def _load_backfill():
    """Import examples/backfill_cycles.py by path, without running main().

    Loaded fresh for each case so one case's monkeypatched module globals
    (ROOT, PLOT, finalize_cycles_and_durations) cannot leak into the next.
    """
    spec = importlib.util.spec_from_file_location(
        'backfill_cycles_undertest', _BACKFILL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # guarded by if __name__ == '__main__'
    return mod


def _backfill_fixture(tmp, folder='10sd'):
    """A ROOT/<folder>/wonambi/{neural_events.db, sub-*.xml} tree to backfill."""
    wonambi_dir = os.path.join(tmp, folder, 'wonambi')
    os.makedirs(wonambi_dir)
    hyp = _threshold_sensitive_hypnogram()
    _build_annotations(wonambi_dir, len(hyp), stem=f'sub-{folder}')
    ann = CustomAnnotations(os.path.join(wonambi_dir, f'sub-{folder}.xml'))
    _stage(ann, hyp)
    db = os.path.join(wonambi_dir, 'neural_events.db')
    _make_events_db(db, int(len(hyp) * EPOCH_LENGTH // 60) + 2)
    return db


def _run_backfill(mod):
    """Run mod.main() on the module's test root with stdout captured.

    The script takes its root from a required ``--root`` argument (no
    built-in default), so the root and subjects set on the module by each
    case are passed as arguments, with ``--yes`` for the confirmation.
    """
    argv = ['--root', mod.ROOT, '--subjects', *mod.SUBJECTS, '--yes']
    if not mod.PLOT:
        argv.append('--no-plot')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main(argv)
    return buf.getvalue()


def test_backfill_plot_failure_is_a_warning_not_a_failure():
    """A failing plot must not turn a written database into a FAIL.

    The gate flagged that the script called finalize_cycles_and_durations with
    plot=True, so an exception from the plotting step -- missing matplotlib,
    an unwritable output dir, a backend problem -- propagated out of a call
    whose sleep_cycles / stage_durations / events.cycle writes had ALREADY
    succeeded, and the per-subject except block counted the subject as FAIL.
    Re-running to "fix" it then re-did work that was already correct. The
    script now plots separately, after the return, and reports a plot failure
    as a warning on a passing subject. A failure at or before the database
    write must still count as FAIL.
    """
    print("\n6. backfill_cycles.py: plot failure warns (PASS), DB failure "
          "still FAILs:")
    tmp = _tmp_dir('backfill')
    try:
        # (a) Plot raises; the database write succeeds.
        db = _backfill_fixture(tmp, folder='10sd')
        mod = _load_backfill()
        mod.ROOT = tmp
        mod.SUBJECTS = ['10sd']
        mod.PLOT = True

        from turtlewave_hdEEG import cycleplot
        original_plot = cycleplot.plot_from_annotations

        def _boom(*args, **kwargs):
            raise RuntimeError('matplotlib exploded')

        cycleplot.plot_from_annotations = _boom
        try:
            out = _run_backfill(mod)
        finally:
            cycleplot.plot_from_annotations = original_plot

        cycle_rows = _fetch(db, 'SELECT COUNT(*) FROM sleep_cycles')[0][0]
        stage_rows = _fetch(db, 'SELECT COUNT(*) FROM stage_durations')[0][0]
        tally = [ln for ln in out.splitlines() if ln.startswith('Done.')]
        warn = [ln for ln in out.splitlines() if 'plot skipped' in ln]
        print(f"   sleep_cycles rows={cycle_rows}, stage_durations rows="
              f"{stage_rows}")
        print(f"   {warn[0].strip() if warn else 'NO WARN LINE'}")
        print(f"   {tally[0] if tally else 'NO TALLY LINE'}")

        assert cycle_rows > 0 and stage_rows == 1, (
            "the database work must have completed before the plot was "
            f"attempted (sleep_cycles={cycle_rows}, "
            f"stage_durations={stage_rows})")
        assert 'PASS:' in out, "the subject should be reported as a PASS"
        assert warn, (
            "a plot failure should print a 'database written, plot skipped' "
            f"warning; got:\n{out}")
        assert 'matplotlib exploded' in warn[0], (
            f"the plot error message must appear in the warning line, got: "
            f"{warn[0]}")
        assert tally and '1 passed, 0 failed' in tally[0], (
            f"a plot-only failure must not count as a failed subject, got: "
            f"{tally}")
        assert 'FAIL:' not in out, (
            f"nothing should be reported as FAIL here; got:\n{out}")

        # (b) The database write itself fails -> still FAIL.
        mod2 = _load_backfill()
        mod2.ROOT = tmp
        mod2.SUBJECTS = ['10sd']
        mod2.PLOT = True

        def _db_boom(*args, **kwargs):
            raise RuntimeError('database write exploded')

        mod2.finalize_cycles_and_durations = _db_boom
        out2 = _run_backfill(mod2)
        tally2 = [ln for ln in out2.splitlines() if ln.startswith('Done.')]
        print(f"   {tally2[0] if tally2 else 'NO TALLY LINE'}")

        assert 'FAIL: database write exploded' in out2, (
            f"a database write failure must still be reported as FAIL, got:\n"
            f"{out2}")
        assert tally2 and '0 passed, 1 failed' in tally2[0], (
            f"a database write failure must still count as a failed subject, "
            f"got: {tally2}")

        # (c) Unpatched: the PNG is actually drawn. Without this, plot_cycles'
        #     catch-all except would turn a wrong call signature into a
        #     permanent "plot skipped" warning that no test ever notices.
        mod3 = _load_backfill()
        mod3.ROOT = tmp
        mod3.SUBJECTS = ['10sd']
        mod3.PLOT = True
        try:
            import matplotlib
            matplotlib.use('Agg')
            have_mpl = True
        except ImportError:
            have_mpl = False
        if have_mpl:
            out3 = _run_backfill(mod3)
            pngs = [f for f in os.listdir(os.path.dirname(db))
                    if f.endswith('.png')]
            print(f"   unpatched run: PNG(s) written={pngs}")
            assert 'plot skipped' not in out3, (
                f"plotting should succeed with matplotlib present; got:\n"
                f"{out3}")
            assert len(pngs) == 1 and 'hypnogram_cycles' in pngs[0], (
                f"the hypnogram/cycle PNG should be beside the database, "
                f"found {pngs}")
            assert f"plot: " in out3, "the PASS line should name the PNG path"
        else:
            print("   unpatched run: SKIPPED (matplotlib not installed)")

        print("[ok] a plot-only failure is a PASS with a warning naming the "
              "plot error; a database write failure is still a FAIL; and the "
              "unpatched plot path still writes its PNG")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TESTING CYCLE RE-RUN REGRESSIONS (4.3.1)")
    print("=========================================")

    test_changed_threshold_rerun_no_stale_tags()
    test_zero_cycle_rerun_replaces_everything()
    test_unscored_hypnogram_is_refused()
    test_all_wake_night_is_accepted()
    test_store_cycles_method_required_to_identify_rows()
    test_backfill_plot_failure_is_a_warning_not_a_failure()

    print("\nAll cycle re-run tests completed!")
