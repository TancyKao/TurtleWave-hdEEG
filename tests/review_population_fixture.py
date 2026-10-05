"""Synthetic ``neural_events.db`` with 4.6 per-event figures, for the review
GUI's population-check and event-panel tests.

The schema is the library's own (``ensure_direct_write_schema`` on top of the
pre-direct-write ``events`` table), so the fixture tracks the real columns.
Values are drawn per channel from a few named profiles so a test can count
the expected shares by hand from the same arrays it inserted.
"""
import json
import sqlite3
import uuid as _uuid

#: Namespace of the fixture's event uuids (fixed, so a seeded run repeats).
_FIXTURE_NS = _uuid.UUID('2b1f0e6c-5d4a-4c3b-9a8f-7e6d5c4b3a29')

import numpy as np

BASE_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
    uuid TEXT PRIMARY KEY, event_type TEXT, channel TEXT,
    start_time REAL, end_time REAL, duration REAL, start_time_hms TEXT,
    stage TEXT, cycle TEXT, method TEXT, freq_band TEXT, freq_lower REAL,
    freq_upper REAL, min_amp REAL, max_amp REAL, peak2peak_amp REAL,
    rms REAL, power REAL, peak_power_freq REAL, energy REAL,
    peak_energy_freq REAL, processing_timestamp TEXT, n_fft_sec INTEGER,
    CONSTRAINT event_chan_time UNIQUE (event_type, channel, start_time,
        method, freq_lower, freq_upper, stage))
"""

#: Columns written per event row, in insert order.
ROW_COLUMNS = (
    'uuid', 'event_type', 'channel', 'start_time', 'end_time', 'duration',
    'stage', 'method', 'freq_lower', 'freq_upper', 'min_amp', 'max_amp',
    'peak2peak_amp', 'run_id', 'epoch_stage',
    'halfwaves_above_bg', 'cycles_nominal', 'peak_freq_ap', 'prominence_db',
    'in_band', 'low_prominence', 'bg_rms', 'bg_n_windows', 'bg_stage_mixed',
    'amp_ratio', 'thresh_ratio', 'near_bound', 'near_splice', 'wave_freq',
    'peak_freq', 'peak_val_det', 'det_trough', 'det_ptp', 'det_zero_time',
)


def open_schema(path):
    """Connection to a new database carrying the library's 4.6 schema."""
    from turtlewave_hdEEG import dbwrite
    con = sqlite3.connect(path)
    con.execute(BASE_EVENTS_DDL)
    dbwrite.ensure_direct_write_schema(con)
    con.commit()
    return con


def add_run(con, run_id, event_type='spindle', method='Moelle2011',
            band=(9.0, 12.0), stages=('NREM2', 'NREM3'), figures=True,
            duration=(0.5, 3.0), ref_chan=None, timestamp='2026-09-14T10:00:00',
            version='4.6.0', ref_in_params=True, figures_off=False):
    """One ``detection_runs`` row, written the way the library writes it:
    ``stages`` and ``ref_chan`` columns hold ``str()`` of the list
    (``"['NREM2', 'NREM3']"``); ``figures=False`` mimics a 4.5 run (no
    ``event_figures`` key), ``figures_off=True`` a 4.6 run detected with the
    figures switched off (``event_figures: None``)."""
    params = {'frequency': list(band), 'duration': list(duration),
              'duration_by_method': {method: list(duration)},
              'ref_chan': list(ref_chan or []),
              'reject_types': ['Artefact', 'Arousal'],
              'event_figures': {'spec_revision': 'v2'}}
    if not ref_in_params:
        params.pop('ref_chan')
    if figures_off:
        params['event_figures'] = None
    elif not figures:
        params.pop('event_figures')
    stages_col = str(list(stages)) if isinstance(stages, (list, tuple)) \
        else str(stages)
    con.execute(
        "INSERT INTO detection_runs (run_id, subject, event_type, method, "
        "params_json, ref_chan, stages, reject_types, reject_artifacts, "
        "reject_arousals, turtlewave_version, timestamp) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, 'sub-fx', event_type, method, json.dumps(params),
         str(list(ref_chan or [])), stages_col, 'Arousal,Artefact', 1, 1,
         version, timestamp))


def insert_rows(con, rows):
    sql = (f"INSERT INTO events ({', '.join(ROW_COLUMNS)}) "
           f"VALUES ({', '.join('?' * len(ROW_COLUMNS))})")
    con.executemany(sql, rows)


def make_rows(channel, n, run_id, rng, *, event_type='spindle',
              method='Moelle2011', band=(9.0, 12.0), off_band=0.07,
              low_prom=0.3, at_floor=0.1, amp_ratio=2.5, thresh_ratio=1.6,
              stages=('NREM2', 'NREM3'), t0=0.0, figures=True,
              no_peak=0.0, floor=0.5, at_ceiling=0.0, ceiling=3.0,
              off_peaks=None):
    """``n`` event rows for one channel; shares are exact fractions of ``n``.

    Returns the rows and a dict of the per-event arrays used, so a test can
    recount any share by hand.
    """
    k_off = int(round(off_band * n))
    k_low = int(round(low_prom * n))
    k_floor = int(round(at_floor * n))
    k_nopeak = int(round(no_peak * n))
    in_band = np.ones(n, dtype=int)
    in_band[:k_off] = 0
    low = np.zeros(n, dtype=int)
    low[n - k_low:] = 1
    near = np.zeros(n, dtype=int)
    near[k_off:k_off + k_floor] = -1
    k_ceil = int(round(at_ceiling * n))
    near[k_off + k_floor:k_off + k_floor + k_ceil] = 1
    dur = np.where(near == -1, floor + 0.02,
                   np.where(near == 1, ceiling - 0.02,
                            floor + 0.4 + rng.uniform(0, 0.6, n)))
    amp = np.full(n, float(amp_ratio)) * rng.uniform(0.8, 1.2, n)
    thr = np.full(n, float(thresh_ratio)) * rng.uniform(0.9, 1.1, n)
    stage = np.array([stages[i % len(stages)] for i in range(n)])
    starts = t0 + np.arange(n) * 7.0 + rng.uniform(0, 1, n)
    pf = np.where(in_band == 1, (band[0] + band[1]) / 2.0, band[0] - 1.5)
    if off_peaks:
        for j, i in enumerate(np.flatnonzero(in_band == 0)):
            pf[i] = off_peaks[j % len(off_peaks)]
    peak_nan = np.zeros(n, dtype=bool)
    peak_nan[k_off:k_off + k_nopeak] = True
    rows = []
    for i in range(n):
        if figures:
            fig = (7, 6.0, None if peak_nan[i] else float(pf[i]),
                   None if peak_nan[i] else (7.2 if low[i] else 15.0),
                   None if peak_nan[i] else int(in_band[i]),
                   None if peak_nan[i] else int(low[i]),
                   1.8, 55, 0, float(amp[i]), float(thr[i]), int(near[i]),
                   0, None)
        else:
            fig = (None,) * 14
        s = float(starts[i])
        # deterministic: uuid5 of a number from the caller's seeded rng
        uid = str(_uuid.uuid5(_FIXTURE_NS, f"{channel}|{run_id}|{i}|"
                                           f"{int(rng.integers(2 ** 62))}"))
        rows.append((uid, event_type, channel, s,
                     s + float(dur[i]), float(dur[i]), '+'.join(stages),
                     method, band[0], band[1], -20.0, 40.0, 60.0, run_id,
                     str(stage[i])) + fig + (9.8, 5.16, None, None, None))
    return rows, {'in_band': np.where(peak_nan, -1, in_band), 'low': low,
                  'near': near, 'amp': amp, 'thr': thr, 'stage': stage,
                  'starts': starts, 'dur': dur, 'peak_nan': peak_nan}
