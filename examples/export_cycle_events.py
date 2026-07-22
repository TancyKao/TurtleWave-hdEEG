"""Export per-cycle event density and feature summaries from neural_events.db.

Reads detected events that have already been tagged with a sleep cycle (run
``backfill_cycles.py`` / ``finalize_cycles_and_durations`` first) and writes two
CSVs:

``cycle_events_summary.csv``
    One row per ``(subject, cycle_number, event_type, method, channel)`` with an
    event count, that cycle's duration context, a **density**, and per-feature
    summary statistics (peak-to-peak amplitude, duration, min/max amplitude).

``cycle_events_raw.csv``
    Every cycle-tagged event, with its cycle-duration context columns appended,
    so any grouping or normalisation can be recomputed downstream.

Density definition
------------------
Density is computed **per cycle, normalised by that cycle's full NREM period**::

    density_per_min = count / cycle_nrem_dur_min

An N2+N3-only density is emitted alongside it in ``density_n23_per_min``::

    density_n23_per_min = count / cycle_nrem_n23_dur_min

both raw durations (``cycle_nrem_dur_min`` full period, ``cycle_nrem_n23_dur_min``
N2+N3 only) are kept as adjacent columns so either denominator can be recomputed
downstream. Cycle durations come from the ``sleep_cycles`` table for the
``'2022'`` definition (the method that owns ``events.cycle``), joined on cycle
number.

CAVEAT: normalising by NREM duration is appropriate for NREM events (slow waves,
spindles, K-complexes). For REM-locked events, divide by REM duration
(``cycle_rem_min``) instead; the NREM denominator would understate their density.

Events with a NULL / empty cycle (outside any detected cycle) are excluded from
both CSVs, and the excluded count is printed so nothing is dropped silently.

Usage
-----
Edit the CONFIG block below, then::

    python examples/export_cycle_events.py

Point either ``DB_PATH`` at a single ``neural_events.db`` **or** ``ROOT`` at a
directory of subject folders (``ROOT/<subj>/wonambi/neural_events.db``); when
``ROOT`` is used, subjects are concatenated into the same CSVs with a ``subject``
column. If both are set, ``DB_PATH`` wins.
"""

import csv
import glob
import os
import sqlite3

import numpy as np

# ===========================================================================
# CONFIG
# ===========================================================================

# Single database to export. Leave as None to walk ROOT instead.
DB_PATH = None

# Root directory of subject folders (used only when DB_PATH is None).
# Each subject folder must contain wonambi/neural_events.db.
ROOT = "/Users/tancykao/Library/CloudStorage/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/Emotion"

# Where the two CSVs are written.
OUTPUT_DIR = "./cycle_event_exports"

# Cycle definition whose durations/tagging are used (must match tag_method).
CYCLE_METHOD = "2022"

# ===========================================================================
# End of CONFIG
# ===========================================================================

# Feature columns summarised in the summary CSV.
SUMMARY_FIELDS = [
    "subject", "cycle_number", "event_type", "method", "channel",
    "count",
    "cycle_nrem_dur_min", "cycle_nrem_n23_dur_min",
    "cycle_rem_min", "cycle_dur_min",
    "density_per_min", "density_n23_per_min",
    "peak2peak_amp_mean", "peak2peak_amp_sd", "peak2peak_amp_median",
    "duration_mean", "duration_sd", "duration_median",
    "min_amp_mean", "max_amp_mean",
]

# Full event schema (kept explicit so the raw CSV column order is stable).
EVENT_COLS = [
    "uuid", "event_type", "channel", "start_time", "end_time", "duration",
    "start_time_hms", "stage", "cycle", "method", "freq_band",
    "freq_lower", "freq_upper", "min_amp", "max_amp", "peak2peak_amp",
    "processing_timestamp", "n_fft_sec",
]

# Cycle-context columns appended to every raw row (output name -> sleep_cycles col).
CYCLE_CONTEXT = [
    ("cycle_nrem_dur_min", "nrem_dur_min"),
    ("cycle_nrem_n23_dur_min", "nrem_n23_dur_min"),
    ("cycle_rem_min", "rem_dur_min"),
    ("cycle_dur_min", "cycle_dur_min"),
]

RAW_FIELDS = ["subject"] + EVENT_COLS + [out for out, _ in CYCLE_CONTEXT]


def find_databases():
    """Resolve the list of databases to export.

    Returns
    -------
    list of str
        Absolute paths to ``neural_events.db`` files.
    """
    if DB_PATH:
        return [DB_PATH]
    if not ROOT or not os.path.isdir(ROOT):
        return []
    pattern = os.path.join(ROOT, "*", "wonambi", "neural_events.db")
    return sorted(glob.glob(pattern))


def db_subjects(conn):
    """Return the subject id(s) recorded in ``sleep_cycles`` for CYCLE_METHOD.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to a ``neural_events.db``.

    Returns
    -------
    list of str
        Distinct subject ids; empty if cycles were never finalized.
    """
    cur = conn.execute(
        "SELECT DISTINCT subject FROM sleep_cycles WHERE method = ?",
        (CYCLE_METHOD,))
    return [r[0] for r in cur.fetchall()]


def _fnum(x):
    """Format a numeric value for CSV: empty string for None/NaN, else the value.

    Parameters
    ----------
    x : float or None
        Value to render.

    Returns
    -------
    float or str
        ``x`` unchanged, or ``""`` when it is None/NaN.
    """
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return x


def export_database(db_path, raw_writer, summary_rows):
    """Export one database: stream raw rows, accumulate summary groups.

    Parameters
    ----------
    db_path : str
        Path to ``neural_events.db``.
    raw_writer : csv.DictWriter
        Writer for the raw CSV (rows are written as they are read).
    summary_rows : list
        Accumulator that receives finished summary-row dicts.

    Returns
    -------
    dict
        ``{"subject", "n_raw", "n_excluded", "n_unmatched", "n_total",
        "n_groups"}`` for the per-database report line, where
        ``n_total == n_raw + n_excluded + n_unmatched`` accounts for every
        event row.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    subjects = db_subjects(conn)
    if not subjects:
        conn.close()
        raise RuntimeError(
            "no rows in sleep_cycles for method="
            f"'{CYCLE_METHOD}' - run backfill_cycles.py first")
    # One DB is one subject in this pipeline; if several are present each is
    # filtered explicitly by the join below.
    subject = subjects[0]
    if len(subjects) > 1:
        print(f"    WARNING: multiple subjects found in sleep_cycles "
              f"({', '.join(map(str, subjects))}); exporting only {subject}")

    # Total events, and those with no cycle at all, so every row is accounted
    # for: total must equal n_raw + n_excluded + n_unmatched (see below).
    n_total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_excluded = conn.execute(
        "SELECT COUNT(*) FROM events WHERE cycle IS NULL OR cycle = ''"
    ).fetchone()[0]

    # Join each cycle-tagged event to its 2022 cycle row for the duration
    # context. events.cycle is TEXT ('1'..) and cycle_number is INTEGER.
    query = f"""
        SELECT e.*,
               sc.nrem_dur_min      AS cycle_nrem_dur_min,
               sc.nrem_n23_dur_min  AS cycle_nrem_n23_dur_min,
               sc.rem_dur_min       AS cycle_rem_min,
               sc.cycle_dur_min     AS cycle_dur_min
        FROM events e
        JOIN sleep_cycles sc
          ON CAST(e.cycle AS INTEGER) = sc.cycle_number
         AND sc.method = ?
         AND sc.subject = ?
        WHERE e.cycle IS NOT NULL AND e.cycle != ''
    """

    # Accumulate feature values per summary group.
    groups = {}
    n_raw = 0
    cur = conn.execute(query, (CYCLE_METHOD, subject))
    for row in cur:
        n_raw += 1

        raw = {"subject": subject}
        for col in EVENT_COLS:
            raw[col] = row[col]
        for out_name, _src in CYCLE_CONTEXT:
            raw[out_name] = row[out_name]
        raw_writer.writerow(raw)

        key = (subject, int(row["cycle"]), row["event_type"],
               row["method"], row["channel"])
        g = groups.get(key)
        if g is None:
            g = {
                "cycle_nrem_dur_min": row["cycle_nrem_dur_min"],
                "cycle_nrem_n23_dur_min": row["cycle_nrem_n23_dur_min"],
                "cycle_rem_min": row["cycle_rem_min"],
                "cycle_dur_min": row["cycle_dur_min"],
                "p2p": [], "dur": [], "min_amp": [], "max_amp": [],
            }
            groups[key] = g
        g["p2p"].append(row["peak2peak_amp"])
        g["dur"].append(row["duration"])
        g["min_amp"].append(row["min_amp"])
        g["max_amp"].append(row["max_amp"])

    conn.close()

    # Finalize summary rows for this database.
    for key, g in groups.items():
        subj, cyc_num, event_type, method, channel = key
        count = len(g["p2p"])
        nrem = g["cycle_nrem_dur_min"]
        n23 = g["cycle_nrem_n23_dur_min"]
        density = count / nrem if nrem else float("nan")
        density_n23 = count / n23 if n23 else float("nan")

        summary_rows.append({
            "subject": subj,
            "cycle_number": cyc_num,
            "event_type": event_type,
            "method": method,
            "channel": channel,
            "count": count,
            "cycle_nrem_dur_min": _fnum(nrem),
            "cycle_nrem_n23_dur_min": _fnum(g["cycle_nrem_n23_dur_min"]),
            "cycle_rem_min": _fnum(g["cycle_rem_min"]),
            "cycle_dur_min": _fnum(g["cycle_dur_min"]),
            "density_per_min": _fnum(density),
            "density_n23_per_min": _fnum(density_n23),
            "peak2peak_amp_mean": _stat(g["p2p"], np.mean),
            "peak2peak_amp_sd": _stat(g["p2p"], _sd),
            "peak2peak_amp_median": _stat(g["p2p"], np.median),
            "duration_mean": _stat(g["dur"], np.mean),
            "duration_sd": _stat(g["dur"], _sd),
            "duration_median": _stat(g["dur"], np.median),
            "min_amp_mean": _stat(g["min_amp"], np.mean),
            "max_amp_mean": _stat(g["max_amp"], np.mean),
        })

    # Events with a non-null cycle that matched no sleep_cycles row for
    # (subject, CYCLE_METHOD) are dropped by the inner join. Recover the count
    # by difference so nothing is ever lost silently.
    n_unmatched = n_total - n_raw - n_excluded

    return {"subject": subject, "n_raw": n_raw, "n_excluded": n_excluded,
            "n_unmatched": n_unmatched, "n_total": n_total,
            "n_groups": len(groups)}


def _sd(values):
    """Sample standard deviation (ddof=1); NaN when fewer than 2 values.

    Parameters
    ----------
    values : array-like
        Feature values for one group.

    Returns
    -------
    float
        Sample SD, or NaN if n < 2.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


def _stat(values, fn):
    """Apply a stat function to a value list, guarding empties, format for CSV.

    Parameters
    ----------
    values : list of float
        Feature values (may contain None).
    fn : callable
        Aggregator such as ``np.mean``, ``np.median`` or ``_sd``.

    Returns
    -------
    float or str
        The statistic, or ``""`` when it is undefined.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return ""
    return _fnum(float(fn(arr)))


def main():
    databases = find_databases()
    if not databases:
        print("No databases found. Set DB_PATH or a valid ROOT in CONFIG.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "cycle_events_summary.csv")
    raw_path = os.path.join(OUTPUT_DIR, "cycle_events_raw.csv")

    print(f"Exporting {len(databases)} database(s) "
          f"(cycle method '{CYCLE_METHOD}') to:\n  {OUTPUT_DIR}\n")

    summary_rows = []
    total_raw = 0
    total_excluded = 0
    total_unmatched = 0
    total_events = 0

    with open(raw_path, "w", newline="") as raw_fh:
        raw_writer = csv.DictWriter(raw_fh, fieldnames=RAW_FIELDS)
        raw_writer.writeheader()

        for db_path in databases:
            try:
                rep = export_database(db_path, raw_writer, summary_rows)
            except Exception as exc:  # noqa: BLE001 - one bad DB must not abort
                print(f"FAIL {db_path}: {exc}")
                continue
            total_raw += rep["n_raw"]
            total_excluded += rep["n_excluded"]
            total_unmatched += rep["n_unmatched"]
            total_events += rep["n_total"]
            print(f"[{rep['subject']}] {rep['n_raw']} cycle-tagged events, "
                  f"{rep['n_excluded']} excluded (null cycle), "
                  f"{rep['n_groups']} summary rows")
            if rep["n_unmatched"]:
                print(f"    WARNING: {rep['n_unmatched']} cycle-tagged but "
                      f"unmatched in sleep_cycles (dropped)")

    # Sort: subject, event_type, channel, cycle_number.
    summary_rows.sort(key=lambda r: (
        r["subject"], r["event_type"], r["channel"], r["cycle_number"]))

    with open(summary_path, "w", newline="") as sum_fh:
        writer = csv.DictWriter(sum_fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 60)
    print(f"Total events                     : {total_events}")
    print(f"Excluded (cycle IS NULL/empty)   : {total_excluded}")
    print(f"Unmatched in sleep_cycles        : {total_unmatched}")
    print(f"Raw rows written                 : {total_raw}  -> {raw_path}")
    print(f"Summary rows written             : {len(summary_rows)}  -> {summary_path}")
    # Full accounting: every event row is either exported, null-cycle, or
    # cycle-tagged-but-unmatched. If this ever fails, a row was lost silently.
    accounted = total_raw + total_excluded + total_unmatched
    status = "OK" if accounted == total_events else "MISMATCH"
    print(f"Accounting (raw+excluded+unmatched == total): "
          f"{accounted} == {total_events}  [{status}]")


if __name__ == "__main__":
    main()
