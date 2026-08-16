# How to Run PAC Analysis

This guide shows you how to run phase-amplitude coupling (PAC) analysis —
e.g. slow-wave/spindle coupling — on already-detected events in
`neural_events.db`, using `ParalPAC.analyze_pac`.

This guide covers **running** PAC analysis. If you already have PAC result
CSVs from an older run and just need to load them into the database, see
[Back-fill PAC Results into the Database](backfill-pac-to-database.md)
instead.

## Prerequisites

- Slow wave and/or spindle events already detected into `neural_events.db`
  (see [Detect Slow Waves](detect-slow-waves.md) / [Detect Spindles](detect-spindles.md))
- `tensorpac` installed (PAC analysis is built on it)

!!! warning "Preferred-phase fix"
    Versions before the 4.0 preferred-phase fix reported
    `preferred_phase_rad` / `preferred_phase_deg` 180 degrees off. If you are
    migrating results produced before this fix, read
    [Upgrade to 4.0 — Step 3](upgrade-to-4.0.md#step-3-regenerate-pac-preferred-phase)
    before trusting any preferred-phase values from old runs.

## Run coupling analysis between detected slow waves and spindles

Mirror
[`examples/hdEEG_pac_detector.py`](https://github.com/TancyKao/TurtleWave-hdEEG/blob/master/examples/hdEEG_pac_detector.py):

```python
from wonambi.dataset import Dataset as WonambiDataset
from turtlewave_hdEEG import ParalPAC, CustomAnnotations

data = WonambiDataset('subject001.set')
annot = CustomAnnotations('subject001_annotations.xml')

pac_processor = ParalPAC(
    dataset=data,
    annotations=annot,
    rootpath='data/subject001',
)

event_opts = {
    'buffer': 1.0,               # seconds of context around each event
    'sw_method': 'Staresina2015',
    'spindle_method': 'Moelle2011',
    'sw_freq_range': None,       # or e.g. [0.5, 2.0] to filter SW events by freq
    'spindle_freq_range': None,
    'stages': 'NREM2NREM3',      # combined stage string, not a list
}

result = pac_processor.analyze_pac(
    chan=['E110', 'E111', 'E112'],
    stage=['NREM2', 'NREM3'],
    phase_freq=(0.5, 1.25),      # slow-wave phase band
    amp_freq=(11, 16),           # spindle amplitude band
    idpac=(1, 2, 4),             # (method, surrogate, correction) — see below
    use_detected_events=True,
    event_type='slow_wave',
    pair_with_spindles=True,     # pair each slow wave with a nearby spindle
    time_window=1.0,             # seconds to search for a spindle around each SW
    db_path='data/subject001/wonambi/neural_events.db',
    out_dir='data/subject001/wonambi/pac_results',
    event_opts=event_opts,
)
```

`analyze_pac` reads events straight out of `neural_events.db` (rather than
re-detecting) when `use_detected_events=True`; `event_type` and
`pair_with_spindles` control which events anchor the analysis:

- `event_type='slow_wave'`, `pair_with_spindles=False` — PAC on slow-wave
  windows alone.
- `event_type='slow_wave'`, `pair_with_spindles=True` — slow-wave/spindle
  coupling: each slow wave is paired with a spindle found within
  `time_window` seconds, and PAC is computed on the paired window. This is
  the canonical slow-wave-phase / spindle-amplitude coupling analysis.
- `event_type='spindle'` — PAC on spindle windows alone.

`idpac` selects the PAC method, surrogate method, and normalization,
following `tensorpac`'s numbering:

| idpac[0] method | idpac[1] surrogate | idpac[2] correction |
|---|---|---|
| 1 = Mean Vector Length (Canolty 2006) | 0 = none | 0 = none |
| 2 = Modulation Index (Tort 2010) | 1 = swap phase/amplitude across trials | 1 = subtract surrogate mean |
| 3 = Heights Ratio (Lakatos 2005) | 2 = swap amplitude time blocks | 2 = divide by surrogate mean |
| 4 = ndPAC (Ozkurt 2012) | 3 = time lag (Canolty 2006) | 3 = subtract then divide |
| 5 = Phase-Locking Value | | 4 = Z-score |
| 6 = Gaussian Copula PAC | | |

Call `pac_processor.pac_method(0, 0, 0, list_methods=True)` to get these
descriptions programmatically at runtime.

## Write results straight to the database

Pass `write_db=True` (and `subject=...`) to persist the per-channel PAC
results directly to the `pac_coupling` table, instead of only writing CSVs:

```python
result = pac_processor.analyze_pac(
    chan=['E110', 'E111', 'E112'],
    stage=['NREM2', 'NREM3'],
    phase_freq=(0.5, 1.25),
    amp_freq=(11, 16),
    idpac=(1, 2, 4),
    use_detected_events=True,
    event_type='slow_wave',
    pair_with_spindles=True,
    db_path='data/subject001/wonambi/neural_events.db',
    out_dir='data/subject001/wonambi/pac_results',
    write_db=True,
    subject='sub-001',
)
```

If `subject` is omitted, it is derived from the root path basename (with a
warning) — pass it explicitly to avoid an ambiguous subject id in
`pac_coupling`.

!!! warning "`write_db=True` must be able to name what it stores"
    `analyze_pac` only writes to `pac_coupling` when `write_db=True` is
    passed explicitly — there is no implicit database write. On the
    event-locked path above (`use_detected_events=True`), the stored
    `event_type`/`method` are derived automatically from `event_type` /
    `pair_with_spindles` / `event_opts`. If you instead run **continuous**
    PAC (`use_detected_events=False`, e.g. theta-gamma coupling with no
    anchoring event), there is no event scope to derive a label from, so you
    must pass `stored_event_type=` and `stored_method=` yourself (e.g.
    `stored_event_type='continuous'`, `stored_method='theta_gamma'`).
    `analyze_pac` raises `ValueError` at entry — before running any
    analysis — rather than storing a continuous result under a guessed
    label that would be indistinguishable from event-locked coupling.

    The database write itself (`store_pac_to_database`) also now raises on
    failure instead of logging a traceback and returning as if nothing had
    gone wrong — a failed write can no longer look like a successful one that
    happened to store zero rows.

## Export results to CSV

Whether or not you used `write_db=True`, you can export the summary CSV from
the in-memory tracking dict:

```python
method_info = {
    'sw_method': 'Staresina2015',
    'spindle_method': 'Moelle2011',
    'event_type': 'slow_wave',
    'pair_with_spindles': True,
    'stage': ['NREM2', 'NREM3'],
}

export_result = pac_processor.export_pac_parameters_to_csv(
    csv_file='data/subject001/wonambi/pac_results/sw_spindle_coupling_pac_summary.csv',
    phase_freq=(0.5, 1.25),
    amp_freq=(11, 16),
    method_info=method_info,
    out_dir='data/subject001/wonambi/pac_results',
)
print(f"Exported {export_result['channels']} channels, {export_result['rows']} rows")
```

`export_pac_parameters_to_csv` writes per-channel CSVs under
`out_dir/<method_dir>/<stage_str>/`, which is exactly the tree layout
[`backfill_pac_to_db.py`](backfill-pac-to-database.md) expects if you need to
re-import them later.

## Generate a comodulogram

To visualize coupling strength across a grid of phase/amplitude bands for one
channel, use `generate_comodulogram`:

```python
comod_result = pac_processor.generate_comodulogram(
    chan='E110',
    stage=['NREM2', 'NREM3'],
    phase_freqs=[(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)],
    amp_freqs=[(8, 12), (12, 16), (16, 20), (20, 25), (25, 30)],
    idpac=(1, 2, 4),
    out_dir='data/subject001/wonambi/pac_results',
)
if comod_result is not None:
    comod_matrix = comod_result['comod']
```

## Common Issues

### No Events Found for the Requested Method/Stage

`analyze_pac` reads events by `(method, stage)` scope from `neural_events.db`.
Verify the scope exists first:

```python
import sqlite3

conn = sqlite3.connect(db_path)
cursor = conn.execute(
    "SELECT DISTINCT method FROM events WHERE event_type = 'slow_wave'")
print([row[0] for row in cursor.fetchall()])
```

`examples/hdEEG_pac_detector.py --list_methods` and `--stats` do this check
for you from the command line.

### No Channels With Both Slow Waves and Spindles

When pairing slow waves with spindles (`pair_with_spindles=True`), only
channels with **both** event types for the requested methods/stage are usable.
`examples/hdEEG_pac_detector.py` auto-selects the channel intersection when no
explicit channel list is given — reuse that logic
(`get_common_channels` in the script) if you're writing your own driver.

## Next Steps

- [Back-fill PAC Results into the Database](backfill-pac-to-database.md) —
  loading PAC CSVs from an older run
- [Upgrade to 4.0](upgrade-to-4.0.md) — the preferred-phase 180-degree fix
- Reference: [PAC Processor](../reference/api/pacprocessor.md)
