"""
hdEEG_cycle_detector.py

Post-detection finalize step for TurtleWave-hdEEG.

Run this ONCE after event detection (spindles / slow waves / K-complexes have
already been detected into ``neural_events.db``). A single call to
``finalize_cycles_and_durations`` fills the database with everything that is
derived from the hypnogram:

1. ``sleep_cycles`` — per-cycle NREM/REM boundaries and durations for BOTH
   definitions (``'2022'`` and ``'1979'``), keyed by ``(subject, method)``.
2. ``stage_durations`` — per-subject minutes in Wake / N1 / N2 / N3 / REM /
   artefact (reconciling to the hypnogram span).
3. ``events.cycle`` — every detected event tagged with its cycle number using
   the canonical ``'2022'`` definition.
4. cycle markers in the annotation XML (so the review GUI and
   ``Annotations.get_cycles()`` show cycle bands), written for ``'2022'`` only.

All writes are idempotent, so re-running is safe. It also works as a backfill
for a database you already detected events into — just point ``db_path`` at the
existing ``neural_events.db``.

The hypnogram/cycle PNG is written on demand only (see the optional block at the
bottom); the default finalize call does not plot.
"""

import os

from turtlewave_hdEEG import (CustomAnnotations,
                              finalize_cycles_and_durations,
                              plot_from_annotations)

# 1. File paths (same layout as the spindle / slow-wave example scripts).
root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/01js/ses-1/"
annotfilename = "sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.xml"
subject = "sub-001js"

annot_file = os.path.join(root_dir, "wonambi", annotfilename)
db_path = os.path.join(root_dir, "wonambi", "neural_events.db")

# 2. Load the annotations (CustomAnnotations exposes get_hypnogram / epochs).
print("Loading annotations...")
annot = CustomAnnotations(annot_file)

# 3. Finalize: detect cycles for both methods, write stage durations, tag
#    events.cycle with the 2022 definition, and write 2022 XML markers. DB only
#    (no plot). Returns {method: [cycle dicts]}.
print("Finalizing sleep cycles + stage durations...")
cycles_by_method = finalize_cycles_and_durations(
    annot,
    db_path,
    subject=subject,
    # Defaults mirror the MATLAB detector; override if needed:
    # methods=('2022', '1979'),  # cycle definitions to store
    # tag_method='2022',         # which definition owns events.cycle + XML
    # wake_thresh=10,            # max Wake epochs absorbed into NREM
    # nrem_min=30,               # min NREM epochs to count as an NREM period
    # rem_min=10,                # min REM epochs to close a cycle (1979 only)
)

# 4. Report.
for method, cycles in cycles_by_method.items():
    print(f"\nMethod '{method}': {len(cycles)} cycle(s):")
    for c in cycles:
        print(f"  cycle {c['cycle_number']}: "
              f"NREM {c['nrem_dur_min']:.1f} min, "
              f"REM {c['rem_dur_min']:.1f} min, "
              f"total {c['cycle_dur_min']:.1f} min "
              f"(start {c['nrem_start_sec']:.0f}s)")

# ---------------------------------------------------------------------------
# 5. OPTIONAL: write the hypnogram/cycle PNG on demand.
#    The finalize step above never plots; uncomment this block when you want the
#    figure (both method rows, blue NREM / red REM bars over the hypnogram).
# ---------------------------------------------------------------------------
# out_png = os.path.join(
#     root_dir, "wonambi",
#     f"{subject}_hypnogram_cycles_2022_vs_1979.png")
# plot_from_annotations(annot, cycles_by_method, out_png, subject=subject)
# print(f"\nWrote cycle plot to {out_png}")
