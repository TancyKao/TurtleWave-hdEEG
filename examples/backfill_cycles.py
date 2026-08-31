"""Backfill sleep cycles and stage durations into existing event databases.

This is the *post-detection finalize* step, run as a batch backfill. Use it when
you already have one or more ``neural_events.db`` files full of detected events
(slow waves, spindles, K-complexes) but have **not** yet populated the derived
hypnogram tables. For every subject it calls
:func:`turtlewave_hdEEG.finalize_cycles_and_durations`, which:

1. detects sleep cycles for both the ``'2022'`` and ``'1979'`` definitions and
   stores them in ``sleep_cycles``;
2. writes per-stage minutes (Wake / N1 / N2 / N3 / REM / artefact) to
   ``stage_durations``;
3. tags every ``events.cycle`` with its cycle number using the ``'2022'``
   definition; and
4. writes ``'2022'`` cycle markers back into the annotation XML.

Every write is idempotent -- re-running never duplicates rows -- so this
doubles as a repair tool for a partially-finalized database. Note that
"idempotent" here means *replace*, not *accumulate*; see "Re-running replaces
the stored cycles" below before you change a threshold.

Cycle thresholds
----------------
Three CONFIG lines set the cycle definition and are passed straight through to
:func:`turtlewave_hdEEG.finalize_cycles_and_durations`:

``EPOCH_LENGTH``
    Epoch length of the hypnogram, in seconds. This is a property of how the
    recording was scored, **not** a tunable: set it to whatever the annotation
    XML actually uses (library default 30). A wrong value is silently
    destructive -- the cycle boundaries themselves come from the epoch grid and
    stay correct, but every minute column (``nrem_dur_min``, ``rem_dur_min``,
    ``cycle_dur_min``, and all of ``stage_durations``) is rescaled by the ratio
    of the true epoch length to this one, with no error raised. ``main()``
    checks it against the annotation's own epoch grid and skips the subject on
    a mismatch.
``WAKE_THRESH_MIN``
    Longest Wake bout, in minutes, that is absorbed into a surrounding NREM
    period instead of breaking it. The bound is **inclusive**: a Wake bout of
    up to and including this many minutes is absorbed; a longer one breaks the
    NREM period. The library default is ``wake_thresh=10`` epochs, i.e. 5
    minutes at a 30 s epoch; this script ships 15 minutes, which is more
    permissive and therefore yields fewer, longer cycles.
``NREM_MIN_MIN``
    How long an NREM run must be to count as an NREM period, in minutes. The
    bound is **exclusive**: a run must be strictly LONGER than this many
    minutes to count, so a run of exactly this length is dropped. The library
    default is ``nrem_min=30`` epochs, i.e. 15 minutes at a 30 s epoch.

Minutes are converted to epochs with ``int(round(minutes * 60 / EPOCH_LENGTH))``
and the resulting epoch counts are printed in the run header.

A fourth CONFIG line, ``PLOT``, is not a threshold. It is True by default: each
subject also gets ``{subject}_hypnogram_cycles_wake{WAKE_THRESH_MIN}_nrem{NREM_MIN_MIN}min.png``
beside its database (cycle bands over the hypnogram, one row per method), and
the path is printed. Set ``PLOT = False`` to turn plotting off. It needs
matplotlib and renders headless. Both thresholds are in the filename, so
re-running with a different ``WAKE_THRESH_MIN`` or ``NREM_MIN_MIN`` leaves the
earlier PNG in place next to the new one rather than overwriting it.

Re-running replaces the stored cycles
-------------------------------------
Re-running this script for a subject **replaces** that subject's ``sleep_cycles``
rows, its ``events.cycle`` tags and the cycle markers in its annotation XML with
the ones implied by the thresholds above. Nothing is appended and nothing from a
previous threshold survives.

The database records no threshold anywhere, so a ``neural_events.db`` only ever
holds the most recently computed cycle definition and there is no way to tell
from the file which ``WAKE_THRESH_MIN`` produced it. If you are comparing
threshold variants, run the export (``examples/export_cycle_events.py``) after
each backfill into an output folder named for the threshold -- e.g.
``cycle_event_exports_wake15`` -- before re-running with a different value.

Layout assumed
--------------
A ``ROOT`` directory with one folder per subject; each subject folder has a
``wonambi/`` subdirectory holding ``neural_events.db`` and the Wonambi
annotation XML (``sub-*.xml``)::

    ROOT/
      10sd/
        wonambi/
          neural_events.db
          sub-10sd_ses-1_task-psg_run-1_desc-inspect_eeg.xml
      11xy/
        wonambi/
          ...

Usage
-----
Edit the CONFIG block below -- set ``ROOT``, optionally list ``SUBJECTS``, and
set the cycle thresholds ``EPOCH_LENGTH`` / ``WAKE_THRESH_MIN`` /
``NREM_MIN_MIN`` -- then::

    python examples/backfill_cycles.py

The script prints one PASS/FAIL line per subject and a final tally. One subject
failing never aborts the whole run.
"""

import glob
import os
import traceback

from turtlewave_hdEEG import CustomAnnotations, finalize_cycles_and_durations
from turtlewave_hdEEG.utils import derive_subject as _derive_subject

# ===========================================================================
# CONFIG  --  edit the paths and the cycle thresholds below
# ===========================================================================

# Root directory containing one folder per subject (each with a wonambi/ subdir).
ROOT = "/Users/tancykao/Library/CloudStorage/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/Emotion"

# Optional explicit subject list (folder names under ROOT). Leave empty to
# auto-discover every subject folder that contains wonambi/neural_events.db.
SUBJECTS = []  # e.g. ["10sd", "11xy"]

# Epoch length of the hypnogram, seconds. A property of the recording's scoring,
# not a tunable -- match the annotation XML. A wrong value leaves the cycle
# boundaries correct but silently rescales every minute column; main() checks it
# against the annotation's epoch grid and skips the subject if they disagree.
EPOCH_LENGTH = 30

# Wake bouts of up to and INCLUDING this many MINUTES inside NREM are absorbed
# into the NREM period instead of breaking it; a longer bout breaks it (library
# default is 5 min = 10 epochs).
WAKE_THRESH_MIN = 15

# An NREM run must be strictly LONGER than this many minutes to count as an NREM
# period -- a run of exactly this length is dropped (library default 15 min).
NREM_MIN_MIN = 15

# Plotting is ON by default: each database also gets a
# {subject}_hypnogram_cycles_wake{WAKE_THRESH_MIN}_nrem{NREM_MIN_MIN}min.png
# beside it (blue NREM / red REM bands over the hypnogram, one row per method).
# Set to False to skip it. Needs matplotlib; headless-safe. Both thresholds are
# in the filename, so re-running with a different WAKE_THRESH_MIN or
# NREM_MIN_MIN writes a differently named PNG alongside the old one instead of
# overwriting it.
PLOT = True

# ===========================================================================
# End of CONFIG
# ===========================================================================

def discover_subjects(root):
    """Return subject folder names under ``root`` that have a detection DB.

    Parameters
    ----------
    root : str
        Directory containing one folder per subject.

    Returns
    -------
    list of str
        Sorted folder names that contain ``wonambi/neural_events.db``.
    """
    found = []
    for name in sorted(os.listdir(root)):
        subj_dir = os.path.join(root, name)
        if not os.path.isdir(subj_dir):
            continue
        if os.path.isfile(os.path.join(subj_dir, "wonambi", "neural_events.db")):
            found.append(name)
    return found


def resolve_paths(subj_dir):
    """Locate the database and annotation XML inside a subject folder.

    Parameters
    ----------
    subj_dir : str
        Path to one subject folder (its ``wonambi/`` subdir holds the files).

    Returns
    -------
    db_path : str
        Path to ``neural_events.db``.
    xml_path : str
        Path to the chosen ``sub-*.xml`` annotation file.

    Raises
    ------
    FileNotFoundError
        If the database or no annotation XML is present.
    """
    wonambi_dir = os.path.join(subj_dir, "wonambi")
    db_path = os.path.join(wonambi_dir, "neural_events.db")
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"no neural_events.db in {wonambi_dir}")

    xmls = sorted(glob.glob(os.path.join(wonambi_dir, "sub-*.xml")))
    if not xmls:
        raise FileNotFoundError(f"no sub-*.xml annotation file in {wonambi_dir}")

    # Prefer an XML whose name matches the subject folder; else take the first
    # and warn so an unexpected multi-file folder is never silently guessed.
    folder = os.path.basename(subj_dir.rstrip(os.sep))
    chosen = xmls[0]
    for candidate in xmls:
        if folder in os.path.basename(candidate):
            chosen = candidate
            break
    if len(xmls) > 1:
        print(f"    WARN: {len(xmls)} annotation XMLs found; using "
              f"{os.path.basename(chosen)}")
    return db_path, chosen


def derive_subject(subj_dir, xml_path):
    """Derive the subject id (``sub-XXXX``) from the XML stem or folder name.

    Thin positional wrapper around
    :func:`turtlewave_hdEEG.utils.derive_subject` so this script and the
    library share one implementation of subject resolution.

    Parameters
    ----------
    subj_dir : str
        Subject folder path (fallback source of the id).
    xml_path : str
        Annotation XML path (preferred source of the id).

    Returns
    -------
    str
        Subject identifier, e.g. ``"sub-10sd"``.
    """
    return _derive_subject(annotation_path=xml_path, root_dir=subj_dir)


def observed_epoch_length(annot):
    """Return the epoch length implied by an annotation's own epoch grid.

    Measured within the first epoch (``end - start``) rather than across the
    first two starts, so a grid with a gap in it -- scoring that skips a
    stretch of the recording -- reports the true epoch length instead of the
    size of the gap. This is the same grid ``ParalCycles`` reads (see
    ``_epoch_starts`` in ``turtlewave_hdEEG/cycleprocessor.py``).

    Parameters
    ----------
    annot : CustomAnnotations
        Loaded annotation wrapper.

    Returns
    -------
    float or None
        Length of the first epoch in seconds, or ``None`` when the grid is
        empty or unreadable (nothing to compare, so the caller should proceed
        rather than skip).
    """
    try:
        epochs = annot.epochs
    except Exception:
        return None
    if not epochs:
        return None
    try:
        return float(epochs[0]["end"]) - float(epochs[0]["start"])
    except (KeyError, TypeError, ValueError):
        return None


def main():
    if not os.path.isdir(ROOT):
        print(f"ERROR: ROOT does not exist: {ROOT}")
        return

    subjects = SUBJECTS if SUBJECTS else discover_subjects(ROOT)
    if not subjects:
        print(f"No subjects with wonambi/neural_events.db found under {ROOT}")
        return

    # Thresholds are configured in minutes for readability; the library takes
    # epoch counts.
    wake_thresh_ep = int(round(WAKE_THRESH_MIN * 60 / EPOCH_LENGTH))
    nrem_min_ep = int(round(NREM_MIN_MIN * 60 / EPOCH_LENGTH))

    print(f"Backfilling cycles + stage durations for {len(subjects)} "
          f"subject(s) under:\n  {ROOT}")
    print(f"  epoch length    : {EPOCH_LENGTH} s")
    print(f"  wake threshold  : {WAKE_THRESH_MIN} min ({wake_thresh_ep} epochs)")
    print(f"  min NREM period : {NREM_MIN_MIN} min ({nrem_min_ep} epochs)")
    print("  (re-run replaces any cycles already stored for these subjects)\n")

    n_pass = 0
    n_fail = 0
    for folder in subjects:
        subj_dir = os.path.join(ROOT, folder)
        try:
            db_path, xml_path = resolve_paths(subj_dir)
            subject = derive_subject(subj_dir, xml_path)
            print(f"[{folder}] subject={subject}")
            print(f"    db  : {db_path}")
            print(f"    xml : {os.path.basename(xml_path)}")

            annot = CustomAnnotations(xml_path)

            # EPOCH_LENGTH is a property of the scoring, not a tunable. A wrong
            # value never raises: boundaries come from the epoch grid and stay
            # correct while every minute column is silently rescaled. Compare
            # against the annotation's own grid and skip rather than write
            # wrong durations.
            observed = observed_epoch_length(annot)
            if observed is not None and abs(observed - EPOCH_LENGTH) > 1e-6:
                print(f"    WARN: annotation epoch grid is {observed:g} s but "
                      f"EPOCH_LENGTH is {EPOCH_LENGTH:g} s; skipping this "
                      f"subject. Set EPOCH_LENGTH to {observed:g} and re-run.")
                n_fail += 1
                continue

            # Name the PNG for the wake and NREM thresholds: the database
            # records neither, so this is the only place the cycle definition
            # that produced a plot is written down. A different
            # WAKE_THRESH_MIN or NREM_MIN_MIN lands beside the old PNG rather
            # than on top of it.
            plot_path = os.path.join(
                os.path.dirname(db_path),
                f"{subject}_hypnogram_cycles_wake{WAKE_THRESH_MIN:g}"
                f"_nrem{NREM_MIN_MIN:g}min.png")

            cycles_by_method = finalize_cycles_and_durations(
                annot, db_path, subject=subject,
                epoch_length=EPOCH_LENGTH,
                wake_thresh=wake_thresh_ep,
                nrem_min=nrem_min_ep,
                plot=PLOT, plot_path=plot_path)

            summary = ", ".join(
                f"{m}={len(c)} cyc" for m, c in cycles_by_method.items())
            print(f"    PASS: {summary}")
            if PLOT:
                print(f"    plot: {plot_path}")
            print()
            n_pass += 1
        except Exception as exc:  # noqa: BLE001 - one bad subject must not abort
            print(f"    FAIL: {exc}")
            print("    " + traceback.format_exc().replace("\n", "\n    "))
            n_fail += 1

    print("=" * 60)
    print(f"Done. {n_pass} passed, {n_fail} failed, "
          f"{len(subjects)} total.")


if __name__ == "__main__":
    main()
