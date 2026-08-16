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

Every write is idempotent, so re-running is safe and this doubles as a repair
tool for a partially-finalized database.

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
Edit the CONFIG block below (set ``ROOT``; optionally list ``SUBJECTS``), then::

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
# CONFIG  --  edit these two lines only
# ===========================================================================

# Root directory containing one folder per subject (each with a wonambi/ subdir).
ROOT = "/Users/tancykao/Library/CloudStorage/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/Emotion"

# Optional explicit subject list (folder names under ROOT). Leave empty to
# auto-discover every subject folder that contains wonambi/neural_events.db.
SUBJECTS = []  # e.g. ["10sd", "11xy"]

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


def main():
    if not os.path.isdir(ROOT):
        print(f"ERROR: ROOT does not exist: {ROOT}")
        return

    subjects = SUBJECTS if SUBJECTS else discover_subjects(ROOT)
    if not subjects:
        print(f"No subjects with wonambi/neural_events.db found under {ROOT}")
        return

    print(f"Backfilling cycles + stage durations for {len(subjects)} "
          f"subject(s) under:\n  {ROOT}\n")

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
            cycles_by_method = finalize_cycles_and_durations(
                annot, db_path, subject=subject)

            summary = ", ".join(
                f"{m}={len(c)} cyc" for m, c in cycles_by_method.items())
            print(f"    PASS: {summary}\n")
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
