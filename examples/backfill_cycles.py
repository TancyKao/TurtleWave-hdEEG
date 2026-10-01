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
the path is printed. Pass ``--no-plot`` (or set ``PLOT = False``) to turn plotting off. It needs
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

Every subject's annotation XML is rewritten
-------------------------------------------
This script leaves ``write_xml`` at its library default (True), so each subject's
annotation XML is rewritten on every run -- **including a subject whose run finds
zero cycles**. Marker writing is a replacement: the existing ``'2022'`` cycle
markers are cleared first, and Wonambi saves the file on that clear, stamping the
rater's ``modified`` attribute with the current time. A zero-cycle night therefore
shows a changed XML with a fresh ``modified`` timestamp even though the only
content change is the *removal* of the previous run's markers, which is the
correct outcome (the XML, ``sleep_cycles`` and ``events.cycle`` must agree).

Two consequences worth knowing before you run this over a shared folder: file
timestamps cannot tell you whether cycles were written or cleared -- read the
``sleep_cycles`` table or ``Annotations.get_cycles()`` instead -- and if the XMLs
are under version control, expect a one-line ``modified`` diff for every subject
processed. Nothing else in the file is touched.

Layout assumed
--------------
A root directory (``--root``) with one folder per subject; each subject folder has a
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
The root folder is a required argument; there is no built-in default, so
running the script (or ``--help``) never touches a database by accident::

    python examples/backfill_cycles.py --root /path/to/Emotion --dry-run
    python examples/backfill_cycles.py --root /path/to/Emotion --subjects 10sd 11xy

``--dry-run`` lists the databases and annotation XMLs that would be modified
and exits. Without it the same list is printed and the script asks for
confirmation (``--yes`` skips the question; without a terminal and without
``--yes`` it stops). The CONFIG block below holds the defaults for the cycle
thresholds; ``--epoch-length``, ``--wake-thresh-min``, ``--nrem-min-min`` and
``--no-plot`` override them.

The script prints one PASS/FAIL line per subject and a final tally. One subject
failing never aborts the whole run. A subject whose database write succeeded but
whose PNG could not be drawn still counts as a PASS, with an extra ``WARN:
database written, plot skipped`` line carrying the plot error -- the derived
tables are the deliverable, the plot is a convenience.
"""

import argparse
import glob
import os
import sys
import traceback

from turtlewave_hdEEG import CustomAnnotations, finalize_cycles_and_durations
from turtlewave_hdEEG.utils import derive_subject as _derive_subject

# ===========================================================================
# CONFIG  --  edit the paths and the cycle thresholds below
# ===========================================================================

# The root folder and the subject list are command-line arguments (--root,
# --subjects); there is deliberately no default root.

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


def plot_cycles(annot, cycles_by_method, plot_path, subject,
                epoch_length=EPOCH_LENGTH):
    """Draw the hypnogram/cycle PNG, returning the error instead of raising.

    Kept out of :func:`turtlewave_hdEEG.finalize_cycles_and_durations` (which
    is called with ``plot=False``) on purpose: that function plots *after* it
    has written ``sleep_cycles``, ``stage_durations`` and ``events.cycle``, so
    an exception from the plotting step would propagate out of a call whose
    database work had already succeeded and the batch loop would count the
    subject as FAIL. Running the plot here, separately and after the return,
    keeps a missing matplotlib, an unwritable output directory or a headless
    backend problem from misreporting a finished backfill as a failure.

    Parameters
    ----------
    annot : CustomAnnotations
        Loaded annotation wrapper (source of the hypnogram and epoch grid).
    cycles_by_method : dict
        ``{method: [cycle dicts]}`` as returned by
        :func:`turtlewave_hdEEG.finalize_cycles_and_durations`.
    plot_path : str
        Destination PNG path.
    subject : str
        Subject label for the figure title.
    epoch_length : float, optional
        Epoch length in seconds. Default :data:`EPOCH_LENGTH`.

    Returns
    -------
    Exception or None
        ``None`` when the PNG was written; otherwise the exception raised,
        for the caller to report as a warning.
    """
    try:
        # Imported lazily so a matplotlib-free environment only fails here,
        # as a warning, rather than at module import.
        from turtlewave_hdEEG.cycleplot import plot_from_annotations
        plot_from_annotations(annot, cycles_by_method, plot_path,
                              epoch_length=epoch_length, subject=subject)
        return None
    except Exception as exc:  # noqa: BLE001 - a plot must not fail a backfill
        return exc


def parse_args(argv=None):
    """Command-line arguments.

    Parameters
    ----------
    argv : list of str or None
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
    """
    ap = argparse.ArgumentParser(
        description="Backfill sleep cycles and stage durations into existing "
                    "neural_events.db files (rewrites sleep_cycles, "
                    "stage_durations, events.cycle and the XML cycle markers).")
    ap.add_argument('--root', required=True,
                    help="folder with one subfolder per subject, each holding "
                         "wonambi/neural_events.db and a sub-*.xml")
    ap.add_argument('--subjects', nargs='*', default=None,
                    help="subject folder names under --root (default: every "
                         "folder with wonambi/neural_events.db)")
    ap.add_argument('--epoch-length', type=float, default=EPOCH_LENGTH)
    ap.add_argument('--wake-thresh-min', type=float, default=WAKE_THRESH_MIN)
    ap.add_argument('--nrem-min-min', type=float, default=NREM_MIN_MIN)
    ap.add_argument('--no-plot', action='store_true', help="skip the PNGs")
    ap.add_argument('--dry-run', action='store_true',
                    help="list what would be modified and exit")
    ap.add_argument('--yes', action='store_true',
                    help="do not ask for confirmation")
    return ap.parse_args(argv)


def confirm(targets, assume_yes=False):
    """Print the files that will be modified and ask before writing.

    Parameters
    ----------
    targets : list of (str, str, str)
        ``(folder, db_path, xml_path)`` per subject.
    assume_yes : bool
        Skip the question.

    Returns
    -------
    bool
        True to proceed.
    """
    print("These databases and annotation XMLs will be modified "
          "(sleep_cycles, stage_durations, events.cycle, XML cycle markers):")
    for folder, db_path, xml_path in targets:
        print(f"  [{folder}] {db_path}")
        print(f"  {' ' * (len(folder) + 2)} {xml_path}")
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("No terminal to confirm on; pass --yes to proceed.")
        return False
    return input("Proceed? [y/N] ").strip().lower() in ('y', 'yes')


def main(argv=None):
    """Backfill every subject under ``--root``, one PASS/FAIL line each.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments (see :func:`parse_args`).

    Two behaviours worth knowing, both deliberate:

    * Each subject's annotation XML is rewritten on every run, including a
      subject that yields zero cycles -- the markers are cleared before new
      ones are written and Wonambi saves the file on the clear, so the rater's
      ``modified`` timestamp is refreshed and the file changes on disk even
      when the only change is the removal of a previous run's markers.
    * A failure to draw the PNG is reported as ``WARN: database written, plot
      skipped`` and the subject still counts as a PASS; only a failure before
      or during the database write counts as FAIL. The final tally names how
      many passing subjects had their plot skipped.
    """
    args = parse_args(argv)
    root = args.root
    epoch_length = args.epoch_length
    if not os.path.isdir(root):
        print(f"ERROR: --root does not exist: {root}")
        return

    subjects = args.subjects if args.subjects else discover_subjects(root)
    if not subjects:
        print(f"No subjects with wonambi/neural_events.db found under {root}")
        return

    targets = []
    for folder in subjects:
        try:
            db_path, xml_path = resolve_paths(os.path.join(root, folder))
            targets.append((folder, db_path, xml_path))
        except FileNotFoundError as exc:
            print(f"[{folder}] skipped: {exc}")
    if not targets:
        return
    if args.dry_run:
        confirm(targets, assume_yes=True)
        print("Dry run: nothing was modified.")
        return
    if not confirm(targets, assume_yes=args.yes):
        print("Aborted: nothing was modified.")
        return
    subjects = [t[0] for t in targets]

    # Thresholds are configured in minutes for readability; the library takes
    # epoch counts.
    wake_min, nrem_min_min = args.wake_thresh_min, args.nrem_min_min
    plot = PLOT and not args.no_plot
    wake_thresh_ep = int(round(wake_min * 60 / epoch_length))
    nrem_min_ep = int(round(nrem_min_min * 60 / epoch_length))

    print(f"Backfilling cycles + stage durations for {len(subjects)} "
          f"subject(s) under:\n  {root}")
    print(f"  epoch length    : {epoch_length} s")
    print(f"  wake threshold  : {wake_min} min ({wake_thresh_ep} epochs)")
    print(f"  min NREM period : {nrem_min_min} min ({nrem_min_ep} epochs)")
    print("  (re-run replaces any cycles already stored for these subjects)\n")

    n_pass = 0
    n_fail = 0
    n_plot_skipped = 0
    for folder in subjects:
        subj_dir = os.path.join(root, folder)
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
            if observed is not None and abs(observed - epoch_length) > 1e-6:
                print(f"    WARN: annotation epoch grid is {observed:g} s but "
                      f"--epoch-length is {epoch_length:g} s; skipping this "
                      f"subject. Pass --epoch-length {observed:g} and re-run.")
                n_fail += 1
                continue

            # Name the PNG for the wake and NREM thresholds: the database
            # records neither, so this is the only place the cycle definition
            # that produced a plot is written down. A different
            # WAKE_THRESH_MIN or NREM_MIN_MIN lands beside the old PNG rather
            # than on top of it.
            plot_path = os.path.join(
                os.path.dirname(db_path),
                f"{subject}_hypnogram_cycles_wake{wake_min:g}"
                f"_nrem{nrem_min_min:g}min.png")

            # plot=False: the PNG is drawn below, outside this call, so that a
            # plotting failure cannot make an already-written database read as
            # a FAIL. Everything this call does -- sleep_cycles,
            # stage_durations, events.cycle, the XML markers -- is done when it
            # returns. It also rewrites the XML (and its `modified` timestamp)
            # even for a subject with zero cycles; see the module docstring.
            cycles_by_method = finalize_cycles_and_durations(
                annot, db_path, subject=subject,
                epoch_length=epoch_length,
                wake_thresh=wake_thresh_ep,
                nrem_min=nrem_min_ep,
                plot=False)

            # --- database and XML are written from this point on ---
            plot_error = None
            if plot:
                plot_error = plot_cycles(annot, cycles_by_method, plot_path,
                                         subject, epoch_length=epoch_length)

            summary = ", ".join(
                f"{m}={len(c)} cyc" for m, c in cycles_by_method.items())
            print(f"    PASS: {summary}")
            if plot and plot_error is None:
                print(f"    plot: {plot_path}")
            elif plot:
                # A PASS with a warning: the derived tables are the
                # deliverable and they are in the database; only the PNG is
                # missing, and re-running the script redraws it.
                print(f"    WARN: database written, plot skipped "
                      f"({type(plot_error).__name__}: {plot_error})")
                n_plot_skipped += 1
            print()
            n_pass += 1
        except Exception as exc:  # noqa: BLE001 - one bad subject must not abort
            # Reaches here only for a failure at or before the database write
            # (bad paths, unscorable hypnogram, SQLite error). Plot failures
            # are handled above as a warning and never land here.
            print(f"    FAIL: {exc}")
            print("    " + traceback.format_exc().replace("\n", "\n    "))
            n_fail += 1

    print("=" * 60)
    print(f"Done. {n_pass} passed, {n_fail} failed, "
          f"{len(subjects)} total.")
    if n_plot_skipped:
        print(f"      {n_plot_skipped} of the {n_pass} passing subject(s) "
              f"had the database written but the plot skipped (see the WARN "
              f"lines above); re-run to redraw them.")


if __name__ == "__main__":
    main()
