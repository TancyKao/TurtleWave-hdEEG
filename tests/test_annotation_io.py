#!/usr/bin/env python3
"""``XLAnnotations`` must save XML, and must report a missing staging import.

Two defects in ``turtlewave_hdEEG/annotation.py``:

* ``save()`` called ``self.annotations.export(filename)``. Wonambi 7.15's
  ``Annotations.export`` defaults to ``xformat='csv'`` and writes a four-column
  epoch/stage CSV to whatever path it is handed, so ``save()`` with its default
  ``annot_file`` OVERWROTE the annotation XML with a CSV -- destroying every
  event and rater and leaving a file ``Annotations()`` can no longer parse.
  The method name, its docstring and its log line all said "XML".
* ``process_all()`` computed ``add_stages_from_header()``'s result and then
  returned a hardcoded ``True``, so a non-GUI caller could not tell that the
  XML had been written WITHOUT sleep stages -- after which every stage-filtered
  detection finds nothing, with no error anywhere.

Run standalone: ``python tests/test_annotation_io.py``.
"""

import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turtlewave_hdEEG.annotation import XLAnnotations  # noqa: E402


def _edf_dataset(tmp, duration=120.0, s_freq=128.0):
    """Write a small EDF and open it as a ``wonambi.Dataset``.

    Parameters
    ----------
    tmp : str
        Directory to write into.
    duration : float, optional
        Recording length in seconds. Default ``120.0``.
    s_freq : float, optional
        Sampling frequency in Hz. Default ``128.0``.

    Returns
    -------
    instance of wonambi.Dataset
        A single-channel dataset whose header carries neither ``event`` nor
        ``stages``.
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

    edf = os.path.join(tmp, 'sub-A.edf')
    write_edf(data, edf)
    return Dataset(edf)


def test_save_writes_xml_not_csv():
    """``save()`` must leave a parseable annotation XML, not a stage CSV.

    Checked on the file itself (it parses as XML and the rater and event
    survive a round trip through ``Annotations``), and against the exact
    artefact the old call produced, so the regression cannot come back
    disguised.
    """
    print("\n1. XLAnnotations.save() writes Wonambi XML:")

    from wonambi.attr import Annotations

    tmp = tempfile.mkdtemp(prefix='tw_annot_io_')
    try:
        dataset = _edf_dataset(tmp)
        annot_file = os.path.join(tmp, 'sub-A.xml')
        xl = XLAnnotations(dataset, annot_file, rater_name='tester')
        assert xl.add_annotation('Artefact', 10.0, 12.0), \
            "could not add the event the round trip is asserted on"

        assert xl.save(), "save() reported failure"

        with open(annot_file) as f:
            head = f.read(200)
        assert head.lstrip().startswith('<?xml') or head.lstrip().startswith('<'), (
            f"save() did not write XML; the file starts with {head[:60]!r}")
        assert not head.startswith('Wonambi v'), (
            "save() wrote Annotations.export()'s CSV header -- the XML has "
            "been overwritten with a stage CSV")

        root = ET.parse(annot_file).getroot()
        assert root.tag == 'annotations', f"root tag is {root.tag!r}"

        # The real contract: wonambi can read it back, with rater and event.
        reread = Annotations(annot_file)
        assert 'tester' in reread.raters, (
            f"the rater did not survive save(): {reread.raters}")
        events = reread.get_events(name='Artefact')
        assert len(events) == 1, (
            f"the event did not survive save(): {events}")
        assert abs(float(events[0]['start']) - 10.0) < 1e-6, events[0]
        print(f"   [ok] {os.path.basename(annot_file)} parses as XML "
              f"(root={root.tag!r}), rater 'tester' and 1 Artefact event "
              f"survive a reload")

        # What the old implementation did, shown on a scratch path so the
        # annotation file is not harmed: export() writes CSV.
        csv_path = os.path.join(tmp, 'what_export_writes.csv')
        xl.annotations.export(csv_path)
        with open(csv_path) as f:
            csv_head = f.readline().strip()
        assert csv_head.startswith('Wonambi v'), csv_head
        print(f"   [ok] Annotations.export() -- the old call -- writes CSV: "
              f"first line {csv_head!r}")

        # save(other_path) is a copy: valid XML there, and the object still
        # points at its own file afterwards.
        copy_path = os.path.join(tmp, 'copy.xml')
        assert xl.save(copy_path), "save(filename) reported failure"
        assert ET.parse(copy_path).getroot().tag == 'annotations'
        assert str(xl.annotations.xml_file) == str(annot_file), (
            f"save(filename) left the object pointing at "
            f"{xl.annotations.xml_file}, so every later write would be "
            f"redirected")
        assert len(Annotations(annot_file).get_events(name='Artefact')) == 1, (
            "the original file changed when saving a copy")
        print("   [ok] save(other_path) writes a valid XML copy and leaves "
              "the object pointing at its original file")

        # CustomAnnotations.save() is the same method name on the sibling
        # class. It already called the XML writer, but it IGNORED its
        # filename argument while printing "Annotations saved to <filename>"
        # -- reporting a file it had not written.
        from turtlewave_hdEEG.annotation import CustomAnnotations

        custom = CustomAnnotations(annot_file)
        custom_copy = os.path.join(tmp, 'custom_copy.xml')
        assert custom.save(custom_copy), "CustomAnnotations.save() failed"
        assert os.path.exists(custom_copy), (
            "CustomAnnotations.save(filename) reported success for a file it "
            "never wrote")
        assert ET.parse(custom_copy).getroot().tag == 'annotations'
        assert str(custom.wonb_annot.xml_file) == str(annot_file), (
            f"CustomAnnotations.save(filename) left the object pointing at "
            f"{custom.wonb_annot.xml_file}")
        print("   [ok] CustomAnnotations.save(other_path) writes the file it "
              "reports, and keeps its own path")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_process_all_reports_missing_staging():
    """``process_all()`` returns the staging outcome, not a hardcoded True.

    A recording whose header carries no staging must come back ``False``; one
    whose header does carry staging must come back ``True`` and leave scored
    epochs in the XML.
    """
    print("\n2. XLAnnotations.process_all() reports the staging outcome:")

    from wonambi.attr import Annotations

    tmp = tempfile.mkdtemp(prefix='tw_annot_stage_')
    try:
        # (a) no 'stages' in the header at all.
        dataset = _edf_dataset(tmp)
        assert 'stages' not in dataset.header, (
            "this EDF unexpectedly carries staging, so the negative case is "
            "not being exercised")
        annot_file = os.path.join(tmp, 'nostage.xml')
        xl = XLAnnotations(dataset, annot_file, rater_name='tester')
        result = xl.process_all()
        assert result is False, (
            f"process_all() returned {result!r} for a header with no staging; "
            f"a caller cannot tell the XML was written without stages")
        print(f"   [ok] header without staging -> process_all() is {result!r}")

        # (b) staging present: Compumedics codes, one per 30 s epoch.
        dataset2 = _edf_dataset(tmp)
        dataset2.header['stages'] = [0, 2, 2, 3]  # 120 s / 30 s = 4 epochs
        annot_file2 = os.path.join(tmp, 'staged.xml')
        xl2 = XLAnnotations(dataset2, annot_file2, rater_name='tester')
        result2 = xl2.process_all()
        assert result2 is True, (
            f"process_all() returned {result2!r} for a header WITH staging")

        scored = [e['stage'] for e in Annotations(annot_file2).epochs]
        assert scored, "no epochs were written despite a True result"
        assert any(s not in ('Unknown', 'Undefined') for s in scored), (
            f"every epoch is unscored despite a True result: {set(scored)}")
        print(f"   [ok] header with staging -> process_all() is {result2!r}, "
              f"{len(scored)} epochs scored {sorted(set(scored))}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TESTING XLAnnotations SAVE / PROCESS_ALL")
    print("=======================================")

    test_save_writes_xml_not_csv()
    test_process_all_reports_missing_staging()

    print("\nAll annotation I/O tests passed.")
