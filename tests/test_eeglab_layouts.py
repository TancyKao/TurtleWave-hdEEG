#!/usr/bin/env python3
"""EEGLAB ``.set`` files load through ``open_dataset`` in every layout.

EEGLAB files come in two struct layouts (fields inside one ``EEG`` variable, or
as top-level variables, which is what the Compumedics export writes) and two
MATLAB versions (7.3 = HDF5, or older). Wonambi 7.15 reads only the first
layout; ``turtlewave_hdEEG.eeglab_io`` reads both. Each test below writes the
same small recording in all four combinations with ``tests/eeglab_fixture.py``
and checks that ``open_dataset`` and ``LargeDataset`` see the same thing:

* header (channels, sampling rate, sample count, start time), stages, events;
* the signal, read back through ``read_data``, equals what was written;
* duplicate channel labels are renamed and both channels stay readable;
* ``header['chan_type']`` and ``header['reference']`` shapes, including the
  files that have no channel types or no reference;
* on the ``group`` layout, the header and signal agree with Wonambi's own
  reader (``wonambi.Dataset(path)``);
* failures are named for the researcher, not for h5py.

``test_defect_*`` functions assert behaviour that is required but currently
fails; they are listed at the end of the run.

Run standalone: ``python tests/test_eeglab_layouts.py``. Exits non-zero if any
test fails.
"""

import datetime
import gc
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import eeglab_fixture as fx  # noqa: E402
from turtlewave_hdEEG import open_dataset  # noqa: E402
from turtlewave_hdEEG.dataset import LargeDataset  # noqa: E402
from turtlewave_hdEEG.eeglab_io import (  # noqa: E402
    EEGLABFormatError, TurtleEEGLAB, dedupe_channel_labels, find_eeg_struct,
    read_eeglab_channel_info)

START = datetime.datetime(2022, 9, 12, 22, 28, 52)
EVENTS = [fx.event(1, 'n2', 30 * 100), fx.event(501, 'boundary', 200),
          fx.event(801, 'arousal', 150, 1)]
STAGES = ['W', '1', '2', '3', 'R']


class Workdir:
    """Temporary directory removed on exit.

    Datasets opened inside keep a memory map (``.fdt``) or an HDF5 handle
    (``.set``) for their lifetime, and Windows refuses to rewrite or delete a
    mapped file. So no test writes the same path twice (``_write`` puts every
    recording in a fresh sub-folder), and exit collects garbage before
    deleting.
    """

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix='tw_eeglab_')
        return self.path

    def __exit__(self, *exc):
        gc.collect()
        shutil.rmtree(self.path, ignore_errors=True)


def _write(tmp, layout, v73, name=None, **kwargs):
    # a fresh folder per call: an earlier dataset may still map its .fdt
    folder = tempfile.mkdtemp(dir=tmp)
    path = os.path.join(folder, name or f"rec_{layout}_{'h5' if v73 else 'mat'}.set")
    truth = fx.write_set(path, layout, v73, **kwargs)
    return path, truth


def _each(tmp, **kwargs):
    """Write the same recording in all four layouts; yield (id, path, truth)."""
    for layout, v73 in fx.ALL_LAYOUTS:
        path, truth = _write(tmp, layout, v73, **kwargs)
        yield fx.layout_id(layout, v73), path, truth


def _header_summary(header):
    return (list(header['chan_name']), float(header['s_freq']),
            int(header['n_samples']), header['start_time'])


def test_four_layouts_give_equal_headers():
    """Channels, rate, sample count and start time agree across layouts."""
    print("\n1. All four layout/version combinations, one header:")
    with Workdir() as tmp:
        seen = {}
        for name, path, truth in _each(tmp, stages=STAGES, events=EVENTS):
            via_open = _header_summary(open_dataset(path).header)
            via_large = _header_summary(LargeDataset(path).header)
            assert via_open[:3] == via_large[:3], (name, via_open, via_large)
            seen[name] = via_large
            chans, s_freq, n, start = via_large
            assert chans == ['Fz', 'Cz', 'Pz', 'ECG', 'ECG_2'], (name, chans)
            assert s_freq == 100.0 and n == 1000, (name, s_freq, n)
            assert start == START, (name, start)
            print(f"   [ok] {name}: {len(chans)} channels, {s_freq:g} Hz, "
                  f"{n} samples, start {start}")
        assert len(set(map(repr, seen.values()))) == 1, seen
        print("   [ok] the four headers are identical")


def test_signal_round_trips_through_read_data():
    """``read_data`` returns exactly the float32 values written to the .fdt."""
    print("\n2. Signal round trip through read_data:")
    with Workdir() as tmp:
        for name, path, truth in _each(tmp):
            ds = LargeDataset(path)
            full = ds.read_data()
            assert np.array_equal(full.data[0], truth['data'].astype(np.float64)), name
            win = ds.read_data(begtime=2.0, endtime=5.0)
            assert np.array_equal(win.data[0], truth['data'][:, 200:500].astype(np.float64)), name
            sub = ds.read_data(chan=['Pz', 'Fz'])
            assert list(sub.chan[0]) == ['Pz', 'Fz'] or list(sub.chan[0]) == ['Fz', 'Pz']
            print(f"   [ok] {name}: full, 2-5 s window and channel subset equal the file")


def test_stages_and_events_extracted():
    """``header['stages']`` and ``header['event']`` come out the same everywhere."""
    print("\n3. Stages and events:")
    with Workdir() as tmp:
        for name, path, truth in _each(tmp, stages=STAGES, events=EVENTS):
            header = LargeDataset(path).header
            assert [str(s) for s in header['stages']] == STAGES, (name, header['stages'])
            ev = header['event']
            assert [float(x) for x in ev['onsets']] == [1, 501, 801], (name, ev)
            assert list(ev['types']) == ['n2', 'boundary', 'arousal'], (name, ev)
            assert [float(x) for x in ev['durations']] == [3000, 200, 150], (name, ev)
            assert [float(x) for x in ev['isreject']] == [0, 0, 1], (name, ev)
            print(f"   [ok] {name}: {len(STAGES)} stages, "
                  f"{len(ev['onsets'])} events with latency/type/duration/is_reject")

        # a file with neither is fine, and the keys are simply absent
        for name, path, truth in _each(tmp):
            header = LargeDataset(path).header
            assert 'stages' not in header and 'event' not in header, (name, sorted(header))
        print("   [ok] no etc.stages and no events -> neither key in the header")


def test_markers_are_decoded_text():
    """``return_markers`` gives event names as text and times from sample 1."""
    print("\n4. TurtleEEGLAB.return_markers:")
    events = EVENTS + [fx.event(901, 7, 10)]
    with Workdir() as tmp:
        for name, path, truth in _each(tmp, events=events):
            ds = open_dataset(path)
            markers = ds.dataset.return_markers()
            assert [m['name'] for m in markers] == ['n2', 'boundary', 'arousal', '7'], (name, markers)
            starts = [m['start'] for m in markers]
            assert np.allclose(starts, [0.0, 5.0, 8.0, 9.0]), (name, starts)
            assert all(m['start'] == m['end'] for m in markers), name
            print(f"   [ok] {name}: names {[m['name'] for m in markers]}, starts {starts}")

        # no event field -> no markers, no error
        for name, path, truth in _each(tmp):
            assert open_dataset(path).dataset.return_markers() == [], name
        print("   [ok] no event field -> []")


def test_dedupe_channel_labels():
    """Repeats become ``_2``, ``_3``; existing suffixes are skipped."""
    print("\n5. dedupe_channel_labels:")
    cases = [
        (['ECG', 'ECG'], ['ECG', 'ECG_2']),
        (['ECG', 'ECG', 'ECG'], ['ECG', 'ECG_2', 'ECG_3']),
        (['ECG', 'ECG', 'ECG_2'], ['ECG', 'ECG_3', 'ECG_2']),
        (['BodyPosition', 'Cz', 'BodyPosition'], ['BodyPosition', 'Cz', 'BodyPosition_2']),
        (['A', 'B', 'C'], ['A', 'B', 'C']),
        ([], []),
    ]
    for labels, expected in cases:
        got = dedupe_channel_labels(labels)
        assert got == expected, (labels, got, expected)
        assert len(set(got)) == len(got), got
        print(f"   [ok] {labels} -> {got}")


def test_duplicate_labels_read_different_data():
    """The two ``ECG`` channels are both reachable and return their own trace."""
    print("\n6. Duplicate channel labels in the file:")
    with Workdir() as tmp:
        for name, path, truth in _each(tmp):
            ds = LargeDataset(path)
            assert ds.channels[3:] == ['ECG', 'ECG_2'], (name, ds.channels)
            first = ds.read_data(chan=['ECG']).data[0][0]
            second = ds.read_data(chan=['ECG_2']).data[0][0]
            assert np.array_equal(first, truth['data'][3].astype(float)), name
            assert np.array_equal(second, truth['data'][4].astype(float)), name
            assert not np.array_equal(first, second), name
            print(f"   [ok] {name}: ECG mean {first.mean():.0f}, "
                  f"ECG_2 mean {second.mean():.0f}")

        # Wonambi's own reader keeps both names 'ECG' on the group layout;
        # that is the reason the rename exists
        import wonambi
        path, truth = _write(tmp, 'group', True, name='wonambi_dups.set')
        assert wonambi.Dataset(path).header['chan_name'].count('ECG') == 2
        print("   [ok] (Wonambi alone leaves two channels called 'ECG')")


def test_chan_type_and_reference_shapes():
    """``chan_type`` and ``reference`` for each combination of what the file has."""
    print("\n7. header['chan_type'] and header['reference']:")
    default_types = fx.DEFAULT_TYPES
    with Workdir() as tmp:
        for name, path, truth in _each(tmp):
            header = open_dataset(path).header
            assert header['chan_type'] == default_types, (name, header['chan_type'])
            assert len(header['chan_type']) == len(header['chan_name'])
            assert header['reference'] == {'ref': 'average', 'n_good': 3}, (name, header['reference'])
        print("   [ok] types + reference: parallel list and {'ref', 'n_good'}")

        for name, path, truth in _each(tmp, types=None, ref=None, n_good=None):
            header = open_dataset(path).header
            assert header['chan_type'] is None, (name, header['chan_type'])
            assert header['reference'] is None, (name, header['reference'])
        print("   [ok] no types, no ref, no n_good -> None, None")

        for name, path, truth in _each(tmp, types=[''] * 5):
            assert open_dataset(path).header['chan_type'] is None, name
        print("   [ok] every type empty -> chan_type None")

        mixed = ['EEG', '', 'EEG', 'ECG', '']
        for name, path, truth in _each(tmp, types=mixed):
            assert open_dataset(path).header['chan_type'] == mixed, name
        print("   [ok] some types empty -> '' kept in place")

        for name, path, truth in _each(tmp, n_good=None):
            ref = open_dataset(path).header['reference']
            assert ref == {'ref': 'average', 'n_good': None}, (name, ref)
        print("   [ok] ref without etc.reference.n_good -> n_good None")

        for name, path, truth in _each(tmp, ref=None, n_good=3):
            assert open_dataset(path).header['reference'] is None, name
        print("   [ok] n_good without ref -> reference None")

        # LargeDataset passes the same header through
        for name, path, truth in _each(tmp):
            header = LargeDataset(path).header
            assert header['chan_type'] == default_types and header['reference']['n_good'] == 3, name
        print("   [ok] LargeDataset.header carries both keys")


def test_read_eeglab_channel_info():
    """The light-weight reader agrees with the header and never reads the signal."""
    print("\n8. read_eeglab_channel_info:")
    with Workdir() as tmp:
        for name, path, truth in _each(tmp):
            info = read_eeglab_channel_info(path)
            assert info['labels'] == ['Fz', 'Cz', 'Pz', 'ECG', 'ECG_2'], (name, info)
            assert info['types'] == fx.DEFAULT_TYPES, (name, info)
            assert info['ref'] == 'average' and info['n_good'] == 3, (name, info)
        print("   [ok] labels (deduplicated), types, ref, n_good in all four")

        # works without the .fdt: it only reads the header
        for name, path, truth in _each(tmp, types=None, ref=None, n_good=None):
            os.remove(truth['fdt'])
            info = read_eeglab_channel_info(path)
            assert info['types'] is None and info['ref'] is None and info['n_good'] is None, (name, info)
        print("   [ok] no types/ref, and no .fdt needed")

        for layout, v73 in [('root', True), ('root', False)]:
            path, _ = _write(tmp, layout, v73, name='nosrate.set', omit={'srate'})
            try:
                read_eeglab_channel_info(path)
            except EEGLABFormatError:
                pass
            else:
                raise AssertionError('a file with no srate must raise EEGLABFormatError')
        print("   [ok] top-level file with no srate -> EEGLABFormatError")


def test_find_eeg_struct():
    """Both containers: dict (scipy) and h5py group."""
    print("\n9. find_eeg_struct:")
    inner = {'srate': 1, 'chanlocs': 2}
    assert find_eeg_struct({'EEG': inner}) is inner
    top = {'srate': 1, 'chanlocs': 2, 'x': 3}
    assert find_eeg_struct(top) is top
    assert find_eeg_struct({'srate': 1}) is None
    assert find_eeg_struct({'chanlocs': 1}) is None
    assert find_eeg_struct({}) is None
    assert find_eeg_struct(None) is None
    print("   [ok] dict: EEG wins, top-level needs srate and chanlocs, else None")

    import h5py
    with Workdir() as tmp:
        for layout in ('group', 'root'):
            path, _ = _write(tmp, layout, True, name=f'{layout}.set')
            with h5py.File(path, 'r') as f:
                found = find_eeg_struct(f)
                assert found is not None and 'srate' in found and 'chanlocs' in found, layout
                assert (found.name == '/EEG') == (layout == 'group'), (layout, found.name)
        print("   [ok] h5py: '/EEG' for group layout, the file itself for root layout")


def test_signal_file_lookup():
    """A renamed .set/.fdt pair loads; the .fdt named in the .set is preferred."""
    print("\n10. Where the signal file is looked for:")
    with Workdir() as tmp:
        # stored name differs from the file on disk, which is <set stem>.fdt
        for layout, v73 in fx.ALL_LAYOUTS:
            name = f"renamed_{layout}_{v73}.set"
            path, truth = _write(tmp, layout, v73, name=name,
                                 datfile='original_name.fdt',
                                 fdt_name=os.path.splitext(name)[0] + '.fdt')
            ds = open_dataset(path)
            assert np.array_equal(ds.read_data().data[0], truth['data'].astype(float))
        print("   [ok] datfile names a missing file, <stem>.fdt exists -> loads it")

        # an existing file named in datfile wins over <stem>.fdt
        for layout, v73 in fx.ALL_LAYOUTS:
            name = f"named_{layout}_{v73}.set"
            path, truth = _write(tmp, layout, v73, name=name, datfile='signal_a.fdt')
            decoy = np.zeros((5, 1000), dtype=np.float32)
            decoy.T.tofile(os.path.join(os.path.dirname(path),
                                        os.path.splitext(name)[0] + '.fdt'))
            ds = open_dataset(path)
            assert np.array_equal(ds.read_data().data[0], truth['data'].astype(float)), (layout, v73)
        print("   [ok] datfile names an existing file -> that one, not <stem>.fdt")

        for name, path, truth in _each(tmp):
            os.remove(truth['fdt'])
            try:
                open_dataset(path)
            except FileNotFoundError as err:
                fdt = os.path.basename(truth['fdt'])
                assert fdt in str(err) and os.path.basename(path) in str(err), str(err)
                assert 'h5py' not in str(err), str(err)
            else:
                raise AssertionError(f'{name}: a missing .fdt must raise FileNotFoundError')
            print(f"   [ok] {name}: FileNotFoundError names {fdt} and the .set")

        # both candidate names missing: the second one is named too
        path, truth = _write(tmp, 'root', True, name='gone.set', datfile='old_name.fdt')
        os.remove(truth['fdt'])
        try:
            open_dataset(path)
        except FileNotFoundError as err:
            assert 'old_name.fdt' in str(err) and 'gone.fdt' in str(err), str(err)
            print(f"   [ok] stored name and <stem>.fdt both named: {err}")
        else:
            raise AssertionError('expected FileNotFoundError')


def test_embedded_data_without_fdt():
    """Signal stored inside the .set (no .fdt) loads in all four layouts."""
    print("\n11. Embedded data (no .fdt):")
    with Workdir() as tmp:
        for name, path, truth in _each(tmp, embed_data=True):
            assert truth['fdt'] is None
            ds = LargeDataset(path)
            assert np.array_equal(ds.read_data().data[0], truth['data'].astype(float)), name
            assert ds.header['n_samples'] == 1000, name
            print(f"   [ok] {name}: signal read from the .set itself")
        for name, path, truth in _each(tmp, embed_data=True, empty_datfile=True):
            ds = open_dataset(path)
            assert np.array_equal(ds.read_data().data[0], truth['data'].astype(float)), name
        print("   [ok] same with an explicit empty datfile field")


def test_start_time_sources():
    """etc.T0, else etc.rec_startdate, else Wonambi's default date."""
    print("\n12. Recording start time:")
    from wonambi.ioeeg.utils import DEFAULT_DATETIME
    later = (2023, 1, 2, 3, 4, 5)
    cases = [
        ('T0 and rec_startdate', dict(), START),
        ('T0 only', dict(rec_startdate=None), START),
        ('rec_startdate only', dict(T0=None), START),
        ('T0 differs from rec_startdate', dict(T0=later), None),
        ('neither', dict(T0=None, rec_startdate=None), DEFAULT_DATETIME),
    ]
    with Workdir() as tmp:
        for label, kwargs, expected in cases:
            starts = {}
            for name, path, truth in _each(tmp, **kwargs):
                starts[name] = LargeDataset(path).header['start_time']
            if expected is not None:
                assert all(v == expected for v in starts.values()), (label, starts)
                print(f"   [ok] {label}: {expected} in all four")
            else:
                # the file states two different times; rec_startdate is applied
                # last by LargeDataset in every layout
                assert all(v == START for v in starts.values()), starts
                print(f"   [ok] {label}: rec_startdate ({START}) wins in all four")


def test_group_layout_matches_wonambi_reader():
    """On the classic layout TurtleEEGLAB gives what Wonambi's own reader gives."""
    print("\n13. TurtleEEGLAB vs wonambi.Dataset on the group layout:")
    import wonambi
    from wonambi.ioeeg.utils import DEFAULT_DATETIME
    labels = ['Fz', 'Cz', 'Pz', 'Oz', 'ECG']
    cases = {
        'T0 + rec_startdate': dict(),
        'T0 only': dict(rec_startdate=None),
        'rec_startdate only': dict(T0=None),
        'no start time': dict(T0=None, rec_startdate=None),
        'renamed fdt': dict(datfile='old.fdt', fdt_name='REPLACE'),
    }
    with Workdir() as tmp:
        for v73 in (False, True):
            for label, kwargs in cases.items():
                kwargs = dict(kwargs)
                name = f"cmp_{'h5' if v73 else 'mat'}_{abs(hash(label)) % 10000}.set"
                if kwargs.get('fdt_name') == 'REPLACE':
                    kwargs['fdt_name'] = os.path.splitext(name)[0] + '.fdt'
                path, truth = _write(tmp, 'group', v73, name=name, labels=labels,
                                     types=None, events=EVENTS, stages=STAGES, **kwargs)
                mine = open_dataset(path)
                theirs = wonambi.Dataset(path)
                for key in ('subj_id', 'start_time', 's_freq', 'chan_name', 'n_samples'):
                    assert mine.header[key] == theirs.header[key], (v73, label, key,
                                                                    mine.header[key], theirs.header[key])
                a, b = mine.dataset.data, theirs.dataset.data
                assert (a.dtype, a.shape, a.mode, a.flags.f_contiguous) == \
                       (b.dtype, b.shape, b.mode, b.flags.f_contiguous), (v73, label)
                assert a.filename == b.filename, (a.filename, b.filename)
                assert mine.dataset.fdtfile == theirs.dataset.fdtfile, (v73, label)
                x = mine.read_data(begtime=1, endtime=9).data[0]
                y = theirs.read_data(begtime=1, endtime=9).data[0]
                assert np.array_equal(x, y), (v73, label)
                if not v73:
                    # scipy branch: markers are identical too
                    assert mine.dataset.return_markers() == theirs.dataset.return_markers()
                print(f"   [ok] {'v7.3' if v73 else 'v7  '} {label}: header, memmap "
                      f"(dtype/shape/mode/order/file) and data equal")

        # the default start date really is what Wonambi reports
        path, _ = _write(tmp, 'group', True, name='nostart.set', T0=None, rec_startdate=None)
        assert open_dataset(path).header['start_time'] == DEFAULT_DATETIME

        # Wonambi's HDF5 marker names are the first character code; ours are text
        path, _ = _write(tmp, 'group', True, name='mk.set', events=EVENTS)
        assert wonambi.Dataset(path).dataset.return_markers()[0]['name'] == '110'
        assert open_dataset(path).dataset.return_markers()[0]['name'] == 'n2'
        print("   [ok] HDF5 event names: Wonambi '110', TurtleEEGLAB 'n2' (intended)")


def test_error_missing_srate():
    """A top-level file without ``srate`` raises a readable EEGLABFormatError."""
    print("\n14. EEGLABFormatError for a file with no srate:")
    with Workdir() as tmp:
        for v73 in (True, False):
            path, _ = _write(tmp, 'root', v73, name='no_srate.set', omit={'srate'})
            for opener in (open_dataset, LargeDataset):
                try:
                    opener(path)
                except EEGLABFormatError as err:
                    assert err.filename == 'no_srate.set', err.filename
                    assert err.is_hdf5 is v73, (err.is_hdf5, v73)
                    assert err.missing == ['srate'], err.missing
                    assert 'chanlocs' in err.top_level_keys and 'srate' not in err.top_level_keys
                    assert not any(k.startswith(('#', '__')) for k in err.top_level_keys)
                    text = str(err)
                    assert 'no_srate.set' in text and 'srate' in text, text
                    assert ('HDF5' in text) == v73, text
                    assert 'Unable to' not in text and 'h5py' not in text, text
                    assert isinstance(err, ValueError)
                    seen = err
                else:
                    raise AssertionError('expected EEGLABFormatError')
            print(f"   [ok] {'v7.3' if v73 else 'v7  '}: filename, is_hdf5, "
                  f"missing={seen.missing}, keys={seen.top_level_keys}")

        # no chanlocs, and no EEGLAB struct at all
        path, _ = _write(tmp, 'root', True, name='no_chan.set', omit={'chanlocs'})
        try:
            open_dataset(path)
        except EEGLABFormatError as err:
            assert err.missing == ['chanlocs'], err.missing
        else:
            raise AssertionError('expected EEGLABFormatError')

        import scipy.io
        other = os.path.join(tmp, 'other.set')
        scipy.io.savemat(other, {'foo': np.arange(3), 'bar': 'x'})
        try:
            open_dataset(other)
        except EEGLABFormatError as err:
            assert err.missing == ['srate', 'chanlocs'] and err.top_level_keys == ['foo', 'bar'], (
                err.missing, err.top_level_keys)
            assert err.is_hdf5 is False
        else:
            raise AssertionError('expected EEGLABFormatError')
        print("   [ok] no chanlocs -> missing ['chanlocs']; unrelated MAT-file -> both missing")

        # many keys are truncated in the message but complete on the attribute
        many = {f'var{i:02d}': np.arange(2) for i in range(20)}
        other = os.path.join(tmp, 'other_many.set')
        scipy.io.savemat(other, many)
        try:
            open_dataset(other)
        except EEGLABFormatError as err:
            assert len(err.top_level_keys) == 20 and 'var19' not in str(err) and '12 of 20' in str(err)
        print("   [ok] 20 top-level keys: message shows 12, attribute has all 20")


def test_error_unreadable_files():
    """Garbage and truncated files raise EEGLABFormatError, not library errors."""
    print("\n15. EEGLABFormatError for unreadable files:")
    with Workdir() as tmp:
        junk = os.path.join(tmp, 'junk.set')
        with open(junk, 'wb') as fh:
            fh.write(os.urandom(600))
        try:
            open_dataset(junk)
        except EEGLABFormatError as err:
            assert err.filename == 'junk.set' and 'junk.set' in str(err)
            print(f"   [ok] random bytes: {str(err)[:90]}")
        else:
            raise AssertionError('expected EEGLABFormatError')

        # valid MATLAB 7.3 header, no HDF5 behind it
        broken = os.path.join(tmp, 'broken73.set')
        with open(broken, 'wb') as fh:
            fh.write(fx._matlab_header() + os.urandom(200))
        try:
            open_dataset(broken)
        except EEGLABFormatError as err:
            assert err.is_hdf5 is True, err.is_hdf5
            assert 'broken73.set' in str(err) and 'Unable to' not in str(err), str(err)
            print(f"   [ok] MATLAB 7.3 header, no HDF5: {str(err)[:90]}")
        else:
            raise AssertionError('expected EEGLABFormatError')


def test_non_set_file_still_opens():
    """An EDF goes through ``open_dataset`` to Wonambi's own reader."""
    print("\n16. Non-.set file (EDF):")
    from wonambi.ioeeg import write_edf
    from wonambi.utils.simulate import create_data

    with Workdir() as tmp:
        s_freq = 128.0
        data = create_data(datatype='ChanTime', n_trial=1, s_freq=s_freq,
                           chan_name=['Cz'], time=(0, 20.0))
        n = len(data.axis['time'][0])
        t = np.arange(n) / s_freq
        data.data[0] = np.asarray(20.0 * np.sin(2 * np.pi * 1.0 * t), dtype='f')[None, :]
        edf = os.path.join(tmp, 'sub-A.edf')
        write_edf(data, edf)

        ds = open_dataset(edf)
        assert ds.header['chan_name'] == ['Cz'] and ds.header['s_freq'] == s_freq
        assert ds.header['chan_type'] is None and ds.header['reference'] is None
        assert not isinstance(ds.dataset, TurtleEEGLAB)
        assert ds.read_data(begtime=0, endtime=2).data[0].shape == (1, 256)
        print("   [ok] EDF opens, chan_type None, reference None, read_data works")

        large = LargeDataset(edf, extract_eeglab_metadata=False)
        assert large.header['chan_type'] is None and large.channels == ['Cz']
        print("   [ok] LargeDataset on an EDF")

        # a .SET in capitals is still an EEGLAB file
        path, truth = _write(tmp, 'root', True, name='UPPER.SET')
        assert isinstance(open_dataset(path).dataset, TurtleEEGLAB)
        print("   [ok] '.SET' suffix is matched case-insensitively")


# ------------------------------------------------------------ known defects

def test_defect_fractional_second_T0_in_pre_v73_file():
    """A pre-7.3 file whose ``etc.T0`` has fractional seconds must still open.

    MATLAB ``clock`` returns fractional seconds, and MATLAB stores the array as
    doubles, so scipy hands ``datetime(*T0)`` numpy floats and it raises
    ``TypeError``. TurtleEEGLAB's scipy branch catches only ``AttributeError``
    (as Wonambi does); its HDF5 branch converts with ``int(...)`` and works.
    """
    print("\n17. DEFECT CHECK: fractional-second etc.T0, all four layouts:")
    with Workdir() as tmp:
        failures = []
        for name, path, truth in _each(tmp, T0=(2022, 9, 12, 22, 28, 52.4),
                                       rec_startdate=None):
            try:
                start = open_dataset(path).header['start_time']
                assert start.replace(microsecond=0) == START, (name, start)
                print(f"   [ok] {name}: {start}")
            except Exception as err:
                failures.append(f"{name}: {type(err).__name__}: {err}")
                print(f"   [FAIL] {name}: {type(err).__name__}: {err}")
        assert not failures, failures


def test_defect_wrapped_layout_missing_required_field():
    """An ``EEG`` variable without ``srate`` should raise EEGLABFormatError too.

    Today the file with the ``EEG`` wrapper gets past ``find_eeg_struct`` and
    then fails on the missing field with ``AttributeError`` (older MATLAB) or
    ``KeyError: "Unable to synchronously open object (object 'srate' doesn't
    exist)"`` (MATLAB 7.3), the h5py wording the error class exists to hide.
    """
    print("\n18. DEFECT CHECK: EEG-wrapped file with no srate / no chanlocs:")
    with Workdir() as tmp:
        failures = []
        for v73 in (False, True):
            for omitted in ('srate', 'chanlocs'):
                path, _ = _write(tmp, 'group', v73, name='wrapped_missing.set', omit={omitted})
                label = f"{'v7.3' if v73 else 'v7'} missing {omitted}"
                try:
                    open_dataset(path)
                except EEGLABFormatError as err:
                    assert omitted in err.missing, (label, err.missing)
                    print(f"   [ok] {label}")
                except Exception as err:
                    failures.append(f"{label}: {type(err).__name__}: {err}")
                    print(f"   [FAIL] {label}: {type(err).__name__}: {err}")
        assert not failures, failures


TESTS = [
    test_four_layouts_give_equal_headers,
    test_signal_round_trips_through_read_data,
    test_stages_and_events_extracted,
    test_markers_are_decoded_text,
    test_dedupe_channel_labels,
    test_duplicate_labels_read_different_data,
    test_chan_type_and_reference_shapes,
    test_read_eeglab_channel_info,
    test_find_eeg_struct,
    test_signal_file_lookup,
    test_embedded_data_without_fdt,
    test_start_time_sources,
    test_group_layout_matches_wonambi_reader,
    test_error_missing_srate,
    test_error_unreadable_files,
    test_non_set_file_still_opens,
    test_defect_fractional_second_T0_in_pre_v73_file,
    test_defect_wrapped_layout_missing_required_field,
]


if __name__ == "__main__":
    print("TESTING EEGLAB layouts through open_dataset")
    print("===========================================")
    failed = []
    for test in TESTS:
        try:
            test()
        except Exception:
            failed.append(test.__name__)
            print(f"   [FAIL] {test.__name__}")
            traceback.print_exc()
    print()
    if failed:
        print(f"FAILED {len(failed)} of {len(TESTS)}: {', '.join(failed)}")
        sys.exit(1)
    print("All EEGLAB layout tests passed.")
