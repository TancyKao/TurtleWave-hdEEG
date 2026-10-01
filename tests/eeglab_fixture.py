"""Small EEGLAB ``.set`` writer for the loader and staging tests.

Not a test itself. ``write_set`` writes a few-channel recording in any of the
four layout/version combinations the library has to read:

============  ==========================================================
layout        where the EEGLAB fields (``srate``, ``chanlocs``, ...) sit
============  ==========================================================
``'group'``   inside one variable called ``EEG`` (the classic layout, and
              the only one Wonambi 7.15 reads)
``'root'``    as top-level variables with no ``EEG`` wrapper (the
              Compumedics export layout)
============  ==========================================================

``v73=True`` writes a MATLAB 7.3 (HDF5) file with h5py, copying the shapes of
a real Compumedics export: scalars are ``(1, 1)`` float64, strings are
``(n, 1)`` uint16 arrays, cell fields (channel labels, event fields, stages)
are ``(n, 1)`` or ``(1, n)`` arrays of object references into ``#refs#``.
The file starts with the 512-byte MATLAB userblock and version bytes
``0x0200 'IM'`` at offset 124. Without those scipy raises ``ValueError``
instead of ``NotImplementedError`` and the HDF5 branch is never reached.
``v73=False`` writes an older MAT-file with ``scipy.io.savemat``.

The signal goes to a float32 column-major ``.fdt`` beside the ``.set`` (named
by ``datfile``), or into the file itself with ``embed_data=True``.
"""

import os
import struct

import h5py
import numpy as np
from scipy.io import savemat

DEFAULT_LABELS = ['Fz', 'Cz', 'Pz', 'ECG', 'ECG']
DEFAULT_TYPES = ['EEG', 'EEG', 'EEG', 'ECG', 'ECG']
DEFAULT_T0 = (2022, 9, 12, 22, 28, 52)
DEFAULT_REC_STARTDATE = '2022-09-12T22:28:52'

USERBLOCK = 512

#: Every combination the readers have to handle, as (layout, v73).
ALL_LAYOUTS = [('group', False), ('group', True), ('root', False), ('root', True)]


def layout_id(layout, v73):
    """Short name for a layout/version pair, for messages."""
    return f"{layout}/{'v7.3' if v73 else 'v7'}"


def make_data(n_chan, n_samples, seed=0):
    """Distinct float32 data per channel, so a wrong-channel read is obvious.

    Parameters
    ----------
    n_chan, n_samples : int
        Shape of the result, ``(n_chan, n_samples)``.
    seed : int
        Random seed.

    Returns
    -------
    ndarray of float32
        Standard-normal noise plus ``100 * channel_index``.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_chan, n_samples)) + 100.0 * np.arange(n_chan)[:, None]
    return data.astype(np.float32)


def event(latency, type, duration=0, is_reject=0):
    """One EEGLAB event record (latency and duration in samples, 1-based)."""
    return {'latency': latency, 'type': type, 'duration': duration,
            'is_reject': is_reject}


def write_set(path, layout='root', v73=True, n_samples=1000, srate=100.0,
              labels=None, types='default', ref='average', n_good=3,
              stages=None, T0=DEFAULT_T0, rec_startdate=DEFAULT_REC_STARTDATE,
              events=None, embed_data=False, data=None, seed=0,
              subject='sub-test', omit=(), datfile=None, fdt_name=None,
              empty_datfile=False, interp_channels=None):
    """Write a small EEGLAB ``.set`` (and ``.fdt``) and return what it holds.

    Parameters
    ----------
    path : str or Path
        The ``.set`` file. The ``.fdt`` is written beside it with the same stem.
    layout : {'root', 'group'}
        Top-level fields (``'root'``) or one ``EEG`` variable (``'group'``).
    v73 : bool
        MATLAB 7.3 (HDF5, written with h5py) or older (scipy ``savemat``).
    n_samples : int
        Samples per channel.
    srate : float
        Sampling rate in Hz.
    labels : list of str or None
        Channel labels; duplicates allowed. Default ``DEFAULT_LABELS``.
    types : list of str, None or 'default'
        ``chanlocs.type`` per channel; ``None`` leaves the field out of the
        file; ``'default'`` uses ``DEFAULT_TYPES`` when the labels are the
        default ones.
    ref : str or None
        The ``ref`` field; ``None`` leaves it out.
    n_good : int or None
        ``etc.reference.n_good``; ``None`` leaves ``etc.reference`` out.
    stages : list of str or None
        ``etc.stages``; ``None`` leaves it out.
    T0 : tuple or None
        ``etc.T0`` (year, month, day, hour, minute, second); ``None`` omits.
    rec_startdate : str or None
        ``etc.rec_startdate``; ``None`` omits.
    events : list of dict or None
        Records from :func:`event`; ``None`` leaves ``event`` out.
    embed_data : bool
        Store the signal in the ``.set`` itself and write no ``.fdt``.
    data : ndarray or None
        Signal, ``(n_chan, n_samples)``; default :func:`make_data`.
    seed : int
        Seed for the default data.
    subject : str
        ``subject`` field.
    omit : iterable of str
        Fields to leave out of the file: ``'srate'``, ``'chanlocs'``,
        ``'pnts'``.
    datfile : str or None
        The signal file name stored in the ``.set``; default the ``.set``'s
        stem plus ``.fdt``.
    fdt_name : str or None
        Name the ``.fdt`` is actually written under (in the ``.set``'s folder);
        default the stored ``datfile``. Set both, differently, to mimic a
        ``.set``/``.fdt`` pair that was renamed together.
    empty_datfile : bool
        With ``embed_data``, also store ``datfile = ''`` (some writers do).
    interp_channels : list or None
        ``etc.interp_channels``, written as an ``(n, 1)`` cell as the
        Compumedics cleaning pipeline does; items are normally str (a number
        is stored as a numeric cell element). ``[]`` writes MATLAB's empty
        cell; ``None`` (default) leaves the field out.

    Returns
    -------
    dict
        ``path``, ``fdt`` (path or None), ``data`` (float32, as stored),
        ``labels`` (as written, duplicates included), ``types``,
        ``srate``, ``n_samples``.
    """
    path = str(path)
    labels = list(DEFAULT_LABELS if labels is None else labels)
    n_chan = len(labels)
    if types == 'default':
        types = list(DEFAULT_TYPES) if len(labels) == len(DEFAULT_TYPES) else None
    if data is None:
        data = make_data(n_chan, n_samples, seed)
    data = np.asarray(data, dtype=np.float32)
    assert data.shape == (n_chan, n_samples), data.shape
    omit = set(omit)

    fdt = None
    stored_name = ''
    if not embed_data:
        stem = os.path.splitext(os.path.basename(path))[0]
        stored_name = datfile or (stem + '.fdt')
        fdt = os.path.join(os.path.dirname(path), fdt_name or stored_name)
        # column-major: channels vary fastest, as EEGLAB writes it
        data.T.astype(np.float32).tofile(fdt)

    spec = dict(layout=layout, srate=srate, n_samples=n_samples, labels=labels,
                types=types, ref=ref, n_good=n_good, stages=stages, T0=T0,
                rec_startdate=rec_startdate, events=events,
                embed_data=embed_data, data=data, fdt_name=stored_name,
                empty_datfile=empty_datfile, interp_channels=interp_channels,
                subject=subject, omit=omit)
    if v73:
        _write_v73(path, spec)
    else:
        _write_v7(path, spec)
    return {'path': path, 'fdt': fdt, 'data': data, 'labels': labels,
            'types': types, 'srate': srate, 'n_samples': n_samples}


# ---------------------------------------------------------------- MATLAB 7.3

def _matlab_header():
    text = ('MATLAB 7.3 MAT-file, Platform: PCWIN64, Created on: '
            'Fri Sep 18 19:27:23 2026 HDF5 schema 1.00 .')
    block = bytearray(text.encode('ascii').ljust(116, b' '))
    block += b'\x00' * 8                       # subsystem data offset
    block += struct.pack('<H', 0x0200) + b'IM'  # version 0x0200, endian 'IM'
    block += b'\x00' * (USERBLOCK - len(block))
    return bytes(block)


class _H5Writer:
    """h5py helpers that reproduce MATLAB's storage of scalars, strings, cells."""

    def __init__(self, f):
        self.f = f
        self.refs = f.require_group('#refs#')
        self._n = 0

    def _store(self, group, name, arr, matlab_class):
        ds = group.create_dataset(name, data=arr)
        ds.attrs['MATLAB_class'] = np.bytes_(matlab_class)
        return ds

    def scalar(self, group, name, value):
        return self._store(group, name, np.array([[float(value)]], dtype=np.float64),
                           'double')

    def string(self, group, name, text):
        if text == '':
            ds = group.create_dataset(name, data=np.array([0, 0], dtype=np.uint64))
            ds.attrs['MATLAB_class'] = np.bytes_('char')
            ds.attrs['MATLAB_empty'] = np.uint32(1)
            return ds
        arr = np.array([ord(c) for c in text], dtype=np.uint16).reshape(-1, 1)
        ds = self._store(group, name, arr, 'char')
        ds.attrs['MATLAB_int_decode'] = np.int32(2)
        return ds

    def vector(self, group, name, values, column=True):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr.reshape(-1, 1) if column else arr.reshape(1, -1)
        return self._store(group, name, arr, 'double')

    def _ref_to_value(self, value):
        """A dataset in ``#refs#`` holding one cell element; returns its ref."""
        name = f'e{self._n}'
        self._n += 1
        if isinstance(value, str):
            if value == '':
                ds = self.refs.create_dataset(name, data=np.array([0, 0], dtype=np.uint64))
                ds.attrs['MATLAB_class'] = np.bytes_('char')
                ds.attrs['MATLAB_empty'] = np.uint32(1)
            else:
                arr = np.array([ord(c) for c in value], dtype=np.uint16).reshape(-1, 1)
                ds = self.refs.create_dataset(name, data=arr)
                ds.attrs['MATLAB_class'] = np.bytes_('char')
                ds.attrs['MATLAB_int_decode'] = np.int32(2)
        else:
            ds = self.refs.create_dataset(
                name, data=np.array([[float(value)]], dtype=np.float64))
            ds.attrs['MATLAB_class'] = np.bytes_('double')
        return ds.ref

    def cell(self, group, name, values, column=True):
        refs = [self._ref_to_value(v) for v in values]
        shape = (len(refs), 1) if column else (1, len(refs))
        ds = group.create_dataset(name, shape=shape, dtype=h5py.ref_dtype)
        ds[...] = np.array(refs, dtype=object).reshape(shape)
        ds.attrs['MATLAB_class'] = np.bytes_('cell')
        return ds

    def empty_cell(self, group, name):
        ds = group.create_dataset(name, data=np.array([0, 0], dtype=np.uint64))
        ds.attrs['MATLAB_class'] = np.bytes_('cell')
        ds.attrs['MATLAB_empty'] = np.uint32(1)
        return ds

    def struct(self, parent, name):
        g = parent.create_group(name)
        g.attrs['MATLAB_class'] = np.bytes_('struct')
        return g


def _write_v73(path, s):
    with h5py.File(path, 'w', userblock_size=USERBLOCK) as f:
        w = _H5Writer(f)
        top = f if s['layout'] == 'root' else w.struct(f, 'EEG')
        _write_fields_h5(w, top, s)
    with open(path, 'r+b') as fh:
        fh.write(_matlab_header())


def _write_fields_h5(w, top, s):
    omit = s['omit']
    if 'srate' not in omit:
        w.scalar(top, 'srate', s['srate'])
    if 'pnts' not in omit:
        w.scalar(top, 'pnts', s['n_samples'])
    w.scalar(top, 'nbchan', len(s['labels']))
    w.string(top, 'subject', s['subject'])
    if s['ref'] is not None:
        w.string(top, 'ref', s['ref'])

    if 'chanlocs' not in omit:
        cl = w.struct(top, 'chanlocs')
        w.cell(cl, 'labels', s['labels'])
        if s['types'] is not None:
            w.cell(cl, 'type', s['types'])

    if s['embed_data']:
        # MATLAB is column-major, so h5py sees the array transposed
        ds = top.create_dataset('data', data=s['data'].T)
        ds.attrs['MATLAB_class'] = np.bytes_('single')
        w.string(top, 'datfile', '')
    else:
        w.string(top, 'data', s['fdt_name'])
        w.string(top, 'datfile', s['fdt_name'])

    etc = w.struct(top, 'etc')
    if s['T0'] is not None:
        w.vector(etc, 'T0', s['T0'])
    if s['rec_startdate'] is not None:
        w.string(etc, 'rec_startdate', s['rec_startdate'])
    if s['stages'] is not None:
        w.cell(etc, 'stages', s['stages'], column=False)
    if s['n_good'] is not None:
        reference = w.struct(etc, 'reference')
        w.scalar(reference, 'n_good', s['n_good'])
        w.string(reference, 'method', 'average')
    if s['interp_channels'] is not None:
        if s['interp_channels']:
            w.cell(etc, 'interp_channels', s['interp_channels'], column=True)
        else:
            w.empty_cell(etc, 'interp_channels')

    if s['events'] is not None:
        ev = w.struct(top, 'event')
        w.cell(ev, 'latency', [e['latency'] for e in s['events']])
        w.cell(ev, 'type', [e['type'] for e in s['events']])
        w.cell(ev, 'duration', [e['duration'] for e in s['events']])
        w.cell(ev, 'is_reject', [e['is_reject'] for e in s['events']])


# --------------------------------------------------------------- older MAT

def _mat_num(value):
    """A number as MATLAB stores it: integer-valued doubles are written in an
    integer type, which is why scipy hands ``pnts`` and ``srate`` back as
    ``int`` for real EEGLAB files (Wonambi slices with ``pnts``, so a float
    would fail there)."""
    arr = np.asarray(value, dtype=np.float64)
    if np.all(arr == np.round(arr)):
        return arr.astype(np.int32) if arr.ndim else int(arr)
    return arr if arr.ndim else float(arr)


def _struct_array(records, fields):
    """A 1 x n MATLAB struct array from a list of dicts."""
    arr = np.empty((1, len(records)), dtype=[(f, 'O') for f in fields])
    for i, rec in enumerate(records):
        for f in fields:
            arr[f][0, i] = rec[f]
    return arr


def _write_v7(path, s):
    omit = s['omit']
    eeg = {}
    if 'srate' not in omit:
        eeg['srate'] = _mat_num(s['srate'])
    if 'pnts' not in omit:
        eeg['pnts'] = _mat_num(s['n_samples'])
    eeg['nbchan'] = _mat_num(len(s['labels']))
    eeg['subject'] = s['subject']
    if s['ref'] is not None:
        eeg['ref'] = s['ref']

    if 'chanlocs' not in omit:
        fields = ['labels'] + (['type'] if s['types'] is not None else [])
        recs = []
        for i, lab in enumerate(s['labels']):
            rec = {'labels': lab}
            if s['types'] is not None:
                rec['type'] = s['types'][i]
            recs.append(rec)
        eeg['chanlocs'] = _struct_array(recs, fields)

    if s['embed_data']:
        eeg['data'] = s['data'].astype(np.float64)
        if s['empty_datfile']:
            eeg['datfile'] = ''
    else:
        eeg['data'] = s['fdt_name']
        eeg['datfile'] = s['fdt_name']

    etc = {}
    if s['T0'] is not None:
        etc['T0'] = _mat_num(list(s['T0']))
    if s['rec_startdate'] is not None:
        etc['rec_startdate'] = s['rec_startdate']
    if s['stages'] is not None:
        cell = np.empty((1, len(s['stages'])), dtype=object)
        for i, code in enumerate(s['stages']):
            cell[0, i] = code
        etc['stages'] = cell
    if s['n_good'] is not None:
        etc['reference'] = {'n_good': _mat_num(s['n_good']), 'method': 'average'}
    if s['interp_channels'] is not None:
        names = list(s['interp_channels'])
        cell = np.empty((len(names), 1) if names else (0, 0), dtype=object)
        for i, name in enumerate(names):
            cell[i, 0] = name if isinstance(name, str) else _mat_num(name)
        etc['interp_channels'] = cell
    eeg['etc'] = etc

    if s['events'] is not None:
        recs = [{'latency': _mat_num(e['latency']),
                 'type': e['type'] if isinstance(e['type'], str) else _mat_num(e['type']),
                 'duration': _mat_num(e['duration']),
                 'is_reject': _mat_num(e['is_reject'])} for e in s['events']]
        eeg['event'] = _struct_array(recs, ['latency', 'type', 'duration', 'is_reject'])

    payload = {'EEG': eeg} if s['layout'] == 'group' else eeg
    savemat(path, payload)
