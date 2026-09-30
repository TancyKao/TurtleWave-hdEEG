"""
EEGLAB ``.set`` reading for both struct layouts.

EEGLAB can save its dataset struct two ways. The classic layout stores one
MATLAB variable called ``EEG`` whose fields are ``srate``, ``chanlocs``,
``event`` and so on. The other layout (written by ``pop_saveset`` with the
"save as individual variables" option, and used by the Compumedics export
pipeline) stores those fields as top-level variables with no ``EEG`` wrapper.
Wonambi 7.15's ``EEGLAB`` reader only knows the first layout and fails with
``object 'EEG' doesn't exist`` on the second. This module reads both.

The module imports no Qt and is safe to use headless.
"""

import logging
import types
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from numpy import memmap
from scipy.io import loadmat
from wonambi import Dataset as WonambiDataset
from wonambi.ioeeg.eeglab import EEGLAB
from wonambi.ioeeg.utils import (read_hdf5_str, read_hdf5_chan_name,
                                 DEFAULT_DATETIME)

logger = logging.getLogger('turtlewave_hdEEG.eeglab_io')

#: Number of top-level keys named in an :class:`EEGLABFormatError` message.
_KEYS_SHOWN = 12


class EEGLABFormatError(ValueError):
    """A ``.set`` file whose EEGLAB structure TurtleWave cannot read.

    The message is written for the researcher (file base name, MATLAB
    version, what is missing, what was found) and never carries HDF5/h5py
    internals; the original exception, if any, is chained as ``__cause__``.

    Attributes
    ----------
    filename : str
        Base name of the file.
    is_hdf5 : bool or None
        ``True`` for MATLAB 7.3 (HDF5), ``False`` for earlier MATLAB
        versions, ``None`` when the file could not be opened as either.
    missing : list of str
        The required fields that were not found (``srate``, ``chanlocs``).
    top_level_keys : list of str
        Top-level variable names found in the file, excluding HDF5
        bookkeeping (``#refs#``) and scipy's ``__header__``-style entries.
    """

    def __init__(self, message, filename='', is_hdf5=None, missing=None,
                 top_level_keys=None):
        super().__init__(message)
        self.filename = filename
        self.is_hdf5 = is_hdf5
        self.missing = list(missing or [])
        self.top_level_keys = list(top_level_keys or [])


def _format_label(is_hdf5):
    return ('a MATLAB 7.3 (HDF5) EEGLAB file' if is_hdf5
            else 'a MATLAB (pre-7.3) EEGLAB file')


def _no_struct_error(filename, container, is_hdf5):
    """Build the :class:`EEGLABFormatError` for a file with no EEGLAB struct."""
    name = Path(filename).name
    try:
        keys = [str(k) for k in container.keys()
                if not str(k).startswith('#') and not str(k).startswith('__')]
    except AttributeError:
        keys = []
    missing = [k for k in ('srate', 'chanlocs') if k not in keys]
    shown = ', '.join(keys[:_KEYS_SHOWN]) if keys else 'nothing'
    more = f' ({_KEYS_SHOWN} of {len(keys)} shown)' if len(keys) > _KEYS_SHOWN else ''
    message = (f"{name} is {_format_label(is_hdf5)}, but TurtleWave could not "
               f"find the sampling rate (srate) and channel list (chanlocs), "
               f"either at the top level or inside an EEG structure. "
               f"Missing at the top level: {', '.join(missing) or 'none'}. "
               f"Found at the top level: {shown}{more}.")
    return EEGLABFormatError(message, filename=name, is_hdf5=is_hdf5,
                             missing=missing, top_level_keys=keys)


#: Fields a readable EEGLAB struct must carry, with the words used for them.
_REQUIRED_FIELDS = (('srate', 'sampling rate'), ('chanlocs', 'channel list'))


def _struct_fields(eeg):
    """Field names of an EEGLAB struct: h5py group, dict, mat_struct or
    SimpleNamespace."""
    if hasattr(eeg, 'keys'):
        names = list(eeg.keys())
    elif hasattr(eeg, '_fieldnames'):
        names = list(eeg._fieldnames)
    else:
        names = list(vars(eeg))
    return [str(k) for k in names
            if not str(k).startswith('#') and not str(k).startswith('__')]


def _require_fields(filename, eeg, is_hdf5, required=('srate', 'chanlocs')):
    """Raise :class:`EEGLABFormatError` when the located struct lacks a
    required field (a file with an ``EEG`` wrapper that has no ``srate``, say).

    ``top_level_keys`` on the error then lists the struct's own fields.
    """
    fields = _struct_fields(eeg)
    missing = [k for k in required if k not in fields]
    if not missing:
        return
    name = Path(filename).name
    words = ' and '.join(f'{label} ({key})' for key, label in _REQUIRED_FIELDS
                         if key in missing)
    shown = ', '.join(fields[:_KEYS_SHOWN]) if fields else 'nothing'
    more = (f' ({_KEYS_SHOWN} of {len(fields)} shown)'
            if len(fields) > _KEYS_SHOWN else '')
    message = (f"{name} is {_format_label(is_hdf5)}, but TurtleWave could not "
               f"find the {words} inside its EEG structure. "
               f"Missing: {', '.join(missing)}. Found: {shown}{more}.")
    raise EEGLABFormatError(message, filename=name, is_hdf5=is_hdf5,
                            missing=missing, top_level_keys=fields)


def _start_from_T0(T0):
    """``datetime`` from an ``etc.T0`` vector (year, month, day, h, min, s).

    Each element is truncated to ``int``, as Wonambi's HDF5 branch does, so a
    fractional second from MATLAB ``clock`` (e.g. 52.4) no longer raises.
    Returns ``DEFAULT_DATETIME`` when ``T0`` is absent or unusable.
    """
    try:
        return datetime(*[int(float(x)) for x in np.ravel(T0)])
    except (AttributeError, TypeError, ValueError):
        return DEFAULT_DATETIME


def _load_mat(filename, **kwargs):
    """``scipy.io.loadmat`` with unreadable files turned into
    :class:`EEGLABFormatError`. ``NotImplementedError`` (MATLAB 7.3) and
    ``FileNotFoundError`` pass through unchanged."""
    try:
        return loadmat(str(filename), struct_as_record=False, squeeze_me=True,
                       **kwargs)
    except (NotImplementedError, FileNotFoundError):
        raise
    except Exception as err:
        name = Path(filename).name
        reason = str(err).splitlines()[0] if str(err) else type(err).__name__
        raise EEGLABFormatError(
            f"{name} could not be read as a MATLAB file ({reason}).",
            filename=name) from err


def _open_h5(filename):
    """Open a MATLAB 7.3 file, hiding h5py's error text from the message."""
    try:
        return h5py.File(str(filename), 'r')
    except Exception as err:
        name = Path(filename).name
        raise EEGLABFormatError(
            f"{name} looks like a MATLAB 7.3 (HDF5) file but could not be "
            f"opened; it may be incomplete or damaged.",
            filename=name, is_hdf5=True) from err


def find_eeg_struct(container):
    """Locate the EEGLAB dataset struct inside a loaded ``.set`` file.

    Parameters
    ----------
    container : h5py.File, h5py.Group, dict or None
        The opened HDF5 file (MATLAB v7.3) or the dict returned by
        ``scipy.io.loadmat`` (earlier MATLAB versions).

    Returns
    -------
    object or None
        ``container['EEG']`` when the file uses the classic layout; the
        container itself when the EEGLAB fields sit at the top level (it holds
        both ``srate`` and ``chanlocs``); ``None`` when neither is true.
    """
    if container is None:
        return None
    try:
        if 'EEG' in container:
            return container['EEG']
        if 'srate' in container and 'chanlocs' in container:
            return container
    except TypeError:
        return None
    return None


def _as_struct(obj):
    """Give a top-level scipy dict the attribute access of a MATLAB struct.

    Parameters
    ----------
    obj : object
        A scipy ``mat_struct`` (returned unchanged) or the dict ``loadmat``
        returns for a file whose EEGLAB fields are top-level variables.

    Returns
    -------
    object
        ``obj`` itself, or a ``types.SimpleNamespace`` of the dict's
        non-private keys so that ``hasattr(eeg, 'etc')`` and friends work the
        same way for both layouts.
    """
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: v for k, v in obj.items()
                                        if not k.startswith('__')})
    return obj


def dedupe_channel_labels(labels):
    """Make channel labels unique by suffixing repeats with ``_2``, ``_3``...

    Parameters
    ----------
    labels : sequence of str
        Channel labels in file order.

    Returns
    -------
    list of str
        Labels of the same length and order. The first occurrence of a label
        keeps its name; later occurrences become ``<label>_2``, ``<label>_3``
        and so on, skipping any suffix that is already a label in the file.
        Each rename is logged at INFO.

    Notes
    -----
    Wonambi selects channels by name, so two channels called ``ECG`` make the
    second one unreachable and can make ``read_data`` return the wrong trace.
    """
    labels = [str(lab) for lab in labels]
    taken = set(labels)
    seen = set()
    out = []
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
            continue
        k = 2
        while f'{lab}_{k}' in taken:
            k += 1
        new = f'{lab}_{k}'
        taken.add(new)
        seen.add(new)
        out.append(new)
        logger.info(f"Duplicate channel label '{lab}' renamed to '{new}'")
    return out


def _h5_is_empty(obj):
    """True for a MATLAB canonical-empty HDF5 dataset (``[]`` or ``''``)."""
    try:
        return bool(obj.attrs.get('MATLAB_empty', 0))
    except AttributeError:
        return False


def _h5_text(obj):
    """Decode an HDF5 MATLAB char array to ``str`` (``''`` when empty)."""
    if obj is None or _h5_is_empty(obj):
        return ''
    return read_hdf5_str(obj)


def _h5_scalar(obj):
    """First element of an HDF5 numeric dataset as ``float``, else ``None``."""
    if obj is None or _h5_is_empty(obj):
        return None
    try:
        return float(np.ravel(obj[()])[0])
    except (TypeError, ValueError, IndexError):
        return None


def _h5_channel_info(f, eeg):
    """Channel types, reference and good-channel count from an HDF5 struct.

    Parameters
    ----------
    f : h5py.File
        The open file (needed to dereference object references).
    eeg : h5py.Group
        The EEGLAB struct located by :func:`find_eeg_struct`.

    Returns
    -------
    types : list of str or None
        Per-channel ``chanlocs.type`` (``''`` where a channel has none), or
        ``None`` when no channel carries a type.
    ref : str or None
        The struct's ``ref`` field (e.g. ``'average'``), ``None`` if absent.
    n_good : int or None
        ``etc.reference.n_good``, ``None`` if absent.
    """
    chan_types = None
    chanlocs = eeg.get('chanlocs')
    if isinstance(chanlocs, h5py.Group) and 'type' in chanlocs:
        chan_types = []
        for ref in np.ravel(chanlocs['type'][()]):
            try:
                chan_types.append(_h5_text(f[ref]).strip() if ref else '')
            except (TypeError, ValueError, KeyError):
                chan_types.append('')
        if not any(chan_types):
            chan_types = None

    ref = _h5_text(eeg['ref']).strip() if 'ref' in eeg else ''
    ref = ref or None

    n_good = None
    etc = eeg.get('etc')
    if isinstance(etc, h5py.Group) and isinstance(etc.get('reference'), h5py.Group):
        value = _h5_scalar(etc['reference'].get('n_good'))
        if value is not None and np.isfinite(value):
            n_good = int(value)
    return chan_types, ref, n_good


def _scipy_text(value):
    """A scipy-loaded MATLAB char field as ``str`` (``''`` for ``[]``)."""
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.size == 0:
        return ''
    if value is None:
        return ''
    return str(value)


def _scipy_channel_info(eeg):
    """Channel types, reference and good-channel count from a scipy struct.

    Parameters
    ----------
    eeg : object
        The EEGLAB struct (``mat_struct`` or ``SimpleNamespace``).

    Returns
    -------
    tuple
        ``(types, ref, n_good)`` with the same meaning as
        :func:`_h5_channel_info`.
    """
    chan_types = None
    chanlocs = getattr(eeg, 'chanlocs', None)
    if chanlocs is not None:
        chans = np.ravel(np.atleast_1d(chanlocs))
        chan_types = [_scipy_text(getattr(ch, 'type', '')).strip() for ch in chans]
        if not any(chan_types):
            chan_types = None

    ref = _scipy_text(getattr(eeg, 'ref', '')).strip() or None

    n_good = None
    reference = getattr(getattr(eeg, 'etc', None), 'reference', None)
    value = getattr(reference, 'n_good', None)
    try:
        if value is not None and np.size(value) == 1 and np.isfinite(float(value)):
            n_good = int(float(value))
    except (TypeError, ValueError):
        n_good = None
    return chan_types, ref, n_good


class TurtleEEGLAB(EEGLAB):
    """Wonambi ``EEGLAB`` reader that also accepts top-level-field files.

    ``return_hdr`` and ``return_markers`` are re-implemented because Wonambi's
    versions index ``EEG`` inline; ``return_dat`` is inherited unchanged. The
    classic ``EEG``-wrapped layout gives the same header values as Wonambi's
    reader, except that duplicate channel labels are made unique (see
    :func:`dedupe_channel_labels`).

    Attributes
    ----------
    chan_type : list of str or None
        Per-channel ``chanlocs.type`` after :meth:`return_hdr`.
    reference : str or None
        The file's ``ref`` field.
    reference_n_good : int or None
        ``etc.reference.n_good``: how many channels formed the reference.
    """

    def return_hdr(self):
        """Read the header.

        Returns
        -------
        subj_id : str
            Subject identification code.
        start_time : datetime
            Start time of the recording (``etc.T0``, else
            ``etc.rec_startdate``, else Wonambi's default date).
        s_freq : float
            Sampling frequency.
        chan_name : list of str
            Unique channel labels.
        n_samples : int
            Number of samples.
        orig : dict
            Empty, as in Wonambi's reader.

        Raises
        ------
        ValueError
            When the file holds neither an ``EEG`` variable nor top-level
            ``srate`` and ``chanlocs``.
        """
        self.fdtfile = None
        self.chan_type = None
        self.reference = None
        self.reference_n_good = None

        try:
            mat = _load_mat(self.filename)
            self.hdf5 = False
        except NotImplementedError:
            self.hdf5 = True

        if not self.hdf5:
            eeg = find_eeg_struct(mat)
            if eeg is None:
                raise _no_struct_error(self.filename, mat, is_hdf5=False)
            _require_fields(self.filename, eeg, is_hdf5=False)
            self.EEG = _as_struct(eeg)
            self.s_freq = self.EEG.srate
            chan_name = [chan.labels for chan in
                         np.ravel(np.atleast_1d(self.EEG.chanlocs))]
            n_samples = self.EEG.pnts

            if isinstance(getattr(self.EEG, 'subject', None), str):
                subj_id = self.EEG.subject
            else:
                subj_id = ''
            start_time = _start_from_T0(
                getattr(getattr(self.EEG, 'etc', None), 'T0', None))

            if isinstance(getattr(self.EEG, 'datfile', None), str):
                self.fdtfile = self.EEG.datfile
            else:
                self.data = self.EEG.data

            (self.chan_type, self.reference,
             self.reference_n_good) = _scipy_channel_info(self.EEG)

        else:
            with _open_h5(self.filename) as f:
                EEG = find_eeg_struct(f)
                if EEG is None:
                    raise _no_struct_error(self.filename, f, is_hdf5=True)
                _require_fields(self.filename, EEG, is_hdf5=True)
                self.s_freq = EEG['srate'][()].item()
                chan_name = read_hdf5_chan_name(f, EEG['chanlocs']['labels'])
                n_samples = int(np.ravel(EEG['pnts'][()])[0])

                subj_id = _h5_text(EEG['subject']) if 'subject' in EEG else ''
                try:
                    etc = EEG['etc'] if 'etc' in EEG else None
                    if etc is not None and 'T0' in list(etc):
                        try:
                            start_time = datetime(*etc['T0'])
                        except Exception:
                            start_time = datetime(*[int(x[0]) for x in etc['T0']])
                    elif etc is not None and 'rec_startdate' in list(etc):
                        raw = etc['rec_startdate'][()]
                        start_time = datetime.fromisoformat(
                            raw.tobytes().decode('utf-16-le'))
                    else:
                        start_time = DEFAULT_DATETIME
                except ValueError:
                    start_time = DEFAULT_DATETIME

                datfile = _h5_text(EEG['datfile']) if 'datfile' in EEG else ''
                if datfile == '':
                    # MATLAB stores column-major, so the array comes back transposed
                    self.data = EEG['data'][()].T
                else:
                    self.fdtfile = datfile

                (self.chan_type, self.reference,
                 self.reference_n_good) = _h5_channel_info(f, EEG)

        chan_name = dedupe_channel_labels(chan_name)
        if self.chan_type is not None and len(self.chan_type) != len(chan_name):
            logger.warning(f"{len(self.chan_type)} channel types for "
                           f"{len(chan_name)} channels; channel types ignored")
            self.chan_type = None

        if self.fdtfile is not None:
            memshape = (len(chan_name), int(n_samples))
            memmap_file = self.filename.parent / self.fdtfile
            if not memmap_file.exists():
                renamed_memmap_file = self.filename.with_suffix('.fdt')
                if not renamed_memmap_file.exists():
                    fdt_name = Path(self.fdtfile).name
                    also = ('' if renamed_memmap_file.name == fdt_name else
                            f" (also looked for {renamed_memmap_file.name})")
                    raise FileNotFoundError(
                        f"The signal data for {self.filename.name} is stored in "
                        f"a separate file, {fdt_name}, which was not found in "
                        f"the same folder{also}.")
                memmap_file = renamed_memmap_file
            self.data = memmap(str(memmap_file), 'float32', mode='c',
                               shape=memshape, order='F')

        return subj_id, start_time, self.s_freq, chan_name, n_samples, {}

    def return_markers(self):
        """Return the EEGLAB events as Wonambi markers.

        Returns
        -------
        list of dict
            One ``{'name', 'start', 'end'}`` per event, times in seconds from
            the first sample (``(latency - 1) / s_freq``). In the HDF5 branch
            the event type is decoded as text; Wonambi's reader returned the
            first character code instead (``'110'`` for ``'n2'``).
        """
        markers = []
        if self.hdf5:
            with h5py.File(self.filename, 'r') as f:
                EEG = find_eeg_struct(f)
                event = EEG.get('event') if EEG is not None else None
                if not isinstance(event, h5py.Group) or 'type' not in event \
                        or 'latency' not in event:
                    return markers
                for evt, lat in zip(np.ravel(event['type'][()]),
                                    np.ravel(event['latency'][()])):
                    mrk_t = (float(np.ravel(f[lat][()])[0]) - 1) / self.s_freq
                    obj = f[evt]
                    if obj.dtype.kind == 'f':
                        val = float(np.ravel(obj[()])[0])
                        name = str(int(val)) if val.is_integer() else str(val)
                    else:
                        name = _h5_text(obj)
                    markers.append({'name': name, 'start': mrk_t, 'end': mrk_t})
        else:
            events = getattr(self.EEG, 'event', None)
            if events is None:
                return markers
            for event in np.ravel(np.atleast_1d(events)):
                markers.append({
                    'name': str(event.type),
                    'start': (event.latency - 1) / self.s_freq,
                    'end': (event.latency - 1) / self.s_freq,
                })
        return markers


def read_eeglab_channel_info(filename):
    """Read channel labels, channel types and the reference from a ``.set``.

    Only the channel and reference fields are read; the signal is never
    touched.

    Parameters
    ----------
    filename : str or Path
        EEGLAB ``.set`` file, either layout, MATLAB v7.3 or earlier.

    Returns
    -------
    dict
        ``labels`` (list of str, made unique as in :func:`open_dataset`),
        ``types`` (list of str parallel to ``labels``, or ``None`` when the
        file has no channel types), ``ref`` (str or ``None``, e.g.
        ``'average'``) and ``n_good`` (int or ``None``, from
        ``etc.reference.n_good``).

    Raises
    ------
    EEGLABFormatError
        When the file holds no EEGLAB structure or cannot be read.
    """
    filename = str(filename)
    try:
        mat = _load_mat(filename, variable_names=['EEG'])
        eeg = find_eeg_struct(mat)
        if eeg is None:
            mat = _load_mat(filename,
                            variable_names=['srate', 'chanlocs', 'ref', 'etc'])
            eeg = find_eeg_struct(mat)
        if eeg is None:
            raise _no_struct_error(filename, _load_mat(filename), is_hdf5=False)
        _require_fields(filename, eeg, is_hdf5=False, required=('chanlocs',))
        eeg = _as_struct(eeg)
        labels = [chan.labels for chan in np.ravel(np.atleast_1d(eeg.chanlocs))]
        chan_types, ref, n_good = _scipy_channel_info(eeg)
    except NotImplementedError:
        with _open_h5(filename) as f:
            eeg = find_eeg_struct(f)
            if eeg is None:
                raise _no_struct_error(filename, f, is_hdf5=True)
            _require_fields(filename, eeg, is_hdf5=True, required=('chanlocs',))
            labels = read_hdf5_chan_name(f, eeg['chanlocs']['labels'])
            chan_types, ref, n_good = _h5_channel_info(f, eeg)

    labels = dedupe_channel_labels(labels)
    if chan_types is not None and len(chan_types) != len(labels):
        chan_types = None
    return {'labels': labels, 'types': chan_types, 'ref': ref, 'n_good': n_good}


def open_dataset(filename):
    """Open a recording as a Wonambi ``Dataset``, handling both EEGLAB layouts.

    Use this instead of ``wonambi.Dataset(filename)`` everywhere.

    Parameters
    ----------
    filename : str or Path
        Recording path. ``.set`` files are read with :class:`TurtleEEGLAB`;
        every other format goes through Wonambi's own format detection.

    Returns
    -------
    wonambi.Dataset
        The dataset. Its ``header`` gains three keys for every format:
        ``chan_type`` (list of str parallel to ``chan_name``, or ``None`` when
        the file has no channel types; ``''`` for a channel without a type)
        and ``reference`` (``None`` when the file states no reference,
        otherwise ``{'ref': str, 'n_good': int or None}``, e.g.
        ``{'ref': 'average', 'n_good': 220}``, where ``n_good`` is
        ``etc.reference.n_good``, the number of channels that formed it).

    Raises
    ------
    EEGLABFormatError
        A ``.set`` whose EEGLAB structure cannot be found or read.
    FileNotFoundError
        The ``.set`` names a separate ``.fdt`` signal file that is missing;
        the message names both files.
    """
    path = Path(filename)
    if path.suffix.lower() == '.set':
        ds = WonambiDataset(filename, IOClass=TurtleEEGLAB)
        io = ds.dataset
        ds.header['chan_type'] = getattr(io, 'chan_type', None)
        ref = getattr(io, 'reference', None)
        ds.header['reference'] = (
            None if not ref else
            {'ref': ref, 'n_good': getattr(io, 'reference_n_good', None)})
    else:
        ds = WonambiDataset(filename)
        ds.header.setdefault('chan_type', None)
        ds.header.setdefault('reference', None)
    return ds
