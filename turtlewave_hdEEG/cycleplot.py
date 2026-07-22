"""
cycleplot.py

Headless-safe hypnogram + sleep-cycle plotting for TurtleWave-hdEEG.

This module renders a night-level figure: the hypnogram (Wake / REM / N1 / N2 /
N3 rows, Wake at top) with one cycle-bar row per detection method drawn above it
(blue rectangles for each NREM period, red rectangles for the following
REM/inter-NREM segment, labelled ``NREM1``, ``REM1``, ``NREM2`` ...). The x-axis
is in hours.

It is deliberately import-safe in headless environments (no ``$DISPLAY``, no
PyQt): it never imports ``matplotlib.pyplot``. Figures are built with the
object-oriented API and rasterised through the Agg canvas::

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

so importing ``turtlewave_hdEEG.cycleplot`` on a compute node without a display
cannot fail on backend selection.
"""

import logging

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle


logger = logging.getLogger('turtlewave_hdEEG.cycleplot')

# Stage colours duplicated verbatim from ``frontend/eeg_review_gui.py``
# ``STAGE_COLOR``. They are copied rather than imported because the library must
# stay importable without PyQt/pyqtgraph (the optional-GUI-import rule), and
# importing anything from ``frontend/`` would pull the GUI stack into a headless
# library import. The five values are kept in sync with the GUI by hand.
_STAGE_COLORS = {
    'Wake': '#5d6776',
    'N1': '#58a6ff',
    'N2': '#5fd3a4',
    'N3': '#3fb950',
    'REM': '#d680e0',
}

# Cycle-bar colours: NREM period vs the following REM/inter-NREM segment.
_NREM_BAR_COLOR = '#3a6ea5'   # blue
_REM_BAR_COLOR = '#c0392b'    # red

# Hypnogram row order, top to bottom, with the numeric hypnogram code each row
# maps to (Wake=0, NREM1/2/3=1/2/3, REM=4 as produced by ``get_hypnogram``).
# Wake sits at the top, N3 at the bottom; REM is drawn just below Wake.
_ROW_ORDER = ['Wake', 'REM', 'N1', 'N2', 'N3']
_CODE_TO_LABEL = {0: 'Wake', 4: 'REM', 1: 'N1', 2: 'N2', 3: 'N3'}


def plot_hypnogram_cycles(hypnogram, cycles_by_method, out_path,
                          epoch_length=30, epoch_starts=None,
                          subject=None, stage_colors=None):
    """Render a hypnogram with per-method sleep-cycle bars and save it to PNG.

    The hypnogram is drawn as a coloured staircase with rows ordered
    Wake / REM / N1 / N2 / N3 (Wake at top). Above it, one thin row per method
    in ``cycles_by_method`` shows the detected cycles: a blue rectangle for each
    NREM period (``nrem_start_sec`` -> ``nrem_end_sec``) and, when the cycle has
    a non-empty REM/inter-NREM segment, a red rectangle
    (``nrem_end_sec`` -> ``rem_end_sec``). Rectangles are labelled ``NREM1``,
    ``REM1``, ``NREM2`` ... per cycle. The x-axis is in hours.

    Parameters
    ----------
    hypnogram : sequence of int
        Numeric per-epoch stage codes (Wake=0, NREM1/2/3=1/2/3, REM=4,
        artefact/undefined=-1) as returned by
        ``CustomAnnotations.get_hypnogram()``.
    cycles_by_method : dict
        Mapping ``method -> list of cycle dicts`` as returned by
        :func:`turtlewave_hdEEG.finalize_cycles_and_durations`. Each cycle dict
        must carry ``cycle_number``, ``nrem_start_sec``, ``nrem_end_sec``,
        ``rem_end_sec``, ``rem_start_epoch`` and ``rem_end_epoch`` (the last two
        decide whether the REM segment is non-empty).
    out_path : str
        Destination PNG path.
    epoch_length : float, optional
        Epoch duration in seconds, used when ``epoch_starts`` is not given
        (default 30).
    epoch_starts : sequence of float, optional
        Start time in seconds of each epoch, same length as ``hypnogram``. If
        omitted, epoch ``i`` starts at ``i * epoch_length``.
    subject : str, optional
        Subject label for the figure title.
    stage_colors : dict, optional
        Override for the stage-colour palette (keys ``Wake``, ``N1``, ``N2``,
        ``N3``, ``REM``). Defaults to the module-level ``_STAGE_COLORS``.

    Returns
    -------
    str
        ``out_path`` (the file that was written).

    Notes
    -----
    Artefact/undefined epochs (code -1) leave a gap in the staircase; they are
    not assigned a row. The figure is rasterised through
    :class:`~matplotlib.backends.backend_agg.FigureCanvasAgg`, so this function
    is safe to call in a headless process.
    """
    colors = dict(_STAGE_COLORS)
    if stage_colors:
        colors.update(stage_colors)

    hyp = np.asarray(list(hypnogram), dtype=float)
    n = hyp.size

    if epoch_starts is not None:
        starts = np.asarray(list(epoch_starts), dtype=float)
        if starts.size != n:
            raise ValueError("epoch_starts must match hypnogram length")
    else:
        starts = np.arange(n, dtype=float) * epoch_length

    def end_sec(i):
        if i + 1 < n:
            return float(starts[i + 1])
        return float(starts[i] + epoch_length)

    # y position for each hypnogram row (top row = highest y).
    n_rows = len(_ROW_ORDER)
    row_y = {label: (n_rows - 1 - idx)
             for idx, label in enumerate(_ROW_ORDER)}

    methods = list(cycles_by_method.keys())
    n_method_rows = len(methods)

    fig = Figure(figsize=(12, 3 + 0.6 * n_method_rows))
    # One thin axes per method (cycle bars), then a tall hypnogram axes; all
    # share the x-axis (time in hours).
    height_ratios = [1] * n_method_rows + [5]
    axes = fig.subplots(
        n_method_rows + 1, 1, sharex=True,
        gridspec_kw={'height_ratios': height_ratios})
    if n_method_rows == 0:
        axes = [axes]
    hyp_ax = axes[-1]
    method_axes = axes[:-1]

    # --- cycle-bar rows (one per method) ---
    for m_ax, method in zip(method_axes, methods):
        m_ax.set_ylim(0, 1)
        m_ax.set_yticks([])
        m_ax.set_ylabel(str(method), rotation=0, ha='right', va='center',
                        labelpad=20, fontsize=9)
        for spine in ('top', 'right', 'left'):
            m_ax.spines[spine].set_visible(False)
        for cyc in cycles_by_method[method]:
            num = cyc['cycle_number']
            nrem_lo = cyc['nrem_start_sec'] / 3600.0
            nrem_hi = cyc['nrem_end_sec'] / 3600.0
            m_ax.add_patch(Rectangle(
                (nrem_lo, 0.1), nrem_hi - nrem_lo, 0.8,
                facecolor=_NREM_BAR_COLOR, edgecolor='none', alpha=0.85))
            m_ax.text((nrem_lo + nrem_hi) / 2.0, 0.5, f"NREM{num}",
                      ha='center', va='center', color='white', fontsize=7)
            has_rem = cyc['rem_end_epoch'] >= cyc['rem_start_epoch']
            if has_rem:
                rem_lo = cyc['nrem_end_sec'] / 3600.0
                rem_hi = cyc['rem_end_sec'] / 3600.0
                if rem_hi > rem_lo:
                    m_ax.add_patch(Rectangle(
                        (rem_lo, 0.1), rem_hi - rem_lo, 0.8,
                        facecolor=_REM_BAR_COLOR, edgecolor='none',
                        alpha=0.85))
                    m_ax.text((rem_lo + rem_hi) / 2.0, 0.5, f"REM{num}",
                              ha='center', va='center', color='white',
                              fontsize=7)

    # --- hypnogram staircase ---
    # Grey connecting line for the vertical stage transitions (artefact -> NaN
    # so the line breaks over gaps).
    y_line = np.full(n, np.nan)
    for i in range(n):
        label = _CODE_TO_LABEL.get(int(hyp[i]))
        if label is not None:
            y_line[i] = row_y[label]
    x_hours = starts / 3600.0
    hyp_ax.plot(x_hours, y_line, drawstyle='steps-post',
                color='#888888', linewidth=0.8, zorder=1)

    # Coloured horizontal segment per epoch on top of the grey line.
    for i in range(n):
        label = _CODE_TO_LABEL.get(int(hyp[i]))
        if label is None:
            continue
        y = row_y[label]
        x0 = starts[i] / 3600.0
        x1 = end_sec(i) / 3600.0
        hyp_ax.hlines(y, x0, x1, color=colors.get(label, '#888888'),
                      linewidth=3.0, zorder=2)

    hyp_ax.set_yticks([row_y[label] for label in _ROW_ORDER])
    hyp_ax.set_yticklabels(_ROW_ORDER)
    hyp_ax.set_ylim(-0.5, n_rows - 0.5)
    hyp_ax.set_xlabel("Time (hours)")
    if n > 0:
        hyp_ax.set_xlim(x_hours[0], end_sec(n - 1) / 3600.0)
    for spine in ('top', 'right'):
        hyp_ax.spines[spine].set_visible(False)

    title_methods = ', '.join(str(m) for m in methods) if methods else 'none'
    title_subject = subject if subject else 'hypnogram'
    fig.suptitle(f"{title_subject} — sleep cycles: {title_methods}",
                 fontsize=11)

    canvas = FigureCanvasAgg(fig)  # noqa: F841 - binds Agg backend to fig
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    logger.info("Wrote hypnogram/cycle plot to %s", out_path)
    return out_path


def plot_from_annotations(annotations, cycles_by_method, out_path,
                          epoch_length=30, subject=None):
    """Read the hypnogram off an annotation object and plot the cycles.

    Convenience wrapper around :func:`plot_hypnogram_cycles` that pulls the
    hypnogram and per-epoch start times from a ``CustomAnnotations`` /
    ``XLAnnotations`` object.

    Parameters
    ----------
    annotations : CustomAnnotations
        Annotation wrapper exposing ``get_hypnogram()`` and (optionally)
        ``epochs``.
    cycles_by_method : dict
        Mapping ``method -> list of cycle dicts`` (see
        :func:`plot_hypnogram_cycles`).
    out_path : str
        Destination PNG path.
    epoch_length : float, optional
        Epoch duration in seconds (default 30), used when the annotation exposes
        no epoch grid.
    subject : str, optional
        Subject label for the figure title.

    Returns
    -------
    str
        ``out_path`` (the file that was written).
    """
    hypnogram = annotations.get_hypnogram()

    epoch_starts = None
    try:
        epochs = annotations.epochs
        if epochs:
            epoch_starts = [float(ep['start']) for ep in epochs]
    except (AttributeError, KeyError, TypeError, ValueError):
        epoch_starts = None

    return plot_hypnogram_cycles(
        hypnogram, cycles_by_method, out_path, epoch_length=epoch_length,
        epoch_starts=epoch_starts, subject=subject)
