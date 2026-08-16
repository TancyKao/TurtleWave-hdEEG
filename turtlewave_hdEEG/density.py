"""Event density derived on read from ``neural_events.db``.

Density is **derived, never materialised**. A cached density table goes stale
silently: a scoped channel re-detection deletes and re-inserts that channel's
rows (:func:`turtlewave_hdEEG.dbwrite.write_channel_events`) and nothing in
that transaction could update a cached number, so the one mechanism that
exists to *correct* data would leave a cache holding the pre-correction
density. The numerator here is a ``GROUP BY`` over indexed columns, which is
milliseconds even on a few hundred thousand rows.

Only the denominator is un-derivable from ``events``: the artefact-free
in-stage time actually fed to the detector. That is stored once at detection
time in ``analysed_time``
(:func:`turtlewave_hdEEG.dbwrite.store_analysed_time`).

.. warning::

   ``stage_durations`` is **not** a fallback denominator. It holds raw
   hypnogram time with no artefact subtraction, while detection rejects
   artefact and arousal epochs. Dividing an artefact-free numerator by raw
   hypnogram time under-estimates density in proportion to each recording's
   artefact load -- a per-subject bias that looks like a real group
   difference. When the denominator is missing this module raises, or with
   ``missing='nan'`` returns ``NaN`` and logs loudly. It never substitutes.

Notes
-----
The denominator is channel-global by construction: detection's ``fetch``
passes no ``chan_full``, so artefacts are removed channel-globally and every
channel was fed the same clean time. Density is per-channel through its
numerator only. See
:func:`turtlewave_hdEEG.utils.compute_analysed_seconds` for the parity
argument.
"""

import os
import logging
import sqlite3

import numpy as np
import pandas as pd

logger = logging.getLogger('turtlewave_hdEEG.density')

#: Column order of the frame returned by :func:`event_density`.
DENSITY_COLUMNS = (
    'subject', 'channel', 'event_type', 'method', 'stage',
    'n_events', 'analysed_minutes', 'density_per_min',
    'artefact_minutes_excluded', 'mean_duration_sec',
    'reject_artifacts', 'reject_arousals', 'denominator_source',
)


def format_density_table(df, max_rows=None, float_fmt='%.3f'):
    """Render an :func:`event_density` frame as a block of text for a log.

    Batch drivers have no CSV to open afterwards, so the density has to be
    legible in the job log. The summary line reports the **median and
    interquartile range** across channels rather than the mean and SD: event
    density across an hd-EEG montage is right-skewed and contains bad-channel
    outliers, so a mean is not a useful description of the middle of it.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`event_density`.
    max_rows : int or None, optional
        Truncate the per-channel listing to this many rows per stage (the
        summary line still covers every channel). ``None`` (default) lists
        every row.
    float_fmt : str, optional
        Printf-style format for the density column. Default ``'%.3f'``.

    Returns
    -------
    str
        Multi-line text: one summary line per stage plus the per-channel
        listing. ``'(no rows)'`` when ``df`` is empty.

    Examples
    --------
    >>> print(format_density_table(df))                    # doctest: +SKIP
    NREM2: 3 channels, 39 events, 95.00 min analysed,
           density/min median 0.116 [IQR 0.105-0.158]
    """
    if df is None or len(df) == 0:
        return '(no rows)'

    lines = []
    for stage, block in df.groupby('stage', sort=True, dropna=False):
        dens = block['density_per_min'].astype(float)
        finite = dens[dens.notna()]
        if len(finite):
            q1, med, q3 = (float(finite.quantile(0.25)), float(finite.median()),
                           float(finite.quantile(0.75)))
            spread = (f"density/min median {float_fmt % med} "
                      f"[IQR {float_fmt % q1}-{float_fmt % q3}]")
        else:
            spread = "density/min unavailable (no stored denominator)"
        analysed = block['analysed_minutes'].dropna()
        analysed_txt = (f"{float(analysed.iloc[0]):.2f} min analysed"
                        if len(analysed) else "analysed time MISSING")
        lines.append(
            f"{stage}: {len(block)} channel(s), "
            f"{int(block['n_events'].sum())} events, {analysed_txt}, {spread}")
        shown = block if max_rows is None else block.head(max_rows)
        for _, r in shown.iterrows():
            d = r['density_per_min']
            d_txt = 'nan' if d != d else float_fmt % float(d)
            lines.append(f"    {str(r['channel']):>8}  n={int(r['n_events']):>6}  "
                         f"{d_txt}/min")
        if max_rows is not None and len(block) > max_rows:
            lines.append(f"    ... {len(block) - max_rows} more channel(s)")
    return "\n".join(lines)


def _resolve_subject(conn, subject):
    """Resolve which subject's denominators to use.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open read connection.
    subject : str or None
        Explicit subject, or ``None`` to infer.

    Returns
    -------
    str or None
        The subject to key ``analysed_time`` on, or ``None`` when the table
        is empty.

    Raises
    ------
    ValueError
        If ``subject`` is ``None`` and ``analysed_time`` holds more than one
        subject, since picking one arbitrarily would divide one recording's
        events by another recording's time.
    """
    if subject is not None:
        # Normalise on READ with the same helper the write path uses. Every
        # driver passes the raw CLI value, which in this deployment is the
        # bare directory name ('10sd'); detection stores 'sub-10sd'. Without
        # this, asking for '10sd' finds no denominator and the whole density
        # feature silently reports "unavailable" on every cluster run.
        from .utils import normalize_subject
        return normalize_subject(str(subject))
    rows = [r[0] for r in conn.execute(
        "SELECT DISTINCT subject FROM analysed_time ORDER BY subject")]
    if not rows:
        return None
    if len(rows) > 1:
        raise ValueError(
            f"analysed_time holds denominators for {len(rows)} subjects "
            f"({', '.join(rows)}) and no subject= was given. Pass subject= "
            f"explicitly; dividing one recording's events by another's "
            f"analysed time is not a density.")
    return rows[0]


def _concat_rows(*frames):
    """Concatenate count frames, skipping empties and fixing ``stage`` dtype.

    An all-NULL ``stage`` column comes back from SQLite as ``float64`` and an
    empty frame carries no dtype at all; concatenating either makes pandas
    warn about how it resolves the result dtype. Normalising first keeps the
    column an object column holding real stage strings and ``None``.

    Parameters
    ----------
    *frames : pandas.DataFrame
        Frames with the count-query column set.

    Returns
    -------
    pandas.DataFrame
        The non-empty frames concatenated, or the first frame when only one
        is non-empty.
    """
    kept = []
    for f in frames:
        if f is None or len(f) == 0:
            continue
        f = f.copy()
        f['stage'] = f['stage'].astype(object).where(f['stage'].notna(), None)
        f['mean_duration_sec'] = f['mean_duration_sec'].astype(float)
        kept.append(f)
    if not kept:
        return frames[0]
    if len(kept) == 1:
        return kept[0]
    return pd.concat(kept, ignore_index=True)


def _method_matches(stored, wanted):
    """True when a stored scope method covers the requested one.

    ``processing_status.method`` and ``detection_runs.method`` hold the run's
    method *set*, ``'_'``-joined for a multi-method run
    (``'Moelle2011_Wamsley2012'``), while ``events.method`` holds the single
    method that detected the event. An equality test alone therefore misses
    every multi-method run.

    .. warning::

       Splitting on ``'_'`` is only exact **because no method name that
       reaches these tables contains an underscore**. Every shipped detector
       method is underscore-free, and a slashed method such as
       ``'AASM/Massimini2004'`` keeps its slash in the database (only
       filenames escape it). PAC's paired tokens
       (``'Staresina2015_paired_Moelle2011'``) do contain underscores but live
       in ``pac_coupling``, which this function never reads. Adding an
       underscore to a detector method name would break this split; join the
       set with a character that cannot occur in a method name instead.

    Parameters
    ----------
    stored : str
        Method token as stored on the scope row.
    wanted : str or None
        Method asked for. ``None`` matches anything.

    Returns
    -------
    bool
    """
    if wanted is None:
        return True
    stored = str(stored)
    return stored == str(wanted) or str(wanted) in stored.split('_')


def _method_components(stored):
    """Split a stored scope method token into the methods it names.

    The inverse of :func:`_method_matches`: where that tests membership, this
    enumerates. ``processing_status`` and ``detection_runs`` store the run's
    method *set*, ``'_'``-joined for a multi-method run, and those components
    are exactly the strings ``events.method`` uses -- so they, not the joined
    token, are the identity labels a density row can carry.

    Subject to the same constraint as :func:`_method_matches`: no method name
    that reaches these tables may contain an underscore.

    Parameters
    ----------
    stored : str
        Method token as stored on the scope row.

    Returns
    -------
    list of str
        Constituent method names, in stored order, with blanks dropped.

    Examples
    --------
    >>> _method_components('Moelle2011_Wamsley2012')
    ['Moelle2011', 'Wamsley2012']
    >>> _method_components('AASM/Massimini2004')
    ['AASM/Massimini2004']
    """
    return [part for part in str(stored).split('_') if part]


def _run_stage_components(conn, event_type, method, freq_lower, freq_upper, log):
    """Recover the stage set the matching detection run(s) actually searched.

    ``analysed_time`` is keyed on ``(subject, stage, reject_*)`` with no event
    type or method, so it is the union over every detector that has ever run
    on this recording. Using it as the stage scope invents stages for a
    detector that never searched them -- a fabricated zero-density row, and a
    pooled denominator containing time whose numerator cannot contain events.

    The run's own scope is recorded twice: ``processing_status.stage`` holds
    the joined stage token per processed channel, and ``detection_runs.stages``
    holds the requested list. Both are scoped by event type and method, so
    either recovers the truth.

    Returns the individual **components** rather than the token: they are what
    ``analysed_time`` is keyed on, and :func:`_identity_scope_tokens` decides
    from them (plus what the events actually carry) whether this identity is
    reported per component or under one joint token.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    event_type, method : str or None
        Detection scope. Callers pass ONE identity at a time --
        :func:`event_density` resolves the scope per ``(event_type, method)``
        precisely so the result is never a union across detectors. ``None``
        means "any", which widens the recovered stage set accordingly and is
        only correct when there genuinely is a single identity.
    freq_lower, freq_upper : float or None
        Band bounds, used to narrow ``processing_status``.
    log : logging.Logger
        Logger for the provenance line.

    Returns
    -------
    list of str
        Sorted stage labels, empty when neither table can say.
    """
    from .dbwrite import split_stage_token

    stages = set()

    # (1) processing_status: the per-channel record of what was processed.
    if _table_exists(conn, 'processing_status'):
        where = ["success = 1"]
        params = []
        if event_type is not None:
            where.append("event_type = ?")
            params.append(str(event_type))
        if freq_lower is not None:
            where.append("freq_lower = ?")
            params.append(float(freq_lower))
        if freq_upper is not None:
            where.append("freq_upper = ?")
            params.append(float(freq_upper))
        for stored_method, token in conn.execute(
                "SELECT DISTINCT method, stage FROM processing_status WHERE "
                + " AND ".join(where), params):
            if not token or not _method_matches(stored_method, method):
                continue
            try:
                stages.update(split_stage_token(token))
            except ValueError:
                log.debug("Unparseable processing_status stage token %r; "
                          "ignored for scope recovery", token)
    if stages:
        log.debug("Stage scope recovered from processing_status: %s",
                  sorted(stages))
        return sorted(stages)

    # (2) detection_runs.stages: str() of the requested stage list.
    if _table_exists(conn, 'detection_runs'):
        import ast
        where = []
        params = []
        if event_type is not None:
            where.append("event_type = ?")
            params.append(str(event_type))
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        for stored_method, stages_repr in conn.execute(
                f"SELECT method, stages FROM detection_runs{clause}", params):
            if not stages_repr or not _method_matches(stored_method, method):
                continue
            try:
                value = ast.literal_eval(stages_repr)
            except (ValueError, SyntaxError):
                value = stages_repr
            if isinstance(value, (list, tuple)):
                stages.update(str(s) for s in value)
            elif isinstance(value, str):
                try:
                    stages.update(split_stage_token(value))
                except ValueError:
                    pass
    if stages:
        log.debug("Stage scope recovered from detection_runs: %s",
                  sorted(stages))
    return sorted(stages)


def _table_exists(conn, name):
    """Return True when ``name`` is a table in this database."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def _components(token):
    """Stage components of one stored token (lazy import of the primitive).

    Parameters
    ----------
    token : str or None
        Stage token as stored in ``events.stage``.

    Returns
    -------
    list of str
        Constituent stage labels; an unrecognised token is returned whole.
    """
    from .dbwrite import stage_components
    return stage_components(token)


def _identity_scope_tokens(observed, run_components, has_other_tokens,
                           explicit):
    """Decide which stage TOKENS one (event_type, method) is reported under.

    Density's stage dimension is the run's stage **token**, not a single
    stage. That one change makes both database shapes fall out of the same
    code: a pre-4.3 row carries a one-component token (``'NREM2'``) and a 4.3
    row carries the run's joint token (``'NREM2NREM3'``), so backward
    compatibility is a special case of the general rule rather than a branch.

    Parameters
    ----------
    observed : sequence of str
        Stage tokens this identity's events actually carry, already narrowed
        to the requested stage set.
    run_components : sequence of str
        Individual stages the run searched, from
        :func:`_run_stage_components` (or the explicit ``stage=`` request).
    has_other_tokens : bool
        True when this identity has events under tokens the request does NOT
        cover -- i.e. the requested stages are a strict subset of a stored
        joint token.
    explicit : bool
        True when the caller named ``stage=``. An explicit request is honoured
        as an error rather than reinterpreted.

    Returns
    -------
    list of str
        Tokens to report, in the order given by ``run_components`` (or by
        first appearance for observed tokens).

    Notes
    -----
    The three cases, in order:

    1. **Joint storage** -- any observed token spans more than one stage. The
       events carry the run's own token; report exactly those. A channel that
       ran and fired nothing then gets its zero row under that same token.
    2. **A slice of a joint token was asked for** -- nothing observed, but
       this identity does have events under a token the request does not
       cover. Report NOTHING. A joint row cannot be attributed to one of its
       stages, so an ``'NREM2'`` row here would divide part of a joint count
       by an N2-only denominator, which is a wrong number where the honest
       answer is "not available at this resolution".
    3. **Per-epoch storage, or nothing fired anywhere** -- report one token
       per searched stage, which is exactly the pre-4.3 behaviour, including
       the zero-event row for a stage that was searched and stayed silent.
    """
    from .dbwrite import stage_components

    observed = [str(t) for t in observed]
    if any(len(stage_components(t)) > 1 for t in observed):
        return list(dict.fromkeys(observed))
    if explicit and not observed and has_other_tokens:
        return []
    return list(dict.fromkeys(str(s) for s in run_components))


def event_density(db_path, event_type=None, method=None, stage=None,
                  channel=None, freq_lower=None, freq_upper=None,
                  subject=None, reject_artifacts=True, reject_arousals=True,
                  combine_stages=False, include_zero_channels=True,
                  missing='raise', logger_=None):
    """Per-channel event density derived from the database.

    The numerator is a ``GROUP BY`` count over ``events`` for the requested
    scope. The denominator is the artefact-free analysed time stored in
    ``analysed_time`` for the same stage and the same rejection settings --
    the time actually fed to the detector, not raw hypnogram time.

    Parameters
    ----------
    db_path : str
        Path to ``neural_events.db``.
    event_type : str or None, optional
        Restrict to one event type (``'spindle'``, ``'slow_wave'``,
        ``'k_complex'``). ``None`` (default) returns every type, grouped
        separately.
    method : str or None, optional
        Restrict to one detection method. Give it **UNESCAPED**, exactly as
        stored (e.g. ``'AASM/Massimini2004'``, not ``'AASM_Massimini2004'``).
        Default ``None``.
    stage : str or list of str or None, optional
        Stage(s) to report, in any form the pipeline uses: a list
        (``['NREM2', 'NREM3']``), a single stage (``'NREM2'``), or the joined
        scope token (``'NREM2NREM3'``). The three are equivalent -- all are
        reduced to a set of stages and matched against the tokens the database
        stores. ``None`` (default) uses the stages the run recorded searching,
        falling back to the stages present in the matching events.

        This set, not the set of stages that produced events, defines the
        scope: a stage that was analysed and fired nothing still contributes
        its analysed time to a pooled denominator.

        **A strict subset of a stored joint token returns no row, not a
        number.** A run over ``['NREM2', 'NREM3']`` searched them as one
        concatenated segment and stores every event under ``'NREM2NREM3'``;
        those events cannot be attributed to N2 or N3 individually, so
        ``stage=['NREM2']`` is answered with a warning and no row rather than
        by dividing part of a joint count by an N2-only denominator. Use
        ``events.epoch_stage`` if the split is what you need.
    channel : str or list of str or None, optional
        Restrict to one or more channels. Default ``None`` (all).
    freq_lower, freq_upper : float or None, optional
        Restrict to one detection band. Default ``None``.
    subject : str or None, optional
        Subject keying ``analysed_time``. ``None`` (default) uses the single
        subject in that table, and raises if there is more than one.
    reject_artifacts, reject_arousals : bool, optional
        The rejection settings of the detection run whose density is wanted.
        These select the denominator row, so they must match the run.
        Default ``True`` for both (the detector defaults).
    combine_stages : bool, optional
        When True and more than one stage is in scope, return one row per
        channel for the pooled stages: the counts are summed, the denominators
        are summed per stage in scope (so a span shared by two stages is not
        double-counted, and a stage that fired nothing still contributes its
        analysed time), and ``stage`` is reported as ``'NREM2+NREM3'``.
        Default ``False``.
    include_zero_channels : bool, optional
        Emit a row with ``n_events = 0`` for every channel/stage in scope that
        was processed (``processing_status.success = 1``) but produced no
        event. Without them a montage summary is computed over the channels
        that fired, which inflates it whenever a flat or rejected channel
        yields zero. Default ``True``.
    missing : {'raise', 'nan'}, optional
        What to do when a stage in scope has no stored denominator.
        ``'raise'`` (default) raises ``ValueError`` naming the missing keys.
        ``'nan'`` emits ``logger.warning`` and returns ``NaN`` densities for
        those rows. There is deliberately no option to substitute
        ``stage_durations``. Default ``'raise'``.
    logger_ : logging.Logger or None, optional
        Logger for the warnings. Default ``None`` (module logger).

    Returns
    -------
    pandas.DataFrame
        One row per (channel, event_type, method, stage token) in scope. The
        ``stage`` column is the run's stage **token** as stored: ``'NREM2'``
        for a single-stage run or a pre-4.3 per-epoch row, ``'NREM2NREM3'``
        for a joint run, and ``'NREM2+NREM3'`` for a ``combine_stages`` row
        pooled across tokens. Columns are ``subject``, ``channel``,
        ``event_type``, ``method``,
        ``stage``, ``n_events``, ``analysed_minutes``, ``density_per_min``,
        ``artefact_minutes_excluded``, ``mean_duration_sec``,
        ``reject_artifacts``, ``reject_arousals``, ``denominator_source``.
        Empty (with those columns) when no event matches the scope and no
        channel was processed. Rows whose ``stage`` is ``None`` are events the
        detector could not attribute to a scored epoch; they carry
        ``density_per_min = NaN`` and are reported at ERROR.

    Raises
    ------
    FileNotFoundError
        If ``db_path`` does not exist.
    ValueError
        If ``missing='raise'`` and a stage in scope has no stored denominator,
        or if the subject cannot be resolved unambiguously.

    Examples
    --------
    >>> df = event_density('neural_events.db', event_type='spindle',
    ...                    method='Moelle2011', stage=['NREM2', 'NREM3'])
    ...                                                    # doctest: +SKIP
    >>> df.groupby('stage')['density_per_min'].median()    # doctest: +SKIP
    """
    log = logger_ or logger

    if missing not in ('raise', 'nan'):
        raise ValueError(
            f"missing must be 'raise' or 'nan', got {missing!r}. Substituting "
            f"raw hypnogram time (stage_durations) for the artefact-free "
            f"denominator is not offered: it biases density by each "
            f"recording's artefact load.")

    if not db_path or not os.path.exists(db_path):
        raise FileNotFoundError(
            f"No database at {db_path!r}. Density is derived from "
            f"neural_events.db; run detection with write_db first.")

    # Accept every form the pipeline uses for a stage set, including the
    # joined scope token ('NREM2NREM3') that filenames and processing_status
    # carry. Matching that literal against events.stage (one scored stage per
    # row) would return nothing and read like an empty night.
    stage_list = None
    if stage is not None:
        from .dbwrite import split_stage_token
        stage_list = [str(s) for s in split_stage_token(stage)]

    chan_list = None
    if channel is not None:
        chan_list = [str(c) for c in (channel if isinstance(channel, (list, tuple, set))
                                      else [channel])]

    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        if not _table_exists(conn, 'events'):
            raise ValueError(
                f"{db_path} has no events table; nothing to compute a density "
                f"from.")

        # Scope WITHOUT the stage filter, so a NULL stage can be counted and
        # reported instead of vanishing out of an `IN (...)` predicate.
        base_where = []
        base_params = []
        if event_type is not None:
            base_where.append("event_type = ?")
            base_params.append(str(event_type))
        if method is not None:
            base_where.append("method = ?")
            base_params.append(str(method))
        if chan_list:
            base_where.append("channel IN (%s)" % ",".join("?" * len(chan_list)))
            base_params.extend(chan_list)
        if freq_lower is not None:
            base_where.append("freq_lower = ?")
            base_params.append(float(freq_lower))
        if freq_upper is not None:
            base_where.append("freq_upper = ?")
            base_params.append(float(freq_upper))

        # Every distinct stage token in the scope, with its event count, before
        # any stage filter. Needed three times: to resolve the requested stages
        # to the tokens actually stored; to tell "this identity fired nothing"
        # from "this identity fired only under tokens the request does not
        # cover"; and to say HOW MANY events a request silently leaves out.
        from .dbwrite import stage_tokens_covering
        base_clause_pre = ((" WHERE " + " AND ".join(base_where))
                           if base_where else "")
        tokens_by_identity = {}
        for et_, m_, tok_, n_ in conn.execute(
                "SELECT event_type, method, stage, COUNT(*) FROM events"
                + base_clause_pre
                + " GROUP BY event_type, method, stage", base_params):
            if tok_ is None:
                continue
            tokens_by_identity.setdefault(
                (str(et_), str(m_)), {})[str(tok_)] = int(n_)

        where = list(base_where)
        params = list(base_params)
        if stage_list:
            # Resolve the requested stages to the tokens the database holds:
            # 'NREM2NREM3' is selected by a request for both its stages and by
            # nothing narrower. Filtering on the raw request would match a
            # per-epoch database only.
            all_tokens = [t for toks in tokens_by_identity.values()
                          for t in toks]
            stage_tokens = stage_tokens_covering(all_tokens, stage_list)
            if stage_tokens:
                where.append("stage IN (%s)" % ",".join("?" * len(stage_tokens)))
                params.extend(stage_tokens)
            else:
                # No stored token lies inside the requested stages. Match
                # nothing: dropping the predicate would count every stage.
                where.append("1 = 0")
        clause = (" WHERE " + " AND ".join(where)) if where else ""

        sql = (
            "SELECT channel, event_type, method, stage, COUNT(*) AS n_events, "
            "AVG(duration) AS mean_duration_sec "
            f"FROM events{clause} "
            "GROUP BY channel, event_type, method, stage "
            "ORDER BY event_type, method, stage, channel")
        counts = pd.read_sql_query(sql, conn, params=params)

        # Events with no resolved stage. These are a broken run, not an empty
        # one: the detector could not attribute them to a scored epoch, so
        # they have no denominator and no density. They must never be
        # silently dropped -- that turns a complete run into "nothing
        # detected".
        base_clause = (" WHERE " + " AND ".join(base_where)) if base_where else ""
        null_sql = (f"SELECT COUNT(*), COUNT(DISTINCT channel) FROM events"
                    f"{base_clause}"
                    f"{' AND' if base_where else ' WHERE'} stage IS NULL")
        n_null, n_null_chan = conn.execute(null_sql, base_params).fetchone()
        if n_null:
            log.error(
                "%d event(s) across %d channel(s) in this scope have a NULL "
                "stage: the detector could not resolve the scored epoch they "
                "fall in (a failed epoch lookup, or an event outside every "
                "scored epoch). They have no denominator, so they are "
                "reported with density NaN and are NOT counted in any "
                "stage's density. Re-run detection with valid scoring rather "
                "than reading these numbers as an empty night.",
                int(n_null), int(n_null_chan))
            # Re-attach them only when a stage filter excluded them. With no
            # filter the main query already returned them, and concatenating
            # again would double every NULL-stage row.
            if stage_list:
                null_rows = pd.read_sql_query(
                    "SELECT channel, event_type, method, stage, "
                    "COUNT(*) AS n_events, AVG(duration) AS mean_duration_sec "
                    f"FROM events{base_clause}"
                    f"{' AND' if base_where else ' WHERE'} stage IS NULL "
                    "GROUP BY channel, event_type, method",
                    conn, params=base_params)
                counts = _concat_rows(counts, null_rows)

        # Channel roster for honest zeros: every channel this scope actually
        # processed, not only the ones that fired. Without it a flat or
        # rejected channel contributes no row and a montage median is taken
        # over the survivors only.
        roster_by_identity = {}   # (event_type, method) -> {channel, ...}
        if include_zero_channels and _table_exists(conn, 'processing_status'):
            ps_where = ["success = 1"]
            ps_params = []
            if event_type is not None:
                ps_where.append("event_type = ?")
                ps_params.append(str(event_type))
            if freq_lower is not None:
                ps_where.append("freq_lower = ?")
                ps_params.append(float(freq_lower))
            if freq_upper is not None:
                ps_where.append("freq_upper = ?")
                ps_params.append(float(freq_upper))
            if chan_list:
                ps_where.append("channel IN (%s)" % ",".join("?" * len(chan_list)))
                ps_params.extend(chan_list)
            # processing_status says WHICH CHANNELS RAN. It is not an identity
            # label: its `method` column holds the run's joined method SET
            # ('Moelle2011_Wamsley2012'), while events.method holds the single
            # method that detected each event. Using the joined string as a
            # label invents an identity that appears nowhere in events, and
            # the channels that ran but fired nothing end up under it instead
            # of under the real methods -- so the honest zeros are lost
            # exactly where they matter.
            #
            # So: filter by method MEMBERSHIP, then label by the constituent
            # methods (_method_components), which are the same strings
            # events.method uses.
            for ch, stored_method, ps_event_type in conn.execute(
                    "SELECT DISTINCT channel, method, event_type FROM "
                    "processing_status WHERE " + " AND ".join(ps_where),
                    ps_params):
                if not _method_matches(stored_method, method):
                    continue
                if method is not None:
                    components = [str(method)]
                else:
                    components = _method_components(stored_method)
                for comp in components:
                    ident = (str(ps_event_type), comp)
                    roster_by_identity.setdefault(ident, set()).add(str(ch))

        if counts.empty and not roster_by_identity:
            log.warning(
                "No events matched the requested scope (event_type=%s, "
                "method=%s, stage=%s) in %s, so there is no density to "
                "report.", event_type, method, stage, db_path)
            return pd.DataFrame(columns=list(DENSITY_COLUMNS))

        if not _table_exists(conn, 'analysed_time'):
            raise ValueError(
                f"{db_path} has no analysed_time table, so the artefact-free "
                f"density denominator was never stored. Re-run detection with "
                f"a current turtlewave, or back-fill analysed_time. Raw "
                f"hypnogram time (stage_durations) is not substituted because "
                f"it biases density by the recording's artefact load.")

        resolved_subject = _resolve_subject(conn, subject)
        denom = {}
        if resolved_subject is not None:
            for row in conn.execute(
                    "SELECT stage, analysed_seconds, artefact_seconds_excluded, "
                    "source FROM analysed_time WHERE subject = ? AND "
                    "reject_artifacts = ? AND reject_arousals = ?",
                    (resolved_subject, 1 if reject_artifacts else 0,
                     1 if reject_arousals else 0)):
                denom[str(row[0])] = {
                    'analysed_seconds': float(row[1]),
                    'artefact_seconds_excluded': row[2],
                    'source': row[3],
                }

        # ------------------------------------------------------------------
        # Stage scope, resolved PER (event_type, method) identity.
        # ------------------------------------------------------------------
        # The scope is what each RUN searched -- never what happened to fire,
        # never the union across detectors, and never the union across
        # identities. analysed_time is keyed on (subject, stage, reject_*)
        # with no event type or method, so it is shared by every detector;
        # using it (or a union over identities) as the scope invents a stage
        # some identity never searched, and a hard 0.0 against a real
        # denominator is a fabricated result, not a conservative one.
        #
        # A combination that was never searched gets NO row at all.
        identities = set()
        for r in counts.itertuples():
            identities.add((str(r.event_type), str(r.method)))
        identities |= set(roster_by_identity)
        if event_type is not None and method is not None:
            identities.add((str(event_type), str(method)))

        scope_by_identity = {}
        for ident in sorted(identities):
            et, m = ident
            # Tokens this identity's events carry ({token: n_events}), and
            # which of them the request leaves OUT. Testing "nothing observed"
            # instead of "something excluded" is not equivalent: an identity
            # holding both 'NREM2' (legacy per-epoch rows) and 'NREM2NREM3'
            # (a later joint run) answers stage=['NREM2'] from the first token
            # alone and silently drops every event under the second -- a
            # number, not a missing row, and the one that looks most like an
            # answer. That state is also what a partially-migrated database
            # is in.
            ident_tokens = tokens_by_identity.get(ident, {})
            observed = stage_tokens_covering(list(ident_tokens), stage_list)
            excluded = [t for t in ident_tokens if t not in set(observed)]
            has_other_tokens = bool(excluded)
            if excluded and observed:
                log.warning(
                    "stage=%s does not cover stage token(s) %s that "
                    "event_type=%s, method=%s also holds, so %d event(s) are "
                    "EXCLUDED from this density and %d are reported. A joint "
                    "token is only selected when every one of its stages was "
                    "asked for, because its events cannot be attributed to "
                    "one of them. Ask for the full set to include them, or "
                    "read events.epoch_stage for the per-epoch split. This "
                    "usually means the scope was detected twice under "
                    "different stage sets, or the database is only partly "
                    "migrated to the joint token.",
                    stage_list, sorted(excluded), et, m,
                    sum(ident_tokens[t] for t in excluded),
                    sum(ident_tokens[t] for t in observed))

            if stage_list:
                # Explicitly requested. The components are honoured as given;
                # which TOKENS they map to depends on how this identity's
                # events are stored.
                tokens = _identity_scope_tokens(
                    observed, list(dict.fromkeys(stage_list)),
                    has_other_tokens, explicit=True)
                if not tokens and has_other_tokens:
                    log.warning(
                        "stage=%s asked for a strict subset of the stage "
                        "token(s) %s that event_type=%s, method=%s is stored "
                        "under, so no row is returned for it. Those events "
                        "were detected over the whole stage set as one "
                        "segment and cannot be attributed to one of its "
                        "stages; ask for the full set. (events.epoch_stage "
                        "holds each event's own scored epoch if you need the "
                        "split.)", stage_list, sorted(set(ident_tokens)), et, m)
                scope_by_identity[ident] = tokens
                continue

            searched = _run_stage_components(
                conn, et, m, freq_lower, freq_upper, log)
            if not searched:
                searched = sorted({s for tok in observed
                                   for s in _components(tok)})
                if searched:
                    log.warning(
                        "stage=None and this database records no detection "
                        "scope for event_type=%s, method=%s, so its stage set "
                        "falls back to the stages that produced events (%s). "
                        "A stage that was searched and fired nothing is not "
                        "represented; pass stage= explicitly for a pooled "
                        "density.", et, m, ", ".join(searched))
            else:
                # Implicit scope only: drop stages with no stored denominator
                # rather than raising. processing_status is not keyed by the
                # rejection settings, so a stale row from an earlier run with
                # different settings would otherwise make the default call
                # fail on databases that exist today. An EXPLICIT stage=
                # request still raises below -- the caller asked for it.
                undenominated = [s for s in searched if s not in denom]
                if undenominated:
                    searched = [s for s in searched if s in denom]
                    log.warning(
                        "stage=None for event_type=%s, method=%s: stage(s) %s "
                        "were searched but have no stored artefact-free time "
                        "for reject_artifacts=%s, reject_arousals=%s. They are "
                        "left out of the stage scope, so no zero-event rows "
                        "are added for them and they take no part in a pooled "
                        "denominator -- but any events actually detected in "
                        "them are STILL REPORTED, with analysed_minutes and "
                        "density_per_min as NaN and denominator_source "
                        "'missing' (format_density_table renders those as "
                        "'nan/min'). This usually means processing_status "
                        "carries rows from a run with different rejection "
                        "settings (it is not keyed by them). Pass stage= "
                        "explicitly to make a missing denominator an error.",
                        et, m, undenominated, reject_artifacts, reject_arousals)
            # Components -> tokens. For a per-epoch database this is the
            # identity mapping (one token per component) and the behaviour is
            # unchanged; for a joint-token run it collapses to the one token
            # the events actually carry.
            scope_by_identity[ident] = _identity_scope_tokens(
                observed, searched, has_other_tokens, explicit=False)
            if scope_by_identity[ident]:
                log.debug("Stage token scope for %s/%s: %s", et, m,
                          scope_by_identity[ident])
    finally:
        conn.close()

    # A missing denominator is an error only for an EXPLICITLY requested stage.
    # The implicit (stage=None) scope already dropped undenominated stages
    # above, per identity.
    if stage_list:
        missing_stages = [s for s in dict.fromkeys(stage_list) if s not in denom]
        if missing_stages:
            who = ("no subject at all (the analysed_time table is empty)"
                   if resolved_subject is None
                   else f"subject {resolved_subject!r}")
            msg = (
                f"No artefact-free denominator stored for {who}, "
                f"stage(s) {missing_stages}, "
                f"reject_artifacts={reject_artifacts}, "
                f"reject_arousals={reject_arousals}. Density is undefined "
                f"without it. Common causes: those rejection settings do not "
                f"match the detection run (they are part of the analysed_time "
                f"key); the stage was never detected on; or the run predates "
                f"4.2 and stored no analysed_time. Re-run detection (which "
                f"stores it), back-fill it, or ask for the stages the run "
                f"actually searched. Raw hypnogram time is deliberately NOT "
                f"substituted: it would under-estimate density in proportion "
                f"to this recording's artefact load.")
            if missing == 'raise':
                raise ValueError(msg)
            log.warning(msg)
    else:
        missing_stages = []

    counts['stage'] = counts['stage'].astype(object)

    # Honest zeros: every (channel, stage) a given identity processed gets a
    # row whether or not it fired. Crossing identities with a shared stage set
    # would fabricate rows for combinations that were never searched, so the
    # fill is done per identity against that identity's own scope.
    if include_zero_channels:
        have = {(str(r.channel), str(r.event_type), str(r.method), str(r.stage))
                for r in counts.itertuples()}
        filler = []
        for ident, ident_stages in scope_by_identity.items():
            et, m = ident
            fired = {str(r.channel) for r in counts.itertuples()
                     if (str(r.event_type), str(r.method)) == ident}
            channels = sorted(roster_by_identity.get(ident, set()) | fired)
            for ch in channels:
                for stg in ident_stages:
                    if (ch, et, m, stg) not in have:
                        filler.append({'channel': ch, 'event_type': et,
                                       'method': m, 'stage': stg,
                                       'n_events': 0,
                                       'mean_duration_sec': np.nan})
        if filler:
            counts = _concat_rows(counts, pd.DataFrame(filler))
            log.info(
                "Added %d zero-event row(s) for channel/stage combinations "
                "that were processed but produced no events, so a montage "
                "summary is not taken over the firing channels only.",
                len(filler))

    poolable = {ident: stgs for ident, stgs in scope_by_identity.items()
                if len(stgs) > 1}
    if combine_stages and not poolable:
        # No identity has more than one stage TOKEN. For a joint-token run
        # that is the normal case and a correct no-op -- the single token
        # 'NREM2NREM3' already pools its stages in its denominator, which is
        # exactly what combine_stages asks for. It is only worth attention
        # when an identity has one stage because an implicit scope dropped an
        # undenominated one. Reported at INFO, naming what each identity is
        # over, so the two situations are told apart by reading it.
        log.info(
            "combine_stages=True: nothing further to pool. Each identity "
            "already has a single stage token in scope (%s). For a run stored "
            "under a joint token this is the expected no-op -- the token's "
            "denominator is already the sum over its stages. The per-token "
            "rows are returned unchanged.",
            "; ".join(f"{et}/{m}: {stgs or 'no stages'}"
                      for (et, m), stgs in sorted(scope_by_identity.items()))
            or 'no identities')
    elif combine_stages:
        skipped = {ident: stgs for ident, stgs in scope_by_identity.items()
                   if len(stgs) <= 1}
        if skipped:
            log.info(
                "combine_stages=True: %s already had a single stage token in "
                "scope, so their rows are returned per token rather than "
                "pooled further.",
                "; ".join(f"{et}/{m} ({stgs or 'no stages'})"
                          for (et, m), stgs in sorted(skipped.items())))
    if combine_stages and poolable:
        # Pool per identity: each (event_type, method) is pooled over ITS OWN
        # stages, so a detector that searched one stage is not given another
        # detector's analysed time.
        blocks = []
        for ident, ident_stages in sorted(scope_by_identity.items()):
            et, m = ident
            sel = counts[(counts['event_type'].astype(str) == et)
                         & (counts['method'].astype(str) == m)
                         & (counts['stage'].astype(object).isin(ident_stages))]
            if sel.empty:
                continue
            # Pool over the DEDUPLICATED UNION of the tokens' components, not
            # over the tokens' pooled seconds. A database holding both
            # 'NREM2' and 'NREM2NREM3' for one identity would otherwise count
            # N2's analysed time twice and halve the pooled density.
            union = []
            for tok in ident_stages:
                for comp in _components(tok):
                    if comp not in union:
                        union.append(comp)
            have_denom = [s for s in union if s in denom]
            complete = len(have_denom) == len(union) and bool(union)
            pooled_sec = sum(denom[s]['analysed_seconds'] for s in have_denom)
            pooled_artefact = sum((denom[s]['artefact_seconds_excluded'] or 0.0)
                                  for s in have_denom)
            block = sel.copy()
            # Count-weighted mean duration, so pooling two stages does not give
            # a 3-event stage the same weight as a 3000-event one.
            block['_dur_sum'] = (block['mean_duration_sec'].astype(float)
                                 * block['n_events'].astype(float))
            grouped = (block.groupby(['channel', 'event_type', 'method'],
                                     dropna=False)
                       .agg(n_events=('n_events', 'sum'),
                            _dur_sum=('_dur_sum', 'sum'))
                       .reset_index())
            grouped['mean_duration_sec'] = np.where(
                grouped['n_events'].to_numpy(dtype=float) > 0,
                grouped['_dur_sum'].to_numpy(dtype=float)
                / grouped['n_events'].to_numpy(dtype=float),
                np.nan)
            grouped = grouped.drop(columns=['_dur_sum'])
            grouped['stage'] = "+".join(ident_stages)
            grouped['analysed_minutes'] = (pooled_sec / 60.0 if complete
                                           else np.nan)
            grouped['artefact_minutes_excluded'] = (pooled_artefact / 60.0
                                                    if complete else np.nan)
            grouped['denominator_source'] = (
                "+".join(sorted({str(denom[s]['source']) for s in have_denom}))
                if complete else 'missing')
            blocks.append(grouped)

        null_stage = int(counts['stage'].isna().sum())
        if null_stage:
            log.warning(
                "%d row(s) with a NULL stage are excluded from the pooled "
                "density; their events have no share of any denominator.",
                null_stage)
        rows = (pd.concat(blocks, ignore_index=True) if blocks
                else pd.DataFrame(columns=list(DENSITY_COLUMNS)))
        if rows.empty:
            log.warning("Nothing to pool for the requested scope.")
            return pd.DataFrame(columns=list(DENSITY_COLUMNS))
    else:
        from .dbwrite import pooled_denominator
        rows = counts.copy()
        # One denominator per stage TOKEN: for a one-component token this is
        # that stage's stored row (unchanged from before); for a joint token
        # it is the sum over its components, all-or-nothing.
        pooled = [pooled_denominator(s, denom) if s is not None
                  and str(s) != 'nan' else None
                  for s in rows['stage']]
        rows['analysed_minutes'] = [
            np.nan if p is None else p.analysed_seconds / 60.0 for p in pooled]
        rows['artefact_minutes_excluded'] = [
            np.nan if p is None else p.artefact_seconds_excluded / 60.0
            for p in pooled]
        rows['denominator_source'] = [
            'missing' if p is None else p.source for p in pooled]

    with np.errstate(divide='ignore', invalid='ignore'):
        rows['density_per_min'] = np.where(
            rows['analysed_minutes'].to_numpy(dtype=float) > 0,
            rows['n_events'].to_numpy(dtype=float)
            / rows['analysed_minutes'].to_numpy(dtype=float),
            np.nan)

    zero_denom = rows['analysed_minutes'].notna() & (rows['analysed_minutes'] <= 0)
    if zero_denom.any():
        log.warning(
            "%d row(s) have a stored analysed time of zero minutes; their "
            "density is NaN, not zero. A stage with no artefact-free time "
            "analysed cannot have a density.", int(zero_denom.sum()))

    rows['subject'] = resolved_subject
    rows['reject_artifacts'] = bool(reject_artifacts)
    rows['reject_arousals'] = bool(reject_arousals)

    rows = rows[list(DENSITY_COLUMNS)].sort_values(
        ['event_type', 'method', 'stage', 'channel']).reset_index(drop=True)
    log.info(
        "Density for %d channel/stage row(s) from %s (denominator: "
        "artefact-free analysed time, subject=%s)",
        len(rows), os.path.basename(db_path), resolved_subject)
    return rows
