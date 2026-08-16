#!/usr/bin/env python
"""Collapse ``events.stage`` to the run's joint stage token, in place.

From 4.3 a detection run stores ONE stage token on every event it writes: the
canonical join of the stage set it searched (``'NREM2NREM3'`` for a run over
N2+N3, ``'NREM2'`` for a single-stage run). Databases written by 4.2 and
earlier stored each event's own scored epoch stage instead, and the two forms
cannot be mixed within one scope, because
:func:`turtlewave_hdEEG.dbwrite.event_uuid5` hashes the stage:

    re-detecting a 4.2 scope under 4.3 produces a new uuid **and** a new
    stage, so neither the primary key nor the ``event_chan_time`` UNIQUE
    constraint matches the stored row, and ``INSERT OR REPLACE`` appends a
    complete duplicate set instead of replacing it. Every count and density in
    that scope doubles, silently.

:func:`turtlewave_hdEEG.dbwrite.assert_stage_format_compatible` refuses that
run and names this script. This script is how the run is made possible: it
rewrites the stored stages to the token the detector will use, then stamps
``db_meta.stage_format = 'joint'`` so the guard stands down.

**Reversal is the backup, and nothing else.** Collapsing ``'NREM2'`` and
``'NREM3'`` into ``'NREM2NREM3'`` destroys the per-epoch distinction; putting
it back means re-reading the hypnogram, which this script does not do and
cannot do from the database alone. That is why a backup is taken before any
write and why the script refuses to overwrite one that already exists. If the
result is wrong, restore the backup file over the database. (Rows written by
4.3 carry the per-epoch stage in ``events.epoch_stage`` as well, so for those
the distinction survives -- but a 4.2 database has no such column populated.)

What it does, in order
----------------------
1. Refuses a path containing ``archive`` unless ``--allow-archive``.
2. ``PRAGMA integrity_check``.
3. Reports what would change, per detection scope
   ``(event_type, method, freq_lower, freq_upper)`` and, **within a scope, per
   group of channels sharing a stage set**. One scope is not necessarily one
   run: ``processing_status`` is keyed per (channel, stage), so it can record
   that one channel was searched over N2 alone while another was searched over
   N2+N3. Those get different tokens. Unioning them would relabel the
   N2-only channel ``'NREM2NREM3'``, and its events would afterwards divide by
   N2+N3 analysed time -- an understated density, permanently, with nothing
   recording it. **Dry run by default**; ``--apply`` is required to write
   anything.
4. Backs the database up with :meth:`sqlite3.Connection.backup` (WAL-safe:
   it copies committed content, not the file, so a ``-wal`` sidecar cannot be
   left behind), refusing if the backup path already exists.
5. Brings the schema current
   (:func:`turtlewave_hdEEG.dbwrite.ensure_direct_write_schema`).
6. Pre-checks for collisions: after the rewrite every row in a group carries
   the same stage, so two rows sharing
   ``(event_type, channel, start_time, method, freq_lower, freq_upper)`` would
   violate the ``event_chan_time`` UNIQUE constraint. The check counts every
   row that will END UP on the target -- including one that already sits
   there, which is exactly what a moving row lands on top of. Structurally
   impossible from one consistent run, possible in a database re-imported
   after re-scoring or holding two runs.
7. One ``UPDATE`` per channel group, all in ONE transaction, touching only
   ``stage``.
8. Asserts afterwards that the row count is unchanged and that every column
   except ``stage`` is byte-identical (a digest taken before and after, over
   the columns that existed **before** step 5 added more).
9. Checks that the result is actually re-detectable, and stamps
   ``db_meta.stage_format = 'joint'`` **only then**. A row the rewrite could
   not resolve -- a NULL stage kept by ``--keep-null-stage``, or a scope whose
   stage set is on record nowhere -- keeps refusing every future re-detection
   of its scope, so claiming success there would send the user in a circle:
   the migration says done, the detector says run the migration. Such a run
   names the scopes, leaves the marker unstamped and exits 3.
10. Optionally back-fills ``analysed_time`` (the density denominator) and
    ``sleep_cycles`` / ``stage_durations`` / ``events.cycle``, both defaulting
    ON when their tables are empty -- a stage collapse alone leaves density
    and cycles exactly as unavailable as they were. These run **before** the
    step-9 verdict, so a database that ends up only partly unblocked still
    gets the back-fill work that was asked for, and the exit-3 message says
    it ran.

A scope whose rows ALREADY carry the joint token (the common case for
spindles, whose 4.0.x CSV importer flattened the requested stage list into the
same joined form) is reported as a no-op and rewritten not at all; the marker
alone is what that database needs.

A row with a **NULL stage** is relabelled with its run's token by default.
That is not inventing a value: from 4.3 ``events.stage`` is the run's stage
SCOPE, which is known for every row in the scope, and the per-epoch
uncertainty a NULL used to express is preserved in ``events.epoch_stage``.
4.2's direct-write path stored NULL whenever no scored epoch contained an
event, and the 4.0.x CSV importer stored NULL for every row of a CSV with no
Stage column, so this is the normal state of the databases this script
targets. ``--keep-null-stage`` opts out, and then the script reports the
database as not unblocked rather than pretending otherwise.

The annotation XML is never written, including by the cycle back-fill.

Examples
--------
Dry run (default; writes nothing)::

    python examples/migrate_stage_to_joint.py /data/sub-10sd/wonambi/neural_events.db

Apply, with both back-fills fed from the scoring::

    python examples/migrate_stage_to_joint.py \\
        /data/sub-10sd/wonambi/neural_events.db --apply \\
        --annot /data/sub-10sd/wonambi/sub-10sd_scoring.xml
"""

import os
import sys
import json
import hashlib
import logging
import sqlite3
import argparse

# Allow running from a source checkout without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turtlewave_hdEEG import dbwrite  # noqa: E402
from turtlewave_hdEEG.dbwrite import (  # noqa: E402
    STAGE_FORMAT_KEY, STAGE_FORMAT_JOINT, join_stage_token, stage_components,
    split_stage_token)

LOG = logging.getLogger('migrate_stage_to_joint')

#: Suffix of the pre-migration backup written beside the database.
BACKUP_SUFFIX = '.pre-joint-stage.bak'


def _connect(db_path, read_only=True):
    """Open the database, read-only by URI when only reading.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    read_only : bool, optional
        Open with ``mode=ro`` so a dry run cannot write, and cannot create the
        file if the path is mistyped. Default ``True``.

    Returns
    -------
    sqlite3.Connection
    """
    if read_only:
        uri = 'file:' + os.path.abspath(db_path).replace('?', '%3f') + '?mode=ro'
        return sqlite3.connect(uri, uri=True, timeout=60.0)
    # Writes go through the shared opener so the journal mode of an existing
    # database (DELETE, on the network shares this pipeline runs on) is
    # preserved rather than silently promoted to WAL.
    return dbwrite.open_write_connection(db_path, logger=LOG)


def _table_exists(conn, name):
    """Return True when ``name`` is a table in this database."""
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone() is not None


def _event_columns(conn):
    """Column names of the ``events`` table, in declared order."""
    return [r[1] for r in conn.execute("PRAGMA table_info(events)")]


def digest_excluding_stage(conn, columns=None):
    """Digest a fixed set of ``events`` columns, excluding ``stage``.

    Taken before and after the rewrite, this is what turns "only the stage
    column changed" from a claim into a check. A migration that touched an
    amplitude, a uuid or a run_id -- through a typo in the UPDATE, or a
    trigger nobody remembered -- changes this digest.

    **The column list must be pinned across the two calls.** Between them the
    script runs
    :func:`turtlewave_hdEEG.dbwrite.ensure_direct_write_schema`, which ADDS
    columns (``epoch_stage``, ``det_*``, ``run_id``) to a pre-4.3 events
    table. Re-deriving the list from ``PRAGMA table_info`` on the second call
    hashes those new all-NULL columns too, so the digests can never match and
    the check fails on precisely the databases it exists to protect -- after
    the rewrite has committed, and before the marker is stamped. Pass the
    first call's ``columns`` back in.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    columns : list of str or None, optional
        Exact columns to hash. ``None`` (the first call) derives them from the
        table, excluding ``stage``.

    Returns
    -------
    tuple
        ``(n_rows, hexdigest, columns)``. ``(0, '', [])`` when there is no
        ``events`` table. ``columns`` is what was hashed, to be passed back
        into the comparison call.

    Raises
    ------
    ValueError
        If a pinned column no longer exists. A column DISAPPEARING is real
        damage rather than the benign additive migration, so it is surfaced
        rather than silently dropped from the comparison.

    Notes
    -----
    Rows are read ordered by ``uuid`` so the digest does not depend on
    physical row order, and each value is hashed via ``repr`` so ``None``,
    ``0`` and ``'0'`` are distinguishable.
    """
    if not _table_exists(conn, 'events'):
        return (0, '', [])
    present = _event_columns(conn)
    if columns is None:
        cols = [c for c in present if c != 'stage']
    else:
        cols = list(columns)
        missing = [c for c in cols if c not in present]
        if missing:
            raise ValueError(
                f"Column(s) {missing} present before the migration are gone "
                f"afterwards. That is not the additive schema migration; the "
                f"database has been damaged. Restore the backup.")
    if not cols:
        return (0, '', [])
    md5 = hashlib.md5()
    n = 0
    for row in conn.execute(
            f"SELECT {', '.join(cols)} FROM events ORDER BY uuid"):
        md5.update(repr(row).encode('utf-8', 'replace'))
        n += 1
    return (n, md5.hexdigest(), cols)


def scopes_with_stages(conn):
    """Enumerate detection scopes and the stage tokens each one holds.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.

    Returns
    -------
    list of dict
        One entry per ``(event_type, method, freq_lower, freq_upper)``, with
        keys ``event_type``, ``method``, ``freq_lower``, ``freq_upper``,
        ``tokens`` (``{token: n_rows}``, NULL stages under the key ``None``)
        and ``n_rows``.
    """
    out = {}
    for et, meth, lo, hi, stg, n in conn.execute(
            "SELECT event_type, method, freq_lower, freq_upper, stage, "
            "COUNT(*) FROM events "
            "GROUP BY event_type, method, freq_lower, freq_upper, stage"):
        key = (et, meth, lo, hi)
        entry = out.setdefault(key, {
            'event_type': et, 'method': meth, 'freq_lower': lo,
            'freq_upper': hi, 'tokens': {}, 'n_rows': 0})
        entry['tokens'][stg] = entry['tokens'].get(stg, 0) + n
        entry['n_rows'] += n
    return [out[k] for k in sorted(out, key=lambda k: tuple(str(x) for x in k))]


def _method_matcher(method):
    """Predicate matching a run's method SET against one event method.

    ``processing_status.method`` / ``detection_runs.method`` hold the run's
    method set, ``'_'``-joined for a multi-method run, while ``events.method``
    holds the single method that detected the event.
    """
    def matches(stored):
        stored = str(stored)
        return stored == str(method) or str(method) in stored.split('_')
    return matches


def recorded_stage_by_channel(conn, event_type, method, freq_lower,
                              freq_upper):
    """The stage set each CHANNEL was recorded as having been searched over.

    Per channel, not per scope. ``processing_status`` is keyed on
    ``(channel, event_type, method, band, stage)``, so it records that Cz was
    searched over N2+N3 while Fz was searched over N2 alone -- and those are
    two different tokens even though the two rows share an event type, a
    method and a band.

    Throwing that dimension away and unioning the stages across the scope is
    not a rounding error: it relabels the N2-only channel ``'NREM2NREM3'``, so
    its events are afterwards divided by N2+N3 analysed time. That channel's
    density is understated by ``N2 / (N2 + N3)``, permanently, with nothing in
    the database recording that it happened.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    event_type, method : str
        Detection scope. ``method`` is matched by membership of the run's
        method set.
    freq_lower, freq_upper : float or None
        Band bounds.

    Returns
    -------
    dict
        ``{channel: canonical token}``. Empty when ``processing_status``
        cannot say. A channel with more than one recorded stage set is
        reported and left out, since no single token is right for it.
    """
    if not _table_exists(conn, 'processing_status'):
        return {}
    matches = _method_matcher(method)
    try:
        rows = conn.execute(
            "SELECT DISTINCT channel, method, stage FROM processing_status "
            "WHERE event_type IS ? AND freq_lower IS ? AND freq_upper IS ? "
            "  AND success = 1",
            (event_type, freq_lower, freq_upper)).fetchall()
    except sqlite3.OperationalError:
        return {}

    per_channel = {}
    for channel, stored_method, token in rows:
        if not token or not matches(stored_method):
            continue
        try:
            stages = split_stage_token(token)
        except ValueError:
            LOG.debug("Unparseable processing_status token %r; ignored", token)
            continue
        per_channel.setdefault(str(channel), set()).update(
            join_stage_token(stages) for _ in (0,))

    out = {}
    for channel, tokens in per_channel.items():
        if len(tokens) == 1:
            out[channel] = next(iter(tokens))
        else:
            LOG.warning(
                "Channel %s has %d different recorded stage sets for %s/%s "
                "(%s). No single token is correct for it, so it is planned "
                "from the stages its own events carry instead.",
                channel, len(tokens), event_type, method, sorted(tokens))
    return out


def recorded_stage_set(conn, event_type, method, freq_lower, freq_upper):
    """The scope-wide stage set on record, used only where per-channel fails.

    Falls back to ``detection_runs.stages`` (the requested list) when
    ``processing_status`` has nothing for a channel. Scope-wide by
    construction, so it is only correct when every channel really did share
    one stage set -- which is why :func:`recorded_stage_by_channel` is
    consulted first.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    event_type, method : str
        Detection scope.
    freq_lower, freq_upper : float or None
        Band bounds.

    Returns
    -------
    tuple
        ``(stages, source)`` -- a set of stage labels and where it came from,
        or ``(set(), None)``.
    """
    matches = _method_matcher(method)
    if _table_exists(conn, 'processing_status'):
        stages = set()
        try:
            rows = conn.execute(
                "SELECT DISTINCT method, stage FROM processing_status "
                "WHERE event_type IS ? AND freq_lower IS ? AND freq_upper IS ?",
                (event_type, freq_lower, freq_upper)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for stored_method, token in rows:
            if not token or not matches(stored_method):
                continue
            try:
                stages.update(split_stage_token(token))
            except ValueError:
                LOG.debug("Unparseable processing_status token %r; ignored",
                          token)
        if stages:
            return (stages, 'processing_status (scope-wide)')

    if _table_exists(conn, 'detection_runs'):
        import ast
        stages = set()
        for stored_method, stages_repr in conn.execute(
                "SELECT method, stages FROM detection_runs WHERE event_type IS ?",
                (event_type,)):
            if not stages_repr or not matches(stored_method):
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
            return (stages, 'detection_runs')

    return (set(), None)


def _observed_by_channel(conn, scope):
    """``{channel: {token: n_rows}}`` for one scope, NULL stages under ``None``."""
    out = {}
    for channel, token, n in conn.execute(
            "SELECT channel, stage, COUNT(*) FROM events "
            "WHERE event_type IS ? AND method IS ? AND freq_lower IS ? "
            "  AND freq_upper IS ? GROUP BY channel, stage",
            (scope['event_type'], scope['method'], scope['freq_lower'],
             scope['freq_upper'])):
        out.setdefault(str(channel), {})[token] = n
    return out


def plan_scope(conn, scope, override_token=None, rewrite_null=True):
    """Plan one scope as one or more per-channel target GROUPS.

    A "scope" (event type, method, band) is not necessarily one run: the same
    detector can have been run over ``['NREM2']`` on one channel and
    ``['NREM2', 'NREM3']`` on another. Those need different tokens, and
    ``processing_status`` knows which is which. So the unit of planning is a
    group of channels sharing a target, not the scope.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    scope : dict
        One entry from :func:`scopes_with_stages`.
    override_token : str or None, optional
        ``--stage-token`` value, applied to every channel when given. Default
        ``None`` (derive it per channel).
    rewrite_null : bool, optional
        Whether NULL-stage rows are relabelled with their group's target.
        Default ``True``; see the note below.

    Returns
    -------
    dict
        ``scope`` plus ``groups`` (a list of
        ``{'target', 'channels', 'source', 'n_to_change', 'n_rows',
        'n_null'}``, where ``channels`` is ``None`` for a scope-wide group),
        ``n_to_change``, ``n_null`` and ``status``
        (``'noop'`` / ``'rewrite'`` / ``'skip'``).

    Notes
    -----
    **NULL stages are rewritten by default, and that is a deliberate reversal
    of what this script did first.** The reasoning:

    * Under the pre-4.3 convention ``events.stage`` was the event's own scored
      epoch, so NULL meant "the epoch could not be resolved" and inventing a
      value would have been a confident, wrong label.
    * Under the 4.3 convention ``events.stage`` is the **run's stage scope**.
      Every row in the scope was detected inside the segments fetched for that
      stage set, whether or not its epoch lookup succeeded, so the run's token
      is the correct label for it. The per-epoch uncertainty is preserved
      exactly where it belongs -- ``events.epoch_stage`` stays NULL and says
      so.
    * Leaving them NULL is not neutral, it is a trap: the duplicate guard
      (:func:`turtlewave_hdEEG.dbwrite.assert_stage_format_compatible`) treats
      NULL as a different token, so a surviving NULL row refuses every future
      re-detection of that scope -- while this script stamps the marker and
      reports success. That closed loop is worse than either label.

    A NULL row is still never given an INVENTED token: a group whose target
    could not be derived (no ``processing_status`` record, no
    ``detection_runs`` row, and no non-NULL stage to take a union over) is
    reported as ``'skip'`` and left alone, and ``--stage-token`` is the way to
    resolve it. ``--keep-null-stage`` restores the conservative behaviour, in
    which case the script reports honestly that the database is NOT unblocked.
    """
    plan = dict(scope)
    plan['n_null'] = scope['tokens'].get(None, 0)
    observed = _observed_by_channel(conn, scope)

    def _target_from_observed(channels):
        components = {c for ch in channels
                      for t in observed.get(ch, {})
                      if t is not None
                      for c in stage_components(t)}
        return join_stage_token(components)

    if override_token is not None:
        # Accept 'NREM2NREM3', 'NREM2,NREM3' and 'NREM2 NREM3' alike; a comma
        # left in the token would otherwise become an unknown stage label and
        # be written verbatim into every row.
        import re
        parts = [s for s in re.split(r'[,\s]+', str(override_token)) if s]
        groups = [{'target': join_stage_token(parts), 'channels': None,
                   'source': '--stage-token'}]
    else:
        by_chan = recorded_stage_by_channel(
            conn, scope['event_type'], scope['method'], scope['freq_lower'],
            scope['freq_upper'])
        # Channels that hold events but whose stage set is not on record.
        unrecorded = sorted(set(observed) - set(by_chan))
        targets = {}
        for channel in sorted(observed):
            if channel not in by_chan:
                continue
            target = by_chan[channel]
            # The channel's own events must fit inside its recorded set,
            # otherwise the record does not describe this data.
            components = {c for t in observed[channel] if t is not None
                          for c in stage_components(t)}
            if not components.issubset(set(stage_components(target))):
                LOG.warning(
                    "Channel %s records searching %r for %s/%s but holds "
                    "events under stages %s, which is not a subset. Planning "
                    "it from its own events instead.", channel, target,
                    scope['event_type'], scope['method'], sorted(components))
                unrecorded.append(channel)
                continue
            targets.setdefault(target, []).append(channel)

        groups = [{'target': t, 'channels': chans,
                   'source': 'processing_status (per channel)'}
                  for t, chans in sorted(targets.items())]
        if unrecorded:
            recorded, source = recorded_stage_set(
                conn, scope['event_type'], scope['method'],
                scope['freq_lower'], scope['freq_upper'])
            fallback_components = {
                c for ch in unrecorded for t in observed.get(ch, {})
                if t is not None for c in stage_components(t)}
            if recorded and fallback_components.issubset(recorded):
                target, source = join_stage_token(recorded), source
            else:
                target = _target_from_observed(unrecorded)
                source = 'union of the stages these channels hold'
            groups.append({'target': target, 'channels': sorted(unrecorded),
                           'source': source})
            LOG.warning(
                "%d channel(s) in %s/%s have no per-channel record of the "
                "stage set they were searched over (%s). Their target is "
                "derived from %s. Check it, or pass --stage-token.",
                len(unrecorded), scope['event_type'], scope['method'],
                sorted(unrecorded)[:10], source)

        # Fast path: one target for the whole scope and no channel left out.
        if (len(groups) == 1 and groups[0]['channels'] is not None
                and set(groups[0]['channels']) == set(observed)):
            groups[0]['channels'] = None

    if len({g['target'] for g in groups}) > 1:
        LOG.warning(
            "Scope %s/%s/%s-%sHz needs %d DIFFERENT stage tokens (%s): its "
            "channels were not all searched over the same stage set. Each "
            "group is rewritten separately -- collapsing them to one token "
            "would divide the narrower group's events by the wider group's "
            "analysed time.", scope['event_type'], scope['method'],
            scope['freq_lower'], scope['freq_upper'], len(groups),
            {g['target']: (len(g['channels']) if g['channels'] else 'all')
             for g in groups})

    total_change = 0
    for g in groups:
        channels = (sorted(observed) if g['channels'] is None
                    else g['channels'])
        g['n_rows'] = sum(n for ch in channels
                          for n in observed.get(ch, {}).values())
        g['n_null'] = sum(n for ch in channels
                          for t, n in observed.get(ch, {}).items()
                          if t is None)
        g['n_to_change'] = sum(
            n for ch in channels
            for t, n in observed.get(ch, {}).items()
            if (t is None if rewrite_null else False) or
            (t is not None and t != g['target']))
        # A NULL row is only rewritten when a target could be derived; it is
        # never given an invented token.
        if not g['target']:
            g['n_to_change'] = 0
        total_change += g['n_to_change']

    plan['groups'] = groups
    plan['n_to_change'] = total_change
    plan['target'] = "/".join(sorted({g['target'] for g in groups}))
    plan['source'] = "; ".join(sorted({g['source'] for g in groups}))
    if not all(g['target'] for g in groups):
        plan['status'] = 'skip'
    elif total_change == 0:
        plan['status'] = 'noop'
    else:
        plan['status'] = 'rewrite'
    return plan


def _group_predicate(scope, group):
    """SQL fragment + params selecting one group's rows within a scope.

    Every scope component is matched with ``IS``, not ``=``. The scopes come
    from a ``GROUP BY``, so any of them can be NULL, and ``= NULL`` is never
    true: such a scope would select no rows, be planned as a no-op with an
    empty group list, and then be reported as unblocked by
    :func:`remaining_blockers` iterating that same empty list -- so the marker
    would be stamped over orphan rows the guard will refuse forever.
    """
    where = ["event_type IS ?", "method IS ?", "freq_lower IS ?",
             "freq_upper IS ?"]
    params = [scope['event_type'], scope['method'], scope['freq_lower'],
              scope['freq_upper']]
    if group['channels'] is not None:
        where.append("channel IN (%s)" % ",".join("?" * len(group['channels'])))
        params.extend(group['channels'])
    return " AND ".join(where), params


def collisions(conn, scope, group, rewrite_null=True):
    """Rows that would violate ``event_chan_time`` once the rewrite lands.

    The ``events`` UNIQUE constraint is
    ``(event_type, channel, start_time, method, freq_lower, freq_upper,
    stage)``. Within one group the first six components are pinned by the
    predicate, so after the rewrite -- when every row in the group carries the
    same ``target`` -- the constraint reduces to: **at most one row per
    ``(channel, start_time)``**. Anything more is a collision and the UPDATE
    fails mid-transaction. One consistent detection run cannot produce such a
    pair (an event has one start time), but a database re-imported after
    re-scoring, or one that accumulated two runs, can.

    The row set is everything whose stage **would be the target afterwards** --
    the rows that move AND the rows already sitting on it. Excluding the
    latter was a blind spot with teeth: the already-at-target row is precisely
    the one a moving row lands on top of, so the pair was never counted, the
    dry run reported a clean bill of health, and ``--apply`` died with an
    unhandled ``IntegrityError`` after the backup had been written. It became
    reachable when NULL rows started moving.

    Checked per GROUP, not per scope: two channels being collapsed to
    different tokens cannot collide with each other, and a scope-wide check
    could not say which group a collision belonged to.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    scope : dict
        Scope entry.
    group : dict
        One entry of ``plan['groups']``.
    rewrite_null : bool, optional
        Whether NULL-stage rows are part of the rewrite. When False they keep
        their NULL, which SQLite treats as distinct in a UNIQUE index, so they
        cannot collide and are excluded from the row set. Default ``True``.

    Returns
    -------
    list of tuple
        ``(channel, start_time, n_rows_landing_there)`` for each colliding
        key, at most 20.
    """
    clause, params = _group_predicate(scope, group)
    # Rows that will carry `target` after the rewrite: the movers, plus the
    # ones already there. With rewrite_null=True that is every row in the
    # group; with it False, every non-NULL row.
    if rewrite_null:
        landing, extra = "1 = 1", []
    else:
        landing, extra = "stage IS NOT NULL", []
    return conn.execute(
        "SELECT channel, start_time, COUNT(*) AS n FROM events "
        "WHERE " + clause + " AND " + landing + " "
        "GROUP BY channel, start_time HAVING COUNT(*) > 1 LIMIT 20",
        params + extra).fetchall()


def predicted_blockers(conn, plans, rewrite_null):
    """Scopes the planned rewrite will leave blocked, computed BEFORE writing.

    The dry run has to be able to say "this will not fully unblock your
    database" before the user commits to it, and the read-only phase has to be
    able to refuse the "nothing to do" shortcut for the same reason. Two
    situations produce a blocker, both visible from the plan alone:

    * a group whose target could not be derived (nothing is rewritten, and
      every row in it will keep refusing);
    * NULL-stage rows under ``--keep-null-stage`` (deliberately left, and they
      keep refusing).

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection.
    plans : list of dict
        Plans from :func:`plan_scope`.
    rewrite_null : bool
        Whether NULL rows are part of the rewrite.

    Returns
    -------
    list of dict
        Same shape as :func:`remaining_blockers`.
    """
    out = []
    for p in plans:
        for g in p['groups']:
            if g['target'] and (rewrite_null or not g['n_null']):
                continue
            clause, params = _group_predicate(p, g)
            if g['target']:
                where, gparams = clause + " AND stage IS NULL", list(params)
            else:
                where, gparams = clause, list(params)
            rows = conn.execute(
                "SELECT COUNT(*) FROM events WHERE " + where,
                gparams).fetchone()
            if not rows or not rows[0]:
                continue
            raw = [r[0] for r in conn.execute(
                "SELECT DISTINCT stage FROM events WHERE " + where, gparams)]
            tokens = sorted(str(r) for r in raw if r is not None)
            if any(r is None for r in raw):
                tokens.append('NULL (no stage at all)')
            channels = [str(r[0]) for r in conn.execute(
                "SELECT DISTINCT channel FROM events WHERE " + where
                + " ORDER BY channel", gparams)]
            out.append({
                'event_type': p['event_type'], 'method': p['method'],
                'freq_lower': p['freq_lower'], 'freq_upper': p['freq_upper'],
                'tokens': tokens, 'channels': channels,
                'n_rows': int(rows[0]),
            })
    return out


def remaining_blockers(conn, plans):
    """Scopes that will STILL refuse re-detection after the rewrite.

    The check behind this script's only real promise. ``events.stage`` is part
    of both ``event_uuid5`` and the ``event_chan_time`` UNIQUE constraint, so
    :func:`turtlewave_hdEEG.dbwrite.assert_stage_format_compatible` refuses any
    scope holding a token other than the one the next run will write -- and it
    counts NULL as "other", by design, because a NULL-stage row duplicates in
    exactly the same way. So a surviving NULL row, or a group whose target
    could not be derived, leaves that scope permanently blocked. Claiming
    success there sends the user into a loop: the migration says done, the
    detector says run the migration.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection, AFTER the rewrite.
    plans : list of dict
        The plans from :func:`plan_scope`.

    Returns
    -------
    list of dict
        One entry per still-blocked group, with ``event_type``, ``method``,
        ``freq_lower``, ``freq_upper``, ``tokens`` (the offending stage values,
        NULL rendered as a readable label), ``channels`` and ``n_rows``.
        Empty when every scope is genuinely unblocked.
    """
    out = []
    for p in plans:
        for g in p['groups']:
            clause, params = _group_predicate(p, g)
            if g['target']:
                where = clause + " AND (stage IS NULL OR stage != ?)"
                gparams = params + [g['target']]
            else:
                # No target could be derived, so EVERY row here is a blocker:
                # the next run's token cannot match a scope nothing describes.
                where = clause
                gparams = list(params)
            rows = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT channel) FROM events WHERE "
                + where, gparams).fetchone()
            if not rows or not rows[0]:
                continue
            raw = [r[0] for r in conn.execute(
                "SELECT DISTINCT stage FROM events WHERE " + where, gparams)]
            tokens = sorted(str(r) for r in raw if r is not None)
            if any(r is None for r in raw):
                tokens.append('NULL (no stage at all)')
            channels = [str(r[0]) for r in conn.execute(
                "SELECT DISTINCT channel FROM events WHERE " + where
                + " ORDER BY channel", gparams)]
            out.append({
                'event_type': p['event_type'], 'method': p['method'],
                'freq_lower': p['freq_lower'], 'freq_upper': p['freq_upper'],
                'tokens': tokens, 'channels': channels,
                'n_rows': int(rows[0]),
            })
    return out


def make_backup(db_path, backup_path=None):
    """Copy the database with :meth:`sqlite3.Connection.backup`.

    WAL-safe by construction: it copies *committed database content* through
    SQLite rather than copying the file, so a database whose latest
    transactions still live in a ``-wal`` sidecar is backed up complete. A
    plain file copy of the ``.db`` alone would silently lose them.

    Parameters
    ----------
    db_path : str
        Database to back up.
    backup_path : str or None, optional
        Destination. Default ``<db_path>.pre-joint-stage.bak``.

    Returns
    -------
    str
        Path of the backup written.

    Raises
    ------
    FileExistsError
        If the destination already exists. **This refusal is the point of the
        backup.** A second run would otherwise overwrite the good
        pre-migration copy with an already-migrated one -- the single way this
        script could destroy the thing it exists to protect. Move or rename
        the existing backup deliberately if you really mean to replace it.
    """
    if backup_path is None:
        backup_path = os.path.abspath(db_path) + BACKUP_SUFFIX
    if os.path.exists(backup_path):
        raise FileExistsError(
            f"A backup already exists at {backup_path}. Refusing to overwrite "
            f"it: if this database has already been migrated, that file is "
            f"the only pre-migration copy and replacing it with a migrated "
            f"one would destroy the only way back. Move or rename it first, "
            f"or pass --backup-path to write elsewhere.")
    src = sqlite3.connect(db_path, timeout=60.0)
    dst = sqlite3.connect(backup_path, timeout=60.0)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    size = os.path.getsize(backup_path)
    LOG.info("Backup written: %s (%.1f MB)", backup_path, size / 1e6)
    return backup_path


def backfill_analysed_time(conn, db_path, annot_path, dataset_path=None,
                           stage_tokens=None):
    """Back-fill the density denominator from the scoring.

    ``analysed_time`` holds the artefact-free in-stage seconds actually fed to
    the detector, per single scored stage -- the one quantity density cannot
    be derived from ``events``. A stage collapse leaves it exactly as empty as
    it was, so density stays unavailable until this runs.

    The rejection settings come from ``detection_runs``, not from a default:
    they are part of the ``analysed_time`` key, and a denominator computed
    under settings the run did not use is a wrong denominator, not a missing
    one. One row set is written per distinct ``(reject_artifacts,
    reject_arousals)`` pair on record.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection.
    db_path : str
        Database path (used to derive the subject).
    annot_path : str
        Scoring XML to compute the denominator from.
    dataset_path : str or None, optional
        Recording, read only for ``header['s_freq']`` (which refines the
        two-sample segment floor). Default ``None``.
    stage_tokens : iterable of str or None, optional
        Tokens present in ``events`` after the collapse; their components are
        the stages a denominator is needed for. Default ``None`` (derive from
        the database).

    Returns
    -------
    dict
        ``{(reject_artifacts, reject_arousals): {stage: seconds}}`` written.
    """
    from turtlewave_hdEEG import CustomAnnotations
    from turtlewave_hdEEG.utils import derive_subject

    annotations = CustomAnnotations(annot_path)
    dataset = None
    if dataset_path:
        try:
            from wonambi import Dataset
            dataset = Dataset(dataset_path)
        except Exception as e:
            LOG.warning("Could not open --dataset %s (%s); continuing without "
                        "it (only s_freq is read from it).", dataset_path, e)

    subject = derive_subject(annotation_path=annot_path,
                             root_dir=dbwrite.recording_root_from_db(db_path))

    if stage_tokens is None:
        stage_tokens = [r[0] for r in conn.execute(
            "SELECT DISTINCT stage FROM events WHERE stage IS NOT NULL")]
    stages = sorted({c for t in stage_tokens for c in stage_components(t)})
    if not stages:
        LOG.warning("No stages in events, so no denominator was computed.")
        return {}

    settings = set()
    if _table_exists(conn, 'detection_runs'):
        for ra, ro in conn.execute(
                "SELECT DISTINCT reject_artifacts, reject_arousals "
                "FROM detection_runs"):
            if ra is None or ro is None:
                continue
            settings.add((bool(ra), bool(ro)))
    if not settings:
        LOG.warning(
            "detection_runs records no artefact/arousal rejection settings, "
            "so the denominator is computed for the detector defaults "
            "(reject_artifacts=True, reject_arousals=True). If those runs used "
            "different settings, density will report the denominator as "
            "missing rather than give a wrong number.")
        settings = {(True, True)}

    written = {}
    for reject_artifacts, reject_arousals in sorted(settings):
        # strict=True: store_analysed_time swallows its own failure by
        # default so a completed DETECTION is never lost to a denominator
        # problem. This caller has nothing else to lose -- the denominator is
        # the entire job -- so the failure must reach the exit code instead of
        # being logged and returning {}.
        rows = dbwrite.store_analysed_time(
            conn, subject, annotations, dataset, stages,
            reject_artifacts, reject_arousals, source='backfill',
            annotation_file=annot_path, logger=LOG, strict=True)
        written[(reject_artifacts, reject_arousals)] = rows
    return written


def backfill_cycles(conn, db_path, annot_path, methods=('2022', '1979'),
                    tag_method='2022'):
    """Back-fill ``sleep_cycles``, ``stage_durations`` and ``events.cycle``.

    Both cycle definitions are stored, keyed by ``(subject, method)``;
    ``tag_method`` owns ``events.cycle``. The annotation XML is **not**
    written (``write_xml=False``): this script never modifies the rater's
    file.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open write connection, reused so no second writer is opened.
    db_path : str
        Database path (used to derive the subject).
    annot_path : str
        Scoring XML.
    methods : sequence of str, optional
        Cycle definitions to store. Default ``('2022', '1979')``.
    tag_method : str, optional
        Definition owning ``events.cycle``. Default ``'2022'``.

    Returns
    -------
    dict
        ``{method: [cycle dicts]}``.
    """
    from turtlewave_hdEEG import CustomAnnotations
    from turtlewave_hdEEG.cycleprocessor import finalize_cycles_and_durations
    from turtlewave_hdEEG.utils import derive_subject

    annotations = CustomAnnotations(annot_path)
    subject = derive_subject(annotation_path=annot_path,
                             root_dir=dbwrite.recording_root_from_db(db_path))
    # run_id=None: a backfill tags the whole table, which is what it is for.
    return finalize_cycles_and_durations(
        annotations, db_path, subject=subject, methods=tuple(methods),
        tag_method=tag_method, write_xml=False, plot=False, conn=conn,
        run_id=None, tag_events=True)


def build_parser():
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        description=("Collapse events.stage to the run's joint stage token "
                     "and stamp db_meta.stage_format='joint'. Dry run unless "
                     "--apply is given."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Reversal is the backup file. Un-collapsing a joint token "
                "without re-reading the hypnogram is impossible."))
    p.add_argument('db_path', help="Path to neural_events.db")
    p.add_argument('--apply', action='store_true',
                   help="Actually write. Without it nothing is modified.")
    p.add_argument('--allow-archive', action='store_true',
                   help="Permit a path containing 'archive' (refused by "
                        "default: archived databases are the irreplaceable "
                        "copies).")
    p.add_argument('--stage-token', default=None,
                   help="Force this stage set as the target for EVERY scope "
                        "(e.g. NREM2NREM3 or 'NREM2,NREM3'), instead of "
                        "deriving it per scope from the run's record.")
    p.add_argument('--backup-path', default=None,
                   help=f"Backup destination (default: "
                        f"<db_path>{BACKUP_SUFFIX}). Never overwritten.")
    p.add_argument('--keep-null-stage', action='store_true',
                   help="Leave NULL-stage rows alone instead of labelling "
                        "them with their run's stage token. They will keep "
                        "REFUSING every re-detection of their scope (the "
                        "duplicate guard treats NULL as a different token), "
                        "so the script will report the database as NOT "
                        "unblocked and exit 3.")
    p.add_argument('--skip-integrity-check', action='store_true',
                   help="Skip PRAGMA integrity_check (it is slow on a large "
                        "database on a network share).")
    p.add_argument('--annot', default=None,
                   help="Scoring XML, required for the back-fills.")
    p.add_argument('--dataset', default=None,
                   help="Recording file; only header['s_freq'] is read from "
                        "it, to refine the analysed-time segment floor.")
    bt = p.add_mutually_exclusive_group()
    bt.add_argument('--backfill-analysed-time', dest='backfill_time',
                    action='store_true', default=None,
                    help="Force the density-denominator back-fill on "
                         "(default: on when analysed_time is empty).")
    bt.add_argument('--no-backfill-analysed-time', dest='backfill_time',
                    action='store_false',
                    help="Never back-fill analysed_time.")
    bc = p.add_mutually_exclusive_group()
    bc.add_argument('--backfill-cycles', dest='backfill_cycles',
                    action='store_true', default=None,
                    help="Force the cycle back-fill on (default: on when "
                         "sleep_cycles is empty).")
    bc.add_argument('--no-backfill-cycles', dest='backfill_cycles',
                    action='store_false',
                    help="Never back-fill cycles.")
    p.add_argument('--verbose', action='store_true', help="Debug logging.")
    return p


def main(argv=None):
    """Run the migration (or the dry run).

    Parameters
    ----------
    argv : list of str or None, optional
        Command line. Default ``None`` (``sys.argv[1:]``).

    Returns
    -------
    int
        ``0`` -- the database is fully migrated (or a dry run says it would
        be). ``1`` -- refused before writing anything (archive guard, failed
        integrity check, collision). ``2`` -- a post-write assertion failed;
        the database is as SQLite committed it and the backup is the recovery.
        ``3`` -- the rewrite is committed but the job is NOT finished: at
        least one scope will still refuse re-detection, or a requested
        back-fill did not complete. The log names exactly which, and what was
        committed. **``0`` means everything asked for succeeded and the
        database is fully re-detectable, and nothing less returns 0** -- a
        batch driver marking a subject done needs one rule it cannot get
        wrong.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s %(message)s')

    db_path = os.path.abspath(os.path.expanduser(args.db_path))

    # --- guard 1: archived databases -------------------------------------
    if 'archive' in db_path.lower() and not args.allow_archive:
        LOG.error(
            "Refusing to touch %s: its path contains 'archive'. Archived "
            "databases are the copies that cannot be regenerated. Pass "
            "--allow-archive if you are certain.", db_path)
        return 1

    if not os.path.isfile(db_path):
        LOG.error("No database at %s.", db_path)
        return 1

    # --- guard: --stage-token must be a token a DETECTOR could write ------
    # Otherwise the whole run is quietly pointless: the rows get a label the
    # detector never produces, so the duplicate guard keeps refusing, while
    # this script stamps the marker and reports success. remaining_blockers
    # cannot catch it -- it compares each row against the plan's own target,
    # which is the wrong thing by construction here.
    if args.stage_token is not None:
        import re as _re
        parts = [s for s in _re.split(r'[,\s]+', str(args.stage_token)) if s]
        token = join_stage_token(parts)
        try:
            split_stage_token(token)
        except ValueError as e:
            LOG.error(
                "--stage-token %r is not a stage set this pipeline can "
                "produce (%s). A detector writes tokens built from those "
                "stage names, so every row this run wrote would still be "
                "refused by the duplicate guard -- while the marker claimed "
                "the database was migrated. Spell the stages out, e.g. "
                "--stage-token 'NREM2,NREM3' or --stage-token NREM2NREM3.",
                args.stage_token, e)
            return 1
        if token != str(args.stage_token):
            LOG.info(
                "--stage-token %r canonicalised to %r (the spelling a "
                "detector writes).", args.stage_token, token)

    # --- guard 2: integrity ----------------------------------------------
    conn = _connect(db_path, read_only=True)
    try:
        if not args.skip_integrity_check:
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if str(result).lower() != 'ok':
                LOG.error(
                    "PRAGMA integrity_check on %s returned %r, not 'ok'. "
                    "Refusing to migrate a database SQLite reports as "
                    "damaged; restore it from a backup first.", db_path,
                    result)
                return 1
            LOG.info("integrity_check: ok")

        if not _table_exists(conn, 'events'):
            LOG.error("%s has no events table; nothing to migrate.", db_path)
            return 1

        marker = dbwrite.get_db_meta(conn, STAGE_FORMAT_KEY, None)
        LOG.info("db_meta.%s = %r", STAGE_FORMAT_KEY, marker)

        # --- plan, per scope ---------------------------------------------
        rewrite_null = not args.keep_null_stage
        plans = [plan_scope(conn, s, args.stage_token,
                            rewrite_null=rewrite_null)
                 for s in scopes_with_stages(conn)]
        # The column list is pinned here and passed to the post-write call:
        # ensure_direct_write_schema adds columns in between, and re-deriving
        # the list would compare different shapes and never match.
        pre_rows, pre_digest, pre_columns = digest_excluding_stage(conn)

        LOG.info("%s holds %d event row(s) across %d detection scope(s):",
                 db_path, pre_rows, len(plans))
        for p in plans:
            tokens = ", ".join(
                f"{'NULL' if t is None else t}={n}"
                for t, n in sorted(p['tokens'].items(),
                                   key=lambda kv: str(kv[0])))
            LOG.info(
                "  [%s] %s / %s / %s-%sHz: %d row(s) {%s}",
                p['status'], p['event_type'], p['method'], p['freq_lower'],
                p['freq_upper'], p['n_rows'], tokens)
            for g in p['groups']:
                if not g['target']:
                    null_note = (" -- NO TARGET could be derived, so these "
                                 "rows are left alone; pass --stage-token")
                elif g['n_null'] and rewrite_null:
                    null_note = (f", including {g['n_null']} NULL-stage row(s) "
                                 f"labelled with this token")
                elif g['n_null']:
                    null_note = (f"; {g['n_null']} NULL-stage row(s) KEPT as "
                                 f"NULL (--keep-null-stage) and will keep "
                                 f"refusing re-detection")
                else:
                    null_note = ""
                LOG.info(
                    "      -> %r for %s (%d of %d row(s) to change; target "
                    "from %s)%s", g['target'],
                    'every channel' if g['channels'] is None
                    else f"{len(g['channels'])} channel(s) "
                         f"{g['channels'][:6]}",
                    g['n_to_change'], g['n_rows'], g['source'], null_note)

        # --- guard: collisions -------------------------------------------
        blocked = False
        for p in plans:
            if p['status'] != 'rewrite':
                continue
            for g in p['groups']:
                if not g['n_to_change']:
                    continue
                hits = collisions(conn, p, g, rewrite_null=rewrite_null)
                if hits:
                    blocked = True
                    LOG.error(
                        "Collision in %s / %s / %s-%sHz (target %r): %d key(s) "
                        "like %s carry more than one stage. Collapsing them "
                        "would make two rows identical under the "
                        "event_chan_time UNIQUE constraint and the UPDATE "
                        "would fail mid-transaction. This means the same event "
                        "is stored twice under different scorings; resolve it "
                        "(or re-detect this scope from scratch) before "
                        "migrating.", p['event_type'], p['method'],
                        p['freq_lower'], p['freq_upper'], g['target'],
                        len(hits), hits[:3])
        if blocked:
            LOG.error("No changes made.")
            return 1

        to_rewrite = [p for p in plans if p['status'] == 'rewrite']
        n_change = sum(p['n_to_change'] for p in to_rewrite)
        already_joint = marker == STAGE_FORMAT_JOINT
        # What this run will NOT fix, known before writing anything.
        will_block = predicted_blockers(conn, plans, rewrite_null)

        # Decide the back-fills BEFORE closing the read connection.
        at_empty = (not _table_exists(conn, 'analysed_time')
                    or conn.execute(
                        "SELECT COUNT(*) FROM analysed_time").fetchone()[0] == 0)
        cyc_empty = (not _table_exists(conn, 'sleep_cycles')
                     or conn.execute(
                         "SELECT COUNT(*) FROM sleep_cycles").fetchone()[0] == 0)
    finally:
        conn.close()

    do_time = at_empty if args.backfill_time is None else args.backfill_time
    do_cycles = cyc_empty if args.backfill_cycles is None else args.backfill_cycles

    # An EXPLICIT --backfill-* without --annot is a request that can never be
    # honoured, so it fails here rather than being downgraded to a no-op. The
    # earlier behaviour warned and cleared both flags, after which every path
    # including `return 0` ran as though nothing had been asked for -- so a
    # batch driver whose --annot glob came back empty marked the subject done
    # with analysed_time still empty, and that subject then dropped out of
    # every density comparison for good. Exit 0 has to keep meaning that
    # everything asked for succeeded.
    if not args.annot and (args.backfill_time is True
                           or args.backfill_cycles is True):
        asked = ', '.join(
            flag for flag, on in (('--backfill-analysed-time',
                                   args.backfill_time is True),
                                  ('--backfill-cycles',
                                   args.backfill_cycles is True)) if on)
        LOG.error(
            "%s was requested but --annot was not given, and a back-fill "
            "cannot run without the scoring file. Nothing has been written. "
            "Re-run with --annot <scoring.xml>, or drop the flag to migrate "
            "the stage tokens alone.", asked)
        return 1

    if (do_time or do_cycles) and not args.annot:
        # Neither flag was given: the back-fill only defaulted on because the
        # table is empty. Nothing was asked for, so warn and carry on.
        LOG.warning(
            "analysed_time is %s and sleep_cycles is %s, but --annot was not "
            "given, so neither can be back-filled. Density and cycles will "
            "stay unavailable. Re-run with --annot <scoring.xml> to fill "
            "them.", 'empty' if at_empty else 'populated',
            'empty' if cyc_empty else 'populated')
        do_time = do_cycles = False

    def _report_blockers(blockers, tense):
        """Log one line per scope that is (or will be) still refusing."""
        for b in blockers:
            LOG.error(
                "%s: %s / %s / %s-%sHz holds %d row(s) under %s on "
                "channel(s) %s. Re-detecting that scope %s REFUSED -- the "
                "duplicate guard treats these as a different stage token. "
                "Resolve with --stage-token (when no stage set is on record), "
                "by re-running without --keep-null-stage, by re-detecting "
                "those channels with replace_channels=[...], or by deleting "
                "those rows.",
                'NOT unblocked' if tense == 'is' else 'WILL NOT be unblocked',
                b['event_type'], b['method'], b['freq_lower'],
                b['freq_upper'], b['n_rows'], b['tokens'], b['channels'][:10],
                'is' if tense == 'is' else 'will be')

    if not n_change and already_joint and not do_time and not do_cycles:
        if will_block:
            _report_blockers(will_block, 'is')
            LOG.error(
                "Nothing to rewrite, but this database is NOT fully migrated: "
                "%d scope(s) above will keep refusing re-detection. The marker "
                "already says '%s', which is why nothing here can fix them "
                "automatically.", len(will_block), STAGE_FORMAT_JOINT)
            return 3
        LOG.info(
            "Nothing to do: every scope already carries its joint token and "
            "db_meta.%s is already '%s'.", STAGE_FORMAT_KEY,
            STAGE_FORMAT_JOINT)
        return 0

    if not n_change:
        # The CSV-imported case: spindle rows already carry the joined token
        # because the importer flattened the requested stage list into the
        # same form. Nothing to rewrite -- the marker alone unblocks
        # re-detection, and refusing to notice that would make this script a
        # blocker for every 4.0.x database rather than a formality.
        LOG.info(
            "No stage value needs rewriting: every scope's rows already carry "
            "the token this release would write. Only db_meta.%s (and any "
            "requested back-fill) will change.", STAGE_FORMAT_KEY)

    if not args.apply:
        LOG.info(
            "DRY RUN. Would rewrite %d stage value(s) across %d scope(s)%s%s. "
            "Re-run with --apply to write; a backup is taken first at %s.",
            n_change, len(to_rewrite),
            ", back-fill analysed_time" if do_time else "",
            ", back-fill cycles + stage durations" if do_cycles else "",
            args.backup_path or (db_path + BACKUP_SUFFIX))
        if will_block:
            _report_blockers(will_block, 'will')
            LOG.error(
                "DRY RUN verdict: this would NOT fully unblock the database "
                "-- %d scope(s) above would still refuse re-detection, and "
                "db_meta.%s would NOT be stamped.", len(will_block),
                STAGE_FORMAT_KEY)
            return 3
        LOG.info("DRY RUN verdict: this would leave every scope re-detectable "
                 "and would stamp db_meta.%s='%s'.",
                 STAGE_FORMAT_KEY, STAGE_FORMAT_JOINT)
        return 0

    # A run with nothing to write that would still leave the database blocked
    # stops HERE, before the backup. Otherwise it would leave a backup file
    # behind, and the re-run the message asks for (with --stage-token) would
    # then be refused by the backup guard -- the same closed loop this section
    # exists to remove, one step longer.
    if not n_change and will_block and not do_time and not do_cycles:
        _report_blockers(will_block, 'is')
        LOG.error(
            "Refusing to write: this run would change no stage value and "
            "would leave %d scope(s) refusing re-detection, so there is "
            "nothing to gain and the marker must not be stamped. No backup "
            "was taken, so re-running (e.g. with --stage-token) needs no "
            "cleanup.", len(will_block))
        return 3

    # --- backup, then write ----------------------------------------------
    try:
        backup = make_backup(db_path, args.backup_path)
    except FileExistsError as e:
        LOG.error("%s", e)
        return 1

    conn = _connect(db_path, read_only=False)
    try:
        # Bring the schema to current (adds epoch_stage, db_meta, the density
        # view) BEFORE the rewrite, so the marker has a table to live in.
        dbwrite.ensure_direct_write_schema(conn, LOG)

        conn.execute('BEGIN')
        try:
            changed = 0
            for p in to_rewrite:
                for g in p['groups']:
                    if not g['n_to_change']:
                        continue
                    clause, gparams = _group_predicate(p, g)
                    # NULL rows are included by default: events.stage is the
                    # RUN's scope from 4.3 on, and leaving them NULL leaves
                    # them refusing every future re-detection of this scope.
                    moving = ("(stage IS NULL OR stage != ?)" if rewrite_null
                              else "(stage IS NOT NULL AND stage != ?)")
                    cur = conn.execute(
                        "UPDATE events SET stage = ? WHERE " + clause
                        + " AND " + moving,
                        [g['target']] + gparams + [g['target']])
                    changed += cur.rowcount
                    LOG.info(
                        "  %s / %s / %s-%sHz [%s]: %d row(s) -> %r",
                        p['event_type'], p['method'], p['freq_lower'],
                        p['freq_upper'],
                        'all channels' if g['channels'] is None
                        else f"{len(g['channels'])} channel(s)",
                        cur.rowcount, g['target'])
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        # --- post-write assertions ---------------------------------------
        # Same columns as the pre-digest, NOT whatever the table has now:
        # ensure_direct_write_schema added epoch_stage/det_*/run_id above.
        try:
            post_rows, post_digest, _ = digest_excluding_stage(
                conn, columns=pre_columns)
        except ValueError as e:
            LOG.error("%s Restore %s over %s.", e, backup, db_path)
            return 2
        recovery = (f"Restore {backup} over {db_path} (copy it over the "
                    f"database, then delete or rename the backup so a later "
                    f"--apply can take a fresh one).")
        if post_rows != pre_rows:
            LOG.error(
                "Row count changed during the migration: %d -> %d. %s",
                pre_rows, post_rows, recovery)
            return 2
        if post_digest != pre_digest:
            LOG.error(
                "A column other than 'stage' changed during the migration "
                "(digest %s -> %s over columns %s). %s", pre_digest,
                post_digest, pre_columns, recovery)
            return 2
        if changed != n_change:
            LOG.warning(
                "Planned %d row change(s) but the UPDATE reported %d. The "
                "database changed between the plan and the write.",
                n_change, changed)
        LOG.info(
            "Rewrote %d stage value(s). Row count %d unchanged; every column "
            "except 'stage' is byte-identical.", changed, post_rows)

        # --- back-fills, BEFORE the verdict --------------------------------
        # Order is load-bearing. These are work the user asked for explicitly,
        # they are independent of the stage token (analysed_time is keyed per
        # single scored stage; cycles are time windows), and the pre-write
        # shortcut above deliberately lets a run proceed *because* a back-fill
        # was requested. Running them after an early `return 3` would make
        # that shortcut's reasoning false and drop the requested work with
        # nothing in the log saying so.
        # Outcomes, not intentions. Every message and the exit code below are
        # derived from what these actually did -- the previous version keyed
        # its summary on whether a back-fill had been REQUESTED, so a run
        # whose back-fills both failed still reported that they "DID run".
        done = []        # human-readable, what succeeded
        failed = []      # human-readable, what did not
        if do_time:
            try:
                written = backfill_analysed_time(
                    conn, db_path, args.annot, args.dataset)
                # Count ROWS, not settings attempted. store_analysed_time
                # returns {} for a setting whose denominator could not be
                # computed, so len(written) counts attempts and would report
                # a total failure as "1 rejection setting(s)".
                n_rows = sum(len(v) for v in written.values())
                if n_rows:
                    done.append(f"analysed_time: {n_rows} row(s) across "
                                f"{len(written)} rejection setting(s)")
                    LOG.info(
                        "Back-filled analysed_time: %d row(s) across %d "
                        "rejection setting(s): %s", n_rows, len(written),
                        json.dumps({str(k): v for k, v in written.items()},
                                   default=str))
                else:
                    failed.append(
                        "analysed_time: no denominator row was written "
                        "(the computation produced nothing for any rejection "
                        "setting)")
            except Exception as e:
                failed.append(f"analysed_time: {type(e).__name__}: {e}")
                LOG.error("analysed_time back-fill FAILED: %s. The stage "
                          "migration itself is committed and unaffected.", e,
                          exc_info=True)
        if do_cycles:
            try:
                cycles = backfill_cycles(conn, db_path, args.annot)
                # A night can legitimately have zero cycles (short or heavily
                # fragmented), so the cycle count is not the pass/fail signal.
                # stage_durations is: finalize_cycles_and_durations writes it
                # whenever the hypnogram is readable at all.
                n_dur = conn.execute(
                    "SELECT COUNT(*) FROM stage_durations").fetchone()[0]
                n_cyc = sum(len(c) for c in cycles.values())
                if n_dur:
                    done.append(
                        f"cycles: {n_cyc} across {len(cycles)} definition(s), "
                        f"stage_durations: {n_dur} row(s)")
                    LOG.info(
                        "Back-filled cycles: %s, stage_durations: %d row(s) "
                        "(annotation XML NOT modified)",
                        {m: len(c) for m, c in cycles.items()}, n_dur)
                else:
                    failed.append(
                        "cycles: no stage_durations row was written, so the "
                        "hypnogram could not be read")
            except Exception as e:
                failed.append(f"cycles: {type(e).__name__}: {e}")
                LOG.error("Cycle back-fill FAILED: %s. The stage migration "
                          "itself is committed and unaffected.", e,
                          exc_info=True)

        # --- did this actually unblock re-detection? ----------------------
        # The one claim this script must never make falsely. The duplicate
        # guard refuses any scope holding a token other than the run's, and it
        # counts NULL as "other" -- so a surviving NULL row, or a scope whose
        # target could not be derived, leaves that scope refusing forever
        # while the marker says it is fine. Check rather than assume.
        blockers = remaining_blockers(conn, plans)

        # rc=3 covers BOTH "a scope still refuses" and "requested work did not
        # complete", deliberately rather than splitting them: the two demand
        # the same response (read the log, fix, re-run), and what a batch
        # driver needs is one rule it cannot get wrong --
        # **rc == 0 means everything asked for succeeded AND the database is
        # fully re-detectable**, nothing less. A failed back-fill returning 0
        # is what marks a subject done in examples/NCI_commands/.
        committed = f"the stage rewrite ({changed} row(s))"
        if done:
            committed += ", " + "; ".join(done)
        if blockers or failed:
            if blockers:
                _report_blockers(blockers, 'is')
            for problem in failed:
                LOG.error("Requested back-fill did NOT complete -- %s",
                          problem)
            # The marker is deliberately NOT stamped when a scope is blocked:
            # it would disable the pre-4.3 check on a database that is still
            # partly pre-4.3 and record as done something that is not, while
            # the token check keeps refusing anyway. A back-fill failure alone
            # does not block re-detection, so the marker IS stamped in that
            # case -- the stage migration genuinely succeeded and withholding
            # it would leave the user unable to re-detect for an unrelated
            # reason.
            if not blockers:
                dbwrite.set_db_meta(conn, STAGE_FORMAT_KEY,
                                    STAGE_FORMAT_JOINT)
                LOG.info(
                    "Stamped db_meta.%s = '%s': the stage migration itself "
                    "succeeded, so re-detection is allowed. The back-fill(s) "
                    "above did not complete and are what exit 3 refers to.",
                    STAGE_FORMAT_KEY, STAGE_FORMAT_JOINT)
            else:
                LOG.error(
                    "db_meta.%s was NOT stamped: %d scope(s) above are still "
                    "blocked, so this database has not been migrated in full.",
                    STAGE_FORMAT_KEY, len(blockers))
            LOG.error(
                "Committed and NOT undone: %s. NOT done: %s. Re-run after "
                "resolving the above (move or rename %s first, or pass "
                "--backup-path, since a backup already exists).",
                committed,
                "; ".join(
                    ([f"{len(blockers)} blocked scope(s)"] if blockers else [])
                    + failed) or 'nothing',
                backup)
            return 3

        dbwrite.set_db_meta(conn, STAGE_FORMAT_KEY, STAGE_FORMAT_JOINT)
        LOG.info("Stamped db_meta.%s = '%s'; re-detection into this database "
                 "is now allowed. Committed: %s.",
                 STAGE_FORMAT_KEY, STAGE_FORMAT_JOINT, committed)
    finally:
        conn.close()

    LOG.info("Done. Pre-migration backup: %s (the only way back).", backup)
    return 0


if __name__ == '__main__':
    sys.exit(main())
