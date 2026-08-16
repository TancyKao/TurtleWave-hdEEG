## test_db_journal.py
#
# Verifies the network-drive SQLite fix (TURTLEWAVE_SQLITE_JOURNAL,
# dbwrite.open_write_connection, dbwrite.set_journal_mode). No pytest; plain
# functions with prints + asserts, matching tests/test_turtlewave.py.
#
# open_write_connection's rule: a database it CREATES gets DELETE (network- and
# cloud-sync safe; WAL is what breaks on those filesystems); an EXISTING
# database's journal mode is preserved, never overridden, unless journal= or
# TURTLEWAVE_SQLITE_JOURNAL explicitly names a mode. set_journal_mode() is the
# one-time, sticky conversion -- once run, open_write_connection will not
# silently flip the database back.
#
# Tests that need a WAL database to convert build it explicitly with
# journal='WAL'; since the create default is DELETE, that fixture no longer
# falls out of a plain open_write_connection.
#
# Every test cleans up its own TURTLEWAVE_SQLITE_JOURNAL state (pop in a
# finally) so ordering between tests never matters.

import os
import shutil
import sqlite3
import tempfile
import time

from turtlewave_hdEEG import dbwrite
from turtlewave_hdEEG import cli

_ENV = 'TURTLEWAVE_SQLITE_JOURNAL'


def _tmp_db(tag):
    """Return a fresh temp directory + db path, and register it for cleanup."""
    d = tempfile.mkdtemp(prefix=f'twjournal_{tag}_')
    return d, os.path.join(d, 'neural_events.db')


def _sidecars(db_path):
    return [suf for suf in ('-wal', '-shm') if os.path.exists(db_path + suf)]


def _mode_of(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
    finally:
        conn.close()


def _populate(conn, n=100, offset=0):
    conn.execute('CREATE TABLE IF NOT EXISTS t (x INTEGER)')
    conn.executemany('INSERT INTO t VALUES (?)',
                      [(i,) for i in range(offset, offset + n)])
    conn.commit()


def test_new_database_created_in_delete():
    """A database open_write_connection CREATES is DELETE, with no sidecars.

    The 4.0.2 create-path fix. A database born in WAL on an SMB share or in a
    Dropbox/OneDrive folder is broken from its first write -- the sync client
    and the -wal/-shm sidecars corrupt each other -- so the mode a *new*
    database gets must be DELETE unless someone asks otherwise. Sidecars are
    sampled at every step, not just at the end, because a -wal file that
    appears and is cleaned up on close is still a file the sync client saw.
    """
    print("\n1. Testing a newly created database is DELETE, never WAL:")
    os.environ.pop(_ENV, None)
    d, db = _tmp_db('new')
    try:
        assert not os.path.exists(db), "db must not exist before this call"
        observed = []
        conn = dbwrite.open_write_connection(db)
        observed += _sidecars(db)
        reported = str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        _populate(conn)
        observed += _sidecars(db)
        conn.close()
        observed += _sidecars(db)
        mode = _mode_of(db)
        print(f"   reported={reported!r}, on-disk={mode!r}, "
              f"sidecars observed at any point={sorted(set(observed))}")
        assert reported == 'delete', f"expected delete, got {reported!r}"
        assert mode == 'delete', f"expected delete on disk, got {mode!r}"
        assert observed == [], (
            "a -wal/-shm sidecar appeared during create/insert/close: "
            f"{sorted(set(observed))}")
        print("[ok] a database this call creates is delete, with no sidecars")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_env_var_wal_still_creates_wal():
    """TURTLEWAVE_SQLITE_JOURNAL=WAL still opts a NEW database into WAL.

    The create default changed direction in 4.0.2, but the env var must still
    override in *both* directions: a user on fast local disk who wants WAL's
    concurrent reader/writer behaviour sets it and gets it.
    """
    print("\n1b. Testing TURTLEWAVE_SQLITE_JOURNAL=WAL still creates WAL:")
    d, db = _tmp_db('envwal')
    os.environ[_ENV] = 'WAL'
    try:
        assert not os.path.exists(db), "db must not exist before this call"
        conn = dbwrite.open_write_connection(db)
        reported = str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        _populate(conn)
        seen_while_open = _sidecars(db)
        conn.close()
        print(f"   reported={reported!r}, on-disk={_mode_of(db)!r}, "
              f"sidecars while open={seen_while_open}")
        assert reported == 'wal', f"expected wal, got {reported!r}"
        assert _mode_of(db) == 'wal'
        assert '-wal' in seen_while_open, "expected a -wal sidecar while open"
        print("[ok] the env var still opts a new database into WAL")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_env_var_delete_never_creates_sidecars():
    """TURTLEWAVE_SQLITE_JOURNAL=DELETE: mode is delete, no sidecar ever appears.

    This is the remaining real job of the env var: forcing DELETE on *new*
    output written straight to a share (a fresh neural_events.db created on
    K:\\ would otherwise still be created in WAL).
    """
    print("\n2. Testing TURTLEWAVE_SQLITE_JOURNAL=DELETE forces DELETE on new "
          "output:")
    d, db = _tmp_db('envdelete')
    os.environ[_ENV] = 'DELETE'
    try:
        observed = []
        conn = dbwrite.open_write_connection(db)
        observed += _sidecars(db)
        reported = str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        conn.execute('CREATE TABLE t (x INTEGER)')
        observed += _sidecars(db)
        conn.executemany('INSERT INTO t VALUES (?)', [(i,) for i in range(100)])
        observed += _sidecars(db)          # pre-commit
        conn.commit()
        observed += _sidecars(db)          # post-commit
        conn.close()
        observed += _sidecars(db)          # post-close

        print(f"   reported={reported!r}, on-disk={_mode_of(db)!r}, "
              f"sidecars observed at any point={sorted(set(observed))}")
        assert reported == 'delete', f"expected delete, got {reported!r}"
        assert _mode_of(db) == 'delete'
        assert observed == [], (
            "a -wal/-shm sidecar appeared at some point during "
            f"create/insert/commit/close: {sorted(set(observed))}")
        print("[ok] DELETE mode never creates a -wal/-shm sidecar")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_bad_env_value_raises_value_error():
    """An unrecognised journal mode raises ValueError naming the variable."""
    print("\n3. Testing a bad TURTLEWAVE_SQLITE_JOURNAL value raises ValueError:")
    d, db = _tmp_db('badenv')
    os.environ[_ENV] = 'DELET'    # typo
    try:
        try:
            dbwrite.open_write_connection(db)
            raise AssertionError("expected ValueError, none was raised")
        except ValueError as e:
            msg = str(e)
            print(f"   ValueError: {msg}")
            assert _ENV in msg, "message should name the environment variable"
            assert 'WAL' in msg and 'DELETE' in msg, (
                "message should list the valid modes")
        print("[ok] bad journal mode raises ValueError naming the variable")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_blank_env_value_treated_as_unset():
    """An empty/whitespace-only TURTLEWAVE_SQLITE_JOURNAL counts as unset.

    Blanking a variable instead of unsetting it is ordinary POSIX practice: a
    PBS/bash job template that does
    ``export TURTLEWAVE_SQLITE_JOURNAL="$MAYBE_UNSET"`` exports an empty
    string. That must mean "no preference", not a mode -- otherwise every
    database open in the job dies with ``ValueError: Unrecognised SQLite
    journal mode ''`` before any detection runs. So a blank value has to
    behave exactly like an absent one: a NEW database gets the DELETE create
    default, and an EXISTING database keeps whatever mode it is in.
    """
    print("\n3b. Testing a blank TURTLEWAVE_SQLITE_JOURNAL is treated as unset:")
    for blank in ('', '   ', '\t'):
        d, db = _tmp_db('blankenv')
        os.environ[_ENV] = blank
        try:
            # (a) resolver: blank at either level resolves to None.
            assert dbwrite._resolve_journal_mode() is None, (
                f"env {blank!r} should resolve to None")
            assert dbwrite._resolve_journal_mode('') is None, (
                "a blank argument should resolve to None too")

            # (b) new database: create default applies, no ValueError.
            conn = dbwrite.open_write_connection(db)
            created = str(conn.execute(
                'PRAGMA journal_mode').fetchone()[0]).lower()
            _populate(conn)
            conn.close()
            assert created == 'delete', (
                f"env {blank!r} should leave the DELETE create default in "
                f"force, got {created!r}")

            # (c) existing database: mode preserved, not overridden.
            conn = dbwrite.open_write_connection(db, journal='WAL')
            conn.close()
            assert _mode_of(db) == 'wal'
            conn = dbwrite.open_write_connection(db)
            preserved = str(conn.execute(
                'PRAGMA journal_mode').fetchone()[0]).lower()
            conn.close()
            print(f"   env={blank!r}: new db -> {created!r}, existing wal db "
                  f"-> {preserved!r}")
            assert preserved == 'wal', (
                f"a blank env var must not re-mode an existing database, got "
                f"{preserved!r}")
        finally:
            os.environ.pop(_ENV, None)
            shutil.rmtree(d, ignore_errors=True)
    print("[ok] a blank env var behaves exactly like an unset one")


def test_valid_journal_modes_is_public():
    """dbwrite.VALID_JOURNAL_MODES is the public source of truth for modes."""
    print("\n4. Testing VALID_JOURNAL_MODES is public and complete:")
    modes = dbwrite.VALID_JOURNAL_MODES
    print(f"   VALID_JOURNAL_MODES={modes}")
    for expected in ('DELETE', 'TRUNCATE', 'PERSIST', 'MEMORY', 'WAL', 'OFF'):
        assert expected in modes, f"{expected} missing from VALID_JOURNAL_MODES"
    print("[ok] VALID_JOURNAL_MODES exposes all six SQLite journal modes")


def test_set_journal_mode_on_populated_wal_db():
    """set_journal_mode converts WAL -> delete, checkpoints, keeps the row count.

    The row-count assertion is the one that catches a skipped checkpoint --
    i.e. silent data loss from committed rows still sitting in the -wal file.
    """
    print("\n5. Testing set_journal_mode on a populated WAL database:")
    d, db = _tmp_db('convert')
    os.environ.pop(_ENV, None)
    try:
        # Explicit journal='WAL': the create default is DELETE, so the
        # legacy WAL database this test converts has to be built on purpose.
        conn = dbwrite.open_write_connection(db, journal='WAL')
        _populate(conn, n=100, offset=0)
        conn.close()
        # Commit more rows and abandon without checkpointing, so committed data
        # genuinely sits only in the -wal sidecar -- the silent-loss scenario.
        raw = sqlite3.connect(db)
        raw.executemany('INSERT INTO t VALUES (?)',
                         [(i,) for i in range(100, 150)])
        raw.commit()
        raw.close()

        before_mode = _mode_of(db)
        before_count = sqlite3.connect(db).execute(
            'SELECT count(*) FROM t').fetchone()[0]
        returned = dbwrite.set_journal_mode(db)
        after_count = sqlite3.connect(db).execute(
            'SELECT count(*) FROM t').fetchone()[0]

        print(f"   before={before_mode!r} ({before_count} rows), "
              f"returned={returned!r}, after={_mode_of(db)!r} "
              f"({after_count} rows), sidecars={_sidecars(db)}")
        assert before_mode == 'wal'
        assert returned == 'delete', f"expected 'delete', got {returned!r}"
        assert _mode_of(db) == 'delete'
        assert _sidecars(db) == [], f"sidecars still present: {_sidecars(db)}"
        assert before_count == after_count == 150, (
            "row count changed across the conversion -- a checkpoint was "
            f"skipped and data was lost (before={before_count}, "
            f"after={after_count})")
        print("[ok] set_journal_mode converts, checkpoints, keeps every row")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_converted_database_stays_converted():
    """The core promise of the fix: a converted database is never re-WALed.

    Create a legacy WAL database (explicitly, since the create default is now
    DELETE), convert it once with set_journal_mode, then reopen it through a
    plain open_write_connection -- the same call every detection processor and
    review GUI makes -- with no journal= and no env var. It must stay in
    DELETE, and no -wal/-shm sidecar may reappear across insert/commit/close.
    """
    print("\n6. Testing a converted database stays converted across a plain "
          "open_write_connection:")
    d, db = _tmp_db('staysconverted')
    os.environ.pop(_ENV, None)
    try:
        conn = dbwrite.open_write_connection(db, journal='WAL')
        _populate(conn, n=100)
        conn.close()
        assert _mode_of(db) == 'wal'
        converted = dbwrite.set_journal_mode(db)
        assert converted == 'delete'

        observed = []
        conn = dbwrite.open_write_connection(db)   # the detector/GUI path
        observed += _sidecars(db)
        reported = str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        conn.executemany('INSERT INTO t VALUES (?)',
                          [(i,) for i in range(100, 110)])
        observed += _sidecars(db)
        conn.commit()
        observed += _sidecars(db)
        conn.close()
        observed += _sidecars(db)

        count = sqlite3.connect(db).execute(
            'SELECT count(*) FROM t').fetchone()[0]
        print(f"   reported={reported!r}, on-disk={_mode_of(db)!r}, "
              f"sidecars observed={sorted(set(observed))}, rows={count}")
        assert reported == 'delete', (
            f"open_write_connection re-imposed a mode on an existing "
            f"database: got {reported!r}")
        assert _mode_of(db) == 'delete'
        assert observed == [], (
            f"a -wal/-shm sidecar reappeared after conversion: "
            f"{sorted(set(observed))}")
        assert count == 110, f"the write should still go through ({count})"
        print("[ok] a converted database stays in DELETE through a plain "
              "open_write_connection, and writes still succeed")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_explicit_journal_overrides_existing_delete_database():
    """journal='WAL' still converts a DELETE-mode database back to WAL.

    The preserve rule only suppresses the *unrequested* default; an explicit
    request (argument or env var) is still an override in either direction.
    The DELETE database here is simply CREATED that way (that is the 4.0.2
    default), not converted by set_journal_mode -- the conversion path is
    test 6's job. Either way what is asserted is the override.
    """
    print("\n7. Testing an explicit journal='WAL' still overrides an existing "
          "DELETE-mode database:")
    d, db = _tmp_db('explicitoverride')
    os.environ.pop(_ENV, None)
    try:
        conn = dbwrite.open_write_connection(db)    # created -> delete
        _populate(conn)
        conn.close()
        assert _mode_of(db) == 'delete'

        conn = dbwrite.open_write_connection(db, journal='WAL')
        reported = str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        conn.close()
        print(f"   after explicit journal='WAL': reported={reported!r}, "
              f"on-disk={_mode_of(db)!r}")
        assert reported == 'wal'
        assert _mode_of(db) == 'wal'
        print("[ok] an explicit journal= argument still overrides an existing "
              "DELETE-mode database")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_zero_byte_placeholder_treated_as_existing_delete():
    """A zero-byte placeholder file counts as existing -> created in DELETE.

    Documented edge case: the existence check runs before connect() (which
    would otherwise create the file itself), so a zero-byte file left by
    `touch`, an interrupted copy, or a failed create is treated as an existing
    database. SQLite reports an unformatted file's journal mode as 'delete',
    and the preserve rule leaves it there. Since 4.0.2 that agrees with the
    create default, so the two paths can no longer disagree about a
    placeholder file.
    """
    print("\n8. Testing a zero-byte placeholder file is treated as existing "
          "(-> delete):")
    d, db = _tmp_db('zerobyte')
    os.environ.pop(_ENV, None)
    try:
        open(db, 'w').close()
        assert os.path.exists(db) and os.path.getsize(db) == 0

        conn = dbwrite.open_write_connection(db)
        reported = str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        _populate(conn)
        conn.close()

        count = sqlite3.connect(db).execute(
            'SELECT count(*) FROM t').fetchone()[0]
        print(f"   reported={reported!r}, rows={count}")
        assert reported == 'delete', (
            f"a zero-byte placeholder should come out delete, not wal, got "
            f"{reported!r}")
        assert count == 100, "the database should still be usable"
        print("[ok] zero-byte placeholder is treated as existing, comes out "
              "delete, and is usable")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_set_journal_mode_blocked_by_another_connection():
    """A competing open connection makes set_journal_mode raise RuntimeError.

    This is the most important test in the file: leaving WAL silently no-ops
    (or raises 'database is locked') when another process holds the database,
    and set_journal_mode must turn either outcome into a loud RuntimeError
    rather than pretend the conversion succeeded.
    """
    print("\n9. Testing set_journal_mode raises when another connection holds "
          "the database:")
    d, db = _tmp_db('blocked')
    os.environ.pop(_ENV, None)
    try:
        # Explicitly WAL: leaving WAL is the conversion that needs the
        # exclusive lock, so DELETE (the create default) would not exercise
        # the blocking path at all.
        conn = dbwrite.open_write_connection(db, journal='WAL')
        _populate(conn)
        conn.close()

        holder = sqlite3.connect(db, timeout=1.0)
        holder.isolation_level = None
        holder.execute('BEGIN')
        holder.execute('SELECT count(*) FROM t').fetchone()  # open read txn
        t0 = time.time()
        try:
            got = dbwrite.set_journal_mode(db)
            raise AssertionError(
                f"expected RuntimeError, set_journal_mode returned {got!r}")
        except RuntimeError as e:
            elapsed = time.time() - t0
            print(f"   RuntimeError after {elapsed:.1f}s: {e}")
            assert 'holding the database' in str(e)
            assert elapsed < 5, (
                f"took {elapsed:.1f}s -- should fail fast, not hang on "
                "busy_timeout")
        finally:
            holder.rollback()
            holder.close()

        assert _mode_of(db) == 'wal', (
            "database should be untouched (still wal) after a blocked "
            "conversion")
        print("[ok] a blocked conversion raises RuntimeError, db left as-is")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_set_journal_mode_does_not_mask_io_error():
    """A genuine 'disk I/O error' must stay an OperationalError, not a RuntimeError.

    set_journal_mode folds a *lock* OperationalError into RuntimeError, but
    only when the message contains 'lock' or 'busy'. A non-lock
    OperationalError -- notably 'disk I/O error', the network-share failure
    this whole feature exists for -- must keep its class, or a user would be
    told to "close every GUI" for a problem no GUI caused. The message is
    rewritten by _explain_io_error to name the likely cause and the fix, so
    this asserts the class and the retained 'disk I/O error' text, not that
    the message is byte-identical.
    """
    print("\n10. Testing a non-lock OperationalError ('disk I/O error') keeps "
          "its class instead of becoming a RuntimeError:")

    class _Row:
        def __init__(self, *v):
            self._v = v

        def fetchone(self):
            return self._v

    class _IOErrConn:
        """Connection stub whose journal_mode=X pragma fails with an I/O error."""

        def __init__(self):
            self.closed = False

        def execute(self, sql, *a):
            low = sql.lower()
            if low.startswith('pragma journal_mode='):
                raise sqlite3.OperationalError('disk I/O error')
            if low.startswith('pragma journal_mode'):
                return _Row('wal')
            if low.startswith('pragma wal_checkpoint'):
                return _Row(0, 0, 0)   # (busy=0, log, checkpointed): not blocked
            return _Row('ok')

        def close(self):
            self.closed = True

    stub = _IOErrConn()
    real_connect = sqlite3.connect
    sqlite3.connect = lambda *a, **k: stub
    try:
        try:
            dbwrite.set_journal_mode('/does/not/matter.db')
            raise AssertionError("expected sqlite3.OperationalError, none raised")
        except RuntimeError as e:
            raise AssertionError(
                f"'disk I/O error' was masked as RuntimeError: {e}")
        except sqlite3.OperationalError as e:
            print(f"   still an OperationalError: {e}")
            assert 'disk I/O error' in str(e)
    finally:
        sqlite3.connect = real_connect
    assert stub.closed, "connection should still be closed in the finally block"
    print("[ok] a non-lock OperationalError is never mis-reported as a lock")


def test_explicit_journal_beats_env_var():
    """open_write_connection(journal='WAL') overrides TURTLEWAVE_SQLITE_JOURNAL."""
    print("\n11. Testing an explicit journal= argument beats the env var:")
    d, db = _tmp_db('override')
    os.environ[_ENV] = 'DELETE'
    try:
        conn = dbwrite.open_write_connection(db, journal='WAL')
        reported = str(conn.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        _populate(conn)
        conn.close()
        print(f"   env={os.environ[_ENV]!r}, explicit journal='WAL' -> "
              f"reported={reported!r}, on-disk={_mode_of(db)!r}")
        assert reported == 'wal'
        assert _mode_of(db) == 'wal'
        print("[ok] explicit journal= wins over TURTLEWAVE_SQLITE_JOURNAL")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_set_journal_mode_none_with_no_env_raises():
    """set_journal_mode(mode=None) with no env var set raises ValueError.

    Unlike open_write_connection, set_journal_mode has no sensible "preserve"
    fallback -- converting is its entire job, so an unresolved mode is fatal.
    """
    print("\n12. Testing set_journal_mode(mode=None) with no env var raises "
          "ValueError:")
    d, db = _tmp_db('noneraises')
    os.environ.pop(_ENV, None)
    try:
        conn = dbwrite.open_write_connection(db)
        _populate(conn)
        conn.close()
        try:
            dbwrite.set_journal_mode(db, mode=None)
            raise AssertionError("expected ValueError, none was raised")
        except ValueError as e:
            print(f"   ValueError: {e}")
        print("[ok] set_journal_mode(mode=None) with no env var raises "
              "ValueError")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_cli_converts_a_tree():
    """turtlewave_set_journal_mode converts a directory of databases, exit 0.

    The console script is the only route a pip user has to this repair --
    examples/ is not in the wheel -- so its argv contract is worth a test: a
    directory argument finds every neural_events.db beneath it, each is
    converted, and the exit status is 0 only when all of them succeeded.
    """
    print("\n13. Testing the turtlewave_set_journal_mode CLI over a tree:")
    d, _ = _tmp_db('cli')
    os.environ.pop(_ENV, None)
    try:
        dbs = []
        for sub in ('s01', 's02'):
            sub_dir = os.path.join(d, sub, 'wonambi')
            os.makedirs(sub_dir)
            db = os.path.join(sub_dir, 'neural_events.db')
            conn = dbwrite.open_write_connection(db, journal='WAL')
            _populate(conn)
            conn.close()
            assert _mode_of(db) == 'wal'
            dbs.append(db)

        found = cli.find_databases(d, None)
        rc = cli.main([d])
        modes = [_mode_of(db) for db in dbs]
        counts = [sqlite3.connect(db).execute(
            'SELECT count(*) FROM t').fetchone()[0] for db in dbs]
        print(f"   found={len(found)} db(s), exit={rc}, modes={modes}, "
              f"rows={counts}")
        assert len(found) == 2, f"expected 2 databases, found {found}"
        assert rc == 0, f"expected exit 0, got {rc}"
        assert modes == ['delete', 'delete'], modes
        assert counts == [100, 100], "rows lost across the conversion"

        # --mode WAL converts back, and a missing file is a failure (exit 1),
        # not a silent pass.
        assert cli.main(['--mode', 'WAL', dbs[0]]) == 0
        assert _mode_of(dbs[0]) == 'wal'
        rc_missing = cli.main([os.path.join(d, 'nope.db')])
        print(f"   --mode WAL -> {_mode_of(dbs[0])!r}; missing file exit="
              f"{rc_missing}")
        assert rc_missing == 1, "a missing database must exit non-zero"
        print("[ok] the CLI converts a tree, honours --mode, and fails loudly")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_cli_directory_mode_matches_only_neural_events_db():
    """A directory search touches neural_events.db and nothing else.

    Journal mode is a persistent on-disk property, so converting a database is
    a real change to a file. A recursive ``*.db`` sweep of a study folder had
    two failure modes, both reproduced here: someone else's WAL database
    sitting under the tree got silently converted and reported ``[PASS]``, and
    a stray non-SQLite ``*.db`` file failed and took the whole batch's exit
    status to 1 even though every real database converted. Narrowing directory
    mode to the canonical filename fixes both; ``--glob`` (or naming the file)
    stays the escape hatch, and still fails loudly on a file that is not a
    database.
    """
    print("\n13b. Testing directory mode only matches neural_events.db:")
    d, _ = _tmp_db('clinarrow')
    os.environ.pop(_ENV, None)
    try:
        sub_dir = os.path.join(d, 's01', 'wonambi')
        os.makedirs(sub_dir)
        ours = os.path.join(sub_dir, 'neural_events.db')
        conn = dbwrite.open_write_connection(ours, journal='WAL')
        _populate(conn)
        conn.close()

        # A third-party WAL database that merely lives under the same tree.
        foreign = os.path.join(d, 's01', 'someone_elses.db')
        conn = dbwrite.open_write_connection(foreign, journal='WAL')
        _populate(conn)
        conn.close()

        # A stray *.db that is not a SQLite database at all.
        junk = os.path.join(d, 's01', 'notes.db')
        with open(junk, 'w') as fh:
            fh.write('this is not a database\n')

        found = cli.find_databases(d, None)
        rc = cli.main([d])
        print(f"   found={[os.path.basename(p) for p in found]}, exit={rc}, "
              f"ours={_mode_of(ours)!r}, foreign={_mode_of(foreign)!r}")
        assert found == [ours], f"expected only {ours}, found {found}"
        assert rc == 0, (
            f"a stray non-SQLite *.db must not fail the batch, got exit {rc}")
        assert _mode_of(ours) == 'delete', "our database should be converted"
        assert _mode_of(foreign) == 'wal', (
            "a third-party database under the tree must be left alone, it is "
            f"now in {_mode_of(foreign)!r}")

        # --glob is still the escape hatch for a non-canonical filename...
        rc_glob = cli.main(['--glob', os.path.join(d, '**', 'someone_elses.db')])
        assert rc_glob == 0, f"expected exit 0 from --glob, got {rc_glob}"
        assert _mode_of(foreign) == 'delete', (
            "--glob should still convert an explicitly named database")

        # ...and an explicitly named non-database still fails loudly (exit 1),
        # which is the behaviour that must NOT be lost by narrowing the search.
        rc_junk = cli.main([junk])
        print(f"   --glob on the non-canonical name -> "
              f"{_mode_of(foreign)!r}; explicitly named non-database exit="
              f"{rc_junk}")
        assert rc_junk == 1, "an explicitly named non-database must exit 1"

        # A directory with no neural_events.db matches nothing -> exit 1.
        empty = os.path.join(d, 'empty')
        os.makedirs(empty)
        rc_empty = cli.main([empty])
        assert rc_empty == 1, "an empty directory must exit non-zero"
        print("[ok] directory mode converts only neural_events.db; --glob and "
              "an explicit filename still reach anything else")
    finally:
        os.environ.pop(_ENV, None)
        shutil.rmtree(d, ignore_errors=True)


def test_cli_usage_error_exits_2():
    """Both a path and --glob is a usage error: argparse exits 2, not 1.

    Documented in main()'s Returns section and in the CLI reference page; the
    0/1 contract of the return value never covers it because argparse calls
    sys.exit itself.
    """
    print("\n13c. Testing a usage error exits 2 via argparse:")
    try:
        cli.main(['/some/db', '--glob', 'ROOT/*/neural_events.db'])
        raise AssertionError("expected SystemExit, none was raised")
    except SystemExit as e:
        print(f"   SystemExit code={e.code}")
        assert e.code == 2, f"expected exit code 2, got {e.code}"
    print("[ok] a usage error exits 2")


def test_cli_requires_exactly_one_target():
    """find_databases rejects both-or-neither of path and --glob."""
    print("\n14. Testing the CLI needs exactly one of a path or --glob:")
    for args in ((None, None), ('/some/db', 'ROOT/*/neural_events.db')):
        try:
            cli.find_databases(*args)
            raise AssertionError(f"expected ValueError for {args}")
        except ValueError as e:
            print(f"   {args} -> ValueError: {e}")
    print("[ok] exactly one target is required")


if __name__ == "__main__":
    print("TESTING DATABASE JOURNAL MODE (network-drive fix)")
    print("===================================================")

    test_new_database_created_in_delete()
    test_env_var_wal_still_creates_wal()
    test_env_var_delete_never_creates_sidecars()
    test_bad_env_value_raises_value_error()
    test_blank_env_value_treated_as_unset()
    test_valid_journal_modes_is_public()
    test_set_journal_mode_on_populated_wal_db()
    test_converted_database_stays_converted()
    test_explicit_journal_overrides_existing_delete_database()
    test_zero_byte_placeholder_treated_as_existing_delete()
    test_set_journal_mode_blocked_by_another_connection()
    test_set_journal_mode_does_not_mask_io_error()
    test_explicit_journal_beats_env_var()
    test_set_journal_mode_none_with_no_env_raises()
    test_cli_converts_a_tree()
    test_cli_directory_mode_matches_only_neural_events_db()
    test_cli_usage_error_exits_2()
    test_cli_requires_exactly_one_target()

    print("\nAll tests completed!")
