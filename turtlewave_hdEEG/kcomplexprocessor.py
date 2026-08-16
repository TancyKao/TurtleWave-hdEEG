import os
import json
import uuid
import bisect
import logging

from wonambi.trans import fetch, math
from wonambi.attr import Annotations

from turtlewave_hdEEG.extensions import ImprovedDetectKComplex
from turtlewave_hdEEG.swprocessor import ParalSWA
from turtlewave_hdEEG import dbwrite


class ParalKC:
    """
    K-complex detection across many channels.

    Mirrors the ParalSWA pattern: per-channel detection writes one JSON per
    channel into ``json_dir``; CSV / SQLite export reuses the slow-wave
    helpers on ParalSWA (KC parameters are structurally identical to SW
    parameters, so duplicating the export code would just be drift bait).

    Detected events are stored under the ``'k_complex'`` event type (in
    Wonambi XML annotations and the SQLite events table) and JSON files use
    the ``kcomplex_`` prefix to keep them distinct from slow waves on disk.
    """

    EVENT_TYPE = 'k_complex'
    FILE_PREFIX = 'kcomplex'
    SUPPORTED_METHODS = ImprovedDetectKComplex.SUPPORTED_METHODS

    def __init__(self, dataset, annotations=None, log_level=logging.INFO,
                 log_file=None):
        self.dataset = dataset
        self.annotations = annotations
        self.logger = self._setup_logger(log_level, log_file)
        self._sw_proxy = ParalSWA(dataset=dataset, annotations=annotations,
                                  log_level=log_level, log_file=log_file)

    def _setup_logger(self, log_level, log_file=None):
        logger = logging.getLogger('turtlewave_hdEEG.kcomplexprocessor')
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if not logger.handlers:
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            logger.addHandler(console)
            if log_file:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
        return logger

    def detect_kcomplexes(self, method='AASM/Massimini2004', chan=None,
                          ref_chan=None, grp_name='eeg',
                          frequency=(0.1, 4.0),
                          trough_duration=(0.25, 1.0),
                          neg_peak_thresh=-37.0, p2p_thresh=70.0,
                          min_isolation=1.0,
                          detrend=False, polar='normal',
                          reject_artifacts=True, reject_arousals=True,
                          stage=None, cat=None,
                          save_to_annotations=False, json_dir=None,
                          create_empty_json=True,
                          *, write_db=False, db_path=None, resume=False,
                          run_params=None, replace_channels=None, n_fft_sec=4):
        """
        Detect K-complexes in the dataset.

        Parameters
        ----------
        method : str or list
            One of 'AASM/Massimini2004' (default) or 'Massimini2004'.
            Other Wonambi SW methods are not exposed here because they
            target slow oscillations rather than KCs.
        chan : list or str
            Channels to analyze.
        frequency : tuple
            Bandpass for KC detection. AASM KC default is (0.1, 4.0) Hz —
            wider than the SO band so the full KC waveform is captured.
        trough_duration : tuple
            Min/max negative half-wave duration (s). AASM KC default
            (0.25, 1.0).
        neg_peak_thresh : float
            Minimum trough amplitude in µV (negative). AASM KC default −37 µV.
        p2p_thresh : float
            Minimum peak-to-peak amplitude in µV. AASM KC default 70 µV.
        min_isolation : float
            Minimum gap in seconds between successive KC trough times. KCs
            closer than this are dropped — this is what distinguishes a KC
            from one cycle of an N3 slow-oscillation train. Set to 0 to
            disable.
        polar : str
            'normal' or 'opposite' — flips the segment polarity before
            detection (kept for parity with ParalSWA, not for the
            reverse-morphology KC literature).
        stage : list or str
            Sleep stages to analyze. KCs are typically scored in N2;
            override to include other stages if needed.
        cat, reject_artifacts, reject_arousals, save_to_annotations,
        json_dir, create_empty_json
            Same semantics as ParalSWA.detect_slow_waves.
        write_db : bool, keyword-only, default False
            When True, write detected K-complexes straight into a SQLite
            database (``db_path``) under the ``'k_complex'`` event type via the
            direct-write path (deterministic uuid5 rows, detector-own
            morphology in ``det_*``, batched re-measured columns, per-scope
            ``processing_status`` tracking, ``detection_runs`` provenance). When
            False the behaviour is byte-identical to the legacy JSON-only path.
        db_path : str or None, keyword-only
            Target SQLite database (or directory -> ``neural_events.db``).
        resume : bool, keyword-only, default False
            When True (and ``write_db``), channels already recorded as
            ``success = 1`` for the same scope are skipped.
        run_params : dict or None, keyword-only
            Extra parameters merged into ``detection_runs.params_json``.
        replace_channels : iterable of str or None, keyword-only
            Scoped channel re-detection (P3). Channels in this set have their
            existing rows for this exact scope (event_type, method, band)
            DELETE-then-INSERT replaced in one transaction; channels not in the
            set keep the append/upsert path and are never touched. Only
            meaningful with ``write_db=True``. ``None`` (default) disables it.
        n_fft_sec : int, keyword-only, default 4
            FFT window (seconds) for the batched spectral re-measurement.

        Returns
        -------
        list
            List of detected K-complex event dicts across all channels.
        """
        if ref_chan is None:
            ref_chan = []

        if isinstance(method, list):
            for m in method:
                if m not in self.SUPPORTED_METHODS:
                    raise ValueError(
                        f"Unsupported KC method '{m}'. "
                        f"Use one of: {self.SUPPORTED_METHODS}")
            methods = list(method)
        else:
            if method not in self.SUPPORTED_METHODS:
                raise ValueError(
                    f"Unsupported KC method '{method}'. "
                    f"Use one of: {self.SUPPORTED_METHODS}")
            methods = [method]

        if isinstance(chan, str):
            chan = [chan]
        if isinstance(stage, str):
            stage = [stage]

        if json_dir:
            os.makedirs(json_dir, exist_ok=True)

        if self.dataset is None:
            self.logger.error("No dataset provided for K-complex detection")
            return []

        if self.annotations is None and save_to_annotations:
            self.logger.warning(
                "save_to_annotations requested but no annotations object — "
                "K-complexes will not be saved to annotations.")
            save_to_annotations = False

        reject_types = []
        if reject_artifacts:
            reject_types.append('Artefact')
        if reject_arousals:
            reject_types.append('Arousal')

        # Two forms of the method, deliberately kept apart:
        #   method_db  - canonical, UNESCAPED ('AASM/Massimini2004'), for
        #                every database write and query, matching the
        #                CSV-import path and the citation table.
        #   method_str - filesystem-safe ('AASM_Massimini2004'), ONLY for
        #                filenames and path components.
        method_db = "_".join(methods)
        method_str = method_db.replace('/', '_')
        # Single source of truth for the band token so the JSON filenames
        # written here and any file_pattern a caller rebuilds later cannot
        # drift (see dbwrite.fmt_freq_token).
        freq_str = dbwrite.fmt_freq_token(frequency[0], frequency[1])
        stages_str = "".join(stage) if stage else "all"

        self.logger.info(
            f"Detecting K-complexes (method={method_db}, "
            f"freq={freq_str}, stages={stages_str}, "
            f"min_isolation={min_isolation}s)")

        new_annotations = None
        annotation_file_path = None
        if save_to_annotations:
            chan_str = ("_".join(chan) if chan and len(chan) <= 3
                        else f"{chan[0]}_plus_{len(chan) - 1}_chans")
            annotation_filename = (
                f"{self.FILE_PREFIX}_{method_str}_{chan_str}_{freq_str}.xml")
            annotation_file_path = (os.path.join(json_dir, annotation_filename)
                                    if json_dir else annotation_filename)
            try:
                import shutil
                if (hasattr(self.annotations, 'xml_file')
                        and os.path.exists(self.annotations.xml_file)):
                    shutil.copy(self.annotations.xml_file,
                                annotation_file_path)
                    new_annotations = Annotations(annotation_file_path)
                    try:
                        existing = new_annotations.get_events(self.EVENT_TYPE)
                        if existing:
                            self.logger.info(
                                f"Removing {len(existing)} existing "
                                f"{self.EVENT_TYPE} events")
                            new_annotations.remove_event_type(self.EVENT_TYPE)
                    except Exception as e:
                        self.logger.debug(
                            f"No existing {self.EVENT_TYPE} events to "
                            f"remove: {e}")
                else:
                    with open(annotation_file_path, 'w', encoding='utf-8') as f:
                        f.write('<?xml version="1.0" ?>\n'
                                '<annotations><dataset><filename>')
                        if hasattr(self.dataset, 'filename'):
                            f.write(str(self.dataset.filename))
                        f.write('</filename></dataset>'
                                '<rater><name>Wonambi</name></rater>'
                                '</annotations>')
                    new_annotations = Annotations(annotation_file_path)
                self.logger.info(
                    f"Will save K-complexes to: {annotation_file_path}")
            except Exception as e:
                self.logger.error(f"Could not prepare annotation file: {e}")
                save_to_annotations = False
                new_annotations = None

        # ------------------------------------------------------------------
        # Direct-to-DB write path setup (opt-in; JSON behaviour unchanged).
        # ------------------------------------------------------------------
        stages_key = stages_str
        db_conn = None
        run_id = None
        db_skip = set()
        rec_start = None
        s_freq = None
        if write_db:
            if db_path is None:
                self.logger.error("write_db=True but db_path is None; skipping DB writes")
                write_db = False
            else:
                try:
                    if os.path.isdir(db_path):
                        db_path = os.path.join(db_path, 'neural_events.db')
                    self.initialize_sqlite_database(db_path)
                    db_conn = dbwrite.open_write_connection(db_path)
                    dbwrite.ensure_direct_write_schema(db_conn, self.logger)
                    try:
                        s_freq = self.dataset.header['s_freq']
                    except Exception:
                        s_freq = None
                    try:
                        rec_start = self.dataset.header.get('start_time')
                    except Exception:
                        rec_start = None
                    run_id = str(uuid.uuid4())
                    params_dict = {
                        'frequency': list(frequency),
                        'trough_duration': list(trough_duration),
                        'neg_peak_thresh': neg_peak_thresh,
                        'p2p_thresh': p2p_thresh,
                        'min_isolation': min_isolation,
                        'detrend': detrend, 'polar': polar,
                        'method': method_db,
                        'ref_chan': ref_chan, 'cat': cat,
                        'reject_artifacts': reject_artifacts,
                        'reject_arousals': reject_arousals,
                        'n_fft_sec': n_fft_sec,
                    }
                    if run_params:
                        params_dict.update(run_params)
                    dbwrite.record_run(
                        db_conn, run_id, self.EVENT_TYPE, method_db,
                        dbwrite.method_citation(method_db),
                        json.dumps(params_dict, default=str),
                        ref_chan, polar, stage, reject_artifacts, reject_arousals)
                    if resume:
                        db_skip = dbwrite.resume_skip_channels(
                            db_conn, self.EVENT_TYPE, method_db,
                            frequency[0], frequency[1], stages_key)
                        if db_skip:
                            self.logger.info(
                                f"Resume: skipping {len(db_skip)} already-completed "
                                f"channels for this scope")
                except Exception as e:
                    self.logger.error(f"Could not set up direct-DB write: {e}", exc_info=True)
                    write_db = False
                    if db_conn is not None:
                        try:
                            db_conn.close()
                        except Exception:
                            pass
                        db_conn = None

        # Epoch -> stage lookup so each KC is attributed to the single scored
        # epoch it falls in (matches the SW convention).
        try:
            _det_epochs = sorted(
                ((float(e['start']), float(e['end']), str(e['stage']))
                 for e in self.annotations.get_epochs()),
                key=lambda x: x[0]
            ) if self.annotations is not None else []
        except Exception as e:
            self.logger.warning(f"Could not build epoch stage lookup: {e}")
            _det_epochs = []
        _det_epoch_starts = [e[0] for e in _det_epochs]

        def _stage_at(t):
            """Return the scored stage of the epoch containing time t, or None."""
            if t is None or not _det_epochs:
                return None
            idx = bisect.bisect_right(_det_epoch_starts, t) - 1
            if 0 <= idx < len(_det_epochs) and _det_epochs[idx][0] <= t < _det_epochs[idx][1]:
                return _det_epochs[idx][2]
            return None

        all_kcs = []

        # Scoped channel re-detection (P3): channels whose existing rows are
        # DELETE-then-INSERT replaced for this scope; all others stay on P2's
        # append/upsert path untouched.
        replace_set = {str(c) for c in replace_channels} if replace_channels else set()

        for ch in chan:
            if write_db and resume and ch in db_skip:
                self.logger.info(f"Resume: channel {ch} already complete for this scope; skipping")
                continue
            try:
                self.logger.info(f"Reading data for channel {ch}")
                segments = fetch(self.dataset, self.annotations, cat=cat,
                                 stage=stage, cycle=None,
                                 reject_epoch=True, reject_artf=reject_types)
                segments.read_data(ch, ref_chan, grp_name=grp_name)

                channel_kcs = []
                channel_json_kcs = []
                channel_db_events = []
                channel_param_segments = []

                for meth in methods:
                    self.logger.info(f"Applying method: {meth}")
                    for i, seg in enumerate(segments):
                        processed_seg = seg.copy()
                        # Do NOT invert here. `polar` is passed to
                        # DetectKComplex below, which forwards it as Wonambi's
                        # own `opts.invert`; the underlying Massimini method
                        # then negates its local copy of the signal exactly
                        # once (wonambi/detect/slowwave.py:192). Any inversion
                        # added here would cancel that and make
                        # polar='opposite' identical to polar='normal'. Note
                        # `seg.copy()` above is a shallow dict copy, so
                        # processed_seg['data'] is still the caller's ChanTime
                        # until detrend replaces it — an in-place negation here
                        # would also leak across methods.
                        # Locked down by test_slow_wave_polarity in
                        # tests/test_turtlewave.py.
                        if detrend:
                            try:
                                processed_seg['data'] = math(
                                    processed_seg['data'],
                                    operator='detrend', axis='time')
                            except Exception as e:
                                self.logger.error(
                                    f"Error detrending segment {i + 1}: {e}")

                        detector = ImprovedDetectKComplex(
                            method=meth,
                            frequency=frequency,
                            duration=trough_duration,
                            neg_peak_thresh=neg_peak_thresh,
                            p2p_thresh=p2p_thresh,
                            polar=polar,
                            min_isolation=min_isolation,
                        )

                        kcs = detector(processed_seg['data'])

                        if kcs and save_to_annotations and new_annotations is not None:
                            kcs.to_annot(new_annotations, self.EVENT_TYPE)

                        for kc in kcs:
                            kc['uuid'] = str(uuid.uuid4())
                            kc['chan'] = ch
                            channel_kcs.append(kc)

                            if write_db:
                                kc_start = float(kc.get('start', 0))
                                kc_end = float(kc.get('end', 0))
                                kc_dur = float(kc.get('dur', kc_end - kc_start))
                                single_stage = _stage_at(kc_start)
                                if single_stage is None and isinstance(stage, (list, tuple)) and len(stage) == 1:
                                    single_stage = stage[0]
                                morph = dbwrite.event_det_morphology(kc)
                                ev = {
                                    'uuid': dbwrite.event_uuid5(
                                        self.EVENT_TYPE, ch, kc_start, meth,
                                        frequency[0], frequency[1], single_stage),
                                    'start_time': kc_start, 'end_time': kc_end,
                                    'duration': kc_dur, 'stage': single_stage,
                                    'method': meth,
                                }
                                ev.update(morph)
                                channel_db_events.append(ev)
                                channel_param_segments.append(
                                    dbwrite.make_param_segment(
                                        processed_seg['data'], kc_start, kc_end,
                                        self.EVENT_TYPE, single_stage, ch))

                            if json_dir:
                                channel_json_kcs.append({
                                    'uuid': kc['uuid'],
                                    'chan': ch,
                                    'start_time': float(kc.get('start', 0)),
                                    'end_time': float(kc.get('end', 0)),
                                    'trough_time': float(kc.get('trough_time', 0)),
                                    'peak_time': float(kc.get('peak_time', 0)),
                                    'duration': float(kc.get('dur', 0)),
                                    'trough_val': float(kc.get('trough_val', 0)),
                                    'peak_val': float(kc.get('peak_val', 0)),
                                    'ptp': float(kc.get('ptp', 0)),
                                    'method': meth,
                                    'min_isolation': min_isolation,
                                    'stage': stage,
                                    'freq_range': frequency,
                                })

                all_kcs.extend(channel_kcs)
                self.logger.info(
                    f"Found {len(channel_kcs)} K-complexes in channel {ch}")

                # Direct-DB write: one batched re-measurement + one transaction
                # per channel, BEFORE the JSON write.
                if write_db and db_conn is not None:
                    batched = dbwrite.compute_batched_params(
                        channel_param_segments, frequency, s_freq,
                        n_fft_sec, self.logger)
                    dbwrite.write_channel_events(
                        db_conn, run_id, self.EVENT_TYPE, ch, method_db,
                        frequency[0], frequency[1], stages_key,
                        channel_db_events, batched, rec_start,
                        n_fft_sec, self.logger,
                        replace=(ch in replace_set), replace_methods=methods)
                    self.logger.info(
                        f"Wrote {len(channel_db_events)} K-complex rows for "
                        f"channel {ch} to the database")

                if json_dir:
                    ch_json = os.path.join(
                        json_dir,
                        f"{self.FILE_PREFIX}_{method_str}_{freq_str}_"
                        f"{stages_str}_{ch}.json")
                    if not channel_json_kcs and create_empty_json:
                        with open(ch_json, 'w', encoding='utf-8') as f:
                            json.dump([], f)
                    elif channel_json_kcs:
                        with open(ch_json, 'w', encoding='utf-8') as f:
                            json.dump(channel_json_kcs, f, indent=2)
                        self.logger.info(
                            f"Saved K-complex data for channel {ch} to "
                            f"{ch_json}")
            except Exception as e:
                # A real read/detection failure is logged as an error with a
                # traceback so it is not mistaken for a channel that legitimately
                # had no K-complexes.
                self.logger.error(
                    f"Failed to process channel {ch}: {e}", exc_info=True)
                # In the direct-DB path, record the failure in processing_status
                # (success=0) instead of an error sentinel JSON, so a resume
                # re-runs only this channel.
                if write_db and db_conn is not None:
                    dbwrite.record_channel_failure(
                        db_conn, self.EVENT_TYPE, ch, method_db,
                        frequency[0], frequency[1], stages_key, e)
                # Write an error sentinel (not an empty list) so downstream import
                # can tell a failed channel apart from one that legitimately had no
                # K-complexes and re-run it.
                elif json_dir and create_empty_json:
                    try:
                        ch_json = os.path.join(
                            json_dir,
                            f"{self.FILE_PREFIX}_{method_str}_{freq_str}_"
                            f"{stages_str}_{ch}.json")
                        with open(ch_json, 'w', encoding='utf-8') as f:
                            json.dump({"error": str(e), "channel": ch}, f)
                    except Exception as je:
                        self.logger.error(
                            f"Could not write sentinel JSON for {ch}: {je}")

        if write_db and db_conn is not None:
            try:
                db_conn.close()
            except Exception:
                pass

        if save_to_annotations and new_annotations is not None and all_kcs:
            try:
                new_annotations.save(annotation_file_path)
                self.logger.info(
                    f"Saved {len(all_kcs)} K-complexes to "
                    f"{annotation_file_path}")
            except Exception as e:
                self.logger.error(f"Error saving annotation file: {e}")

        self.logger.info(
            f"Total K-complexes detected across all channels: {len(all_kcs)}")
        return all_kcs

    # --- Export delegation -------------------------------------------------
    # KC params and SW params are structurally identical (neg-peak amp,
    # pos-peak amp, PTP, duration), so the SW exporters are reused as-is.
    # Pass file_pattern with the KC prefix so only KC JSONs are picked up.

    def export_kc_parameters_to_csv(self, json_input, csv_file,
                                    export_params='all', frequency=None,
                                    ref_chan=None, grp_name='eeg',
                                    n_fft_sec=4, file_pattern=None,
                                    skip_empty_files=True, strict=True):
        """Export K-complex parameters to CSV.

        Parameters
        ----------
        json_input : str
            Directory holding the per-channel K-complex JSON files.
        csv_file : str
            Output CSV path.
        export_params : dict or str
            Parameters to export; ``'all'`` exports everything available.
        frequency : tuple or None
            Band used for the power calculations.
        ref_chan : list or None
            Reference channel(s) used when re-measuring parameters.
        grp_name : str
            Channel group name.
        n_fft_sec : int
            FFT window length in seconds.
        file_pattern : str or None
            Filename prefix selecting the JSON files to aggregate.
        skip_empty_files : bool
            Whether to list the channels whose JSON held no events in the log
            at INFO (``False``) or only at DEBUG (``True``, the default).
        strict : bool
            If True (default), raise ``FileNotFoundError`` when
            ``file_pattern`` matches no JSON file rather than returning
            quietly. See
            :meth:`ParalSWA.export_slow_wave_parameters_to_csv`.

        Returns
        -------
        dict or None
            Dictionary of calculated parameters, or None when there was
            nothing to export. A run that detected no K-complexes writes a
            header-only CSV and returns None; only an unmatched
            ``file_pattern`` raises.
        """
        # Pass event_type=EVENT_TYPE so the SW exporter writes 'k_complex'
        # into the CSV's "Event type" column instead of the default
        # 'slow_wave'. Without this, the importer (and downstream review GUI
        # filters) would mislabel KCs as slow waves.
        return self._sw_proxy.export_slow_wave_parameters_to_csv(
            json_input=json_input, csv_file=csv_file,
            export_params=export_params, frequency=frequency,
            ref_chan=ref_chan, grp_name=grp_name, n_fft_sec=n_fft_sec,
            file_pattern=file_pattern, skip_empty_files=skip_empty_files,
            event_type=self.EVENT_TYPE, strict=strict)

    def export_kc_density_to_csv(self, json_input, csv_file, stage=None,
                                 file_pattern=None, reject_artifacts=None,
                                 reject_arousals=None):
        """Export K-complex statistics to CSV with whole-night and per-stage densities.

        Delegates to :meth:`ParalSWA.export_slow_wave_density_to_csv`, since
        K-complex and slow-wave density are computed identically; pass
        ``file_pattern`` with the K-complex prefix so only K-complex JSONs are
        aggregated.

        The stage-specific density denominator is the artefact-free in-stage
        time actually fed to the detector (per channel), computed with
        :func:`turtlewave_hdEEG.utils.compute_analysed_seconds`, not the sum of
        all scored epochs of the stage. Detection rejects artefact/arousal
        epochs, so using all scored epochs as the denominator systematically
        under-estimates density in proportion to each recording's artefact
        load.

        Parameters
        ----------
        json_input : str or list
            Path to JSON file, directory of JSON files, or list of JSON files
        csv_file : str
            Path to output CSV file
        stage : str or list
            Sleep stage(s) to include
        file_pattern : str or None
            Pattern to filter JSON files
        reject_artifacts : bool or None, optional
            Subtract time overlapped by 'Artefact' events from the density
            denominator. Should match the detection run's setting. ``None``
            (the default) assumes True and logs a warning saying so; pass the
            value explicitly to confirm it matches the run and silence the
            warning.
        reject_arousals : bool or None, optional
            Subtract time overlapped by 'Arousal' events from the density
            denominator. Should match the detection run's setting. ``None``
            (the default) assumes True and logs a warning saying so; pass the
            value explicitly to confirm it matches the run and silence the
            warning.

        Returns
        -------
        dict or None
            Per-stage, per-channel statistics keyed by stage, or None when
            ``file_pattern`` matched no JSON file (a placeholder CSV is written
            in that case).
        """
        return self._sw_proxy.export_slow_wave_density_to_csv(
            json_input=json_input, csv_file=csv_file, stage=stage,
            file_pattern=file_pattern, reject_artifacts=reject_artifacts,
            reject_arousals=reject_arousals)

    def initialize_sqlite_database(self, db_path='neural_events.db'):
        """Create the ``neural_events.db`` schema if it is not already there.

        Delegates to :meth:`ParalSWA.initialize_sqlite_database`; the schema is
        shared by every event type, so K-complexes need no table of their own.

        Parameters
        ----------
        db_path : str, optional
            Path to the SQLite database. Default ``'neural_events.db'``.

        Returns
        -------
        str
            The path to the initialised database.
        """
        return self._sw_proxy.initialize_sqlite_database(db_path)

    def import_parameters_csv_to_database(self, csv_file, db_path,
                                          append=True, method=None,
                                          force=False):
        """Import a K-complex parameters CSV into ``neural_events.db``.

        Parameters
        ----------
        csv_file : str
            Path to the parameters CSV produced by
            :meth:`export_kc_parameters_to_csv`.
        db_path : str
            Path to the SQLite database.
        append : bool
            If True, add without replacing existing entries.
        method : str or None
            The *original* method string, e.g. ``'AASM/Massimini2004'``. Pass
            it explicitly: the importer's filename parser breaks on the
            escaped form ``AASM_Massimini2004`` and would store just
            ``'AASM'``.
        force : bool
            Overwrite rows that carry a ``run_id`` from the direct-to-database
            path, losing their provenance link. Default False refuses.

        Returns
        -------
        dict
            Import statistics, including ``"ok": True`` on success. A run that
            detected no K-complexes yields a header-only CSV and imports as a
            clean no-op: ``{"ok": True, "no_events": True, "added": 0, ...}``.
            See :meth:`ParalSWA.import_parameters_csv_to_database`.
        """
        # Pass event_type='k_complex' so the row lands under the right type
        # in the events table.
        return self._sw_proxy.import_parameters_csv_to_database(
            csv_file=csv_file, db_path=db_path, append=append,
            event_type=self.EVENT_TYPE, method=method, force=force)

    def save_detection_summary(self, output_dir, method, parameters,
                               results_summary):
        return self._sw_proxy.save_detection_summary(
            output_dir=output_dir, method=method, parameters=parameters,
            results_summary=results_summary)
