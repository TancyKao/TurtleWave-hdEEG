import os
import json
import uuid
import logging

from wonambi.trans import fetch, math
from wonambi.attr import Annotations

from turtlewave_hdEEG.extensions import ImprovedDetectKComplex
from turtlewave_hdEEG.swprocessor import ParalSWA


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
                          create_empty_json=True):
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

        method_str = "_".join(methods).replace('/', '_')
        freq_str = f"{frequency[0]}-{frequency[1]}Hz"
        stages_str = "".join(stage) if stage else "all"

        self.logger.info(
            f"Detecting K-complexes (method={method_str}, "
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
                    with open(annotation_file_path, 'w') as f:
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

        all_kcs = []

        for ch in chan:
            try:
                self.logger.info(f"Reading data for channel {ch}")
                segments = fetch(self.dataset, self.annotations, cat=cat,
                                 stage=stage, cycle=None,
                                 reject_epoch=True, reject_artf=reject_types)
                segments.read_data(ch, ref_chan, grp_name=grp_name)

                channel_kcs = []
                channel_json_kcs = []

                for meth in methods:
                    self.logger.info(f"Applying method: {meth}")
                    for i, seg in enumerate(segments):
                        processed_seg = seg.copy()
                        if polar == 'opposite':
                            processed_seg['data'].data[0][0] = (
                                -processed_seg['data'].data[0][0])
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

                if json_dir:
                    ch_json = os.path.join(
                        json_dir,
                        f"{self.FILE_PREFIX}_{method_str}_{freq_str}_"
                        f"{stages_str}_{ch}.json")
                    if not channel_json_kcs and create_empty_json:
                        with open(ch_json, 'w') as f:
                            json.dump([], f)
                    elif channel_json_kcs:
                        with open(ch_json, 'w') as f:
                            json.dump(channel_json_kcs, f, indent=2)
                        self.logger.info(
                            f"Saved K-complex data for channel {ch} to "
                            f"{ch_json}")
            except Exception as e:
                self.logger.warning(
                    f"No K-complexes in channel {ch}: {e}")
                if json_dir and create_empty_json:
                    try:
                        ch_json = os.path.join(
                            json_dir,
                            f"{self.FILE_PREFIX}_{method_str}_{freq_str}_"
                            f"{stages_str}_{ch}.json")
                        with open(ch_json, 'w') as f:
                            json.dump([], f)
                    except Exception as je:
                        self.logger.error(
                            f"Could not write empty JSON for {ch}: {je}")

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
                                    skip_empty_files=True):
        # Pass event_type=EVENT_TYPE so the SW exporter writes 'k_complex'
        # into the CSV's "Event type" column instead of the default
        # 'slow_wave'. Without this, the importer (and downstream review GUI
        # filters) would mislabel KCs as slow waves.
        return self._sw_proxy.export_slow_wave_parameters_to_csv(
            json_input=json_input, csv_file=csv_file,
            export_params=export_params, frequency=frequency,
            ref_chan=ref_chan, grp_name=grp_name, n_fft_sec=n_fft_sec,
            file_pattern=file_pattern, skip_empty_files=skip_empty_files,
            event_type=self.EVENT_TYPE)

    def export_kc_density_to_csv(self, json_input, csv_file, stage=None,
                                 file_pattern=None):
        return self._sw_proxy.export_slow_wave_density_to_csv(
            json_input=json_input, csv_file=csv_file, stage=stage,
            file_pattern=file_pattern)

    def initialize_sqlite_database(self, db_path='neural_events.db'):
        return self._sw_proxy.initialize_sqlite_database(db_path)

    def import_parameters_csv_to_database(self, csv_file, db_path,
                                          append=True, method=None):
        # Pass event_type='k_complex' so the row lands under the right type
        # in the events table. Pass method= explicitly because the SW
        # importer's filename parser breaks on escaped methods like
        # 'AASM_Massimini2004' (returns just 'AASM'). Caller should pass
        # the *original* method string here (e.g. 'AASM/Massimini2004').
        return self._sw_proxy.import_parameters_csv_to_database(
            csv_file=csv_file, db_path=db_path, append=append,
            event_type=self.EVENT_TYPE, method=method)

    def save_detection_summary(self, output_dir, method, parameters,
                               results_summary):
        return self._sw_proxy.save_detection_summary(
            output_dir=output_dir, method=method, parameters=parameters,
            results_summary=results_summary)
