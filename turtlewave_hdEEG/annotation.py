"""
Annotations module for turtlewave_hdEEG
Provides tools to create and save annotations using event information from EEGLAB
"""

from pathlib import Path
import datetime
import tempfile
import os
import time
import numpy as np
from wonambi.attr import Annotations as WonambiAnnotations
from wonambi.attr.annotations import create_empty_annotations

class XLAnnotations:
    """Simplified annotations for large datasets"""

    def __init__(self, dataset, annot_file,rater_name="Anon"):
        """
        Initialize annotations object.

        Parameters
        ----------
        dataset : LargeDataset
            Dataset to associate with annotations.
        annot_file : str
            Path to the annotation file.
        """
        self.dataset = dataset
        self.annot_file = annot_file
        self.rater_name = rater_name

        # Create or load annotations
        if not Path(annot_file).exists():
            self.annotations = create_empty_annotations(annot_file, dataset)
            self.annotations = WonambiAnnotations(annot_file)
            self.annotations.add_rater(self.rater_name)
            print(f"Created a new annotation object for {annot_file}")
        else:
            # Load existing annotations
            self.annotations = WonambiAnnotations(annot_file)
            if self.rater_name not in self.annotations.raters:
                self.annotations.add_rater(self.rater_name)

            print(f"Loaded existing annotation file: {annot_file}")


    #: Half-width, in seconds, of the ``Artefact`` window written around an
    #: EEGLAB ``boundary`` (splice) marker. The window is
    #: ``[onset - BOUNDARY_PAD_SECONDS, onset + BOUNDARY_PAD_SECONDS]``.
    BOUNDARY_PAD_SECONDS = 2.0

    def _recording_length_seconds(self):
        """Recording length in seconds, for clipping annotation windows.

        Returns
        -------
        float or None
            ``header['n_samples'] / sampling_rate`` when the header carries a
            sample count, else ``header['recording_duration']``, else ``None``
            when neither is available -- in which case callers must clip at
            zero only and leave the upper bound open.
        """
        header = getattr(self.dataset, 'header', {}) or {}
        s_freq = getattr(self.dataset, 'sampling_rate', None)

        n_samples = header.get('n_samples')
        if n_samples is not None and s_freq:
            try:
                return float(n_samples) / float(s_freq)
            except (TypeError, ValueError):
                pass

        duration = header.get('recording_duration')
        if duration is not None:
            try:
                return float(duration)
            except (TypeError, ValueError):
                pass

        return None

    def add_artefacts_from_events(self):
        """
        Add artefact, arousal and other exclusion annotations from the
        dataset's EEGLAB event information.

        Every event in ``dataset.header['event']`` is matched, case-insensitively,
        against the rules below and written as a Wonambi event on ``'(all)'``
        channels. Rules are evaluated in the order listed and an event matches at
        most one of them: ``boundary`` is tested first and its events are removed
        from the remaining masks, so a splice marker can never also be counted as,
        say, a movement.

        | Rule        | Event type matches (lower-cased)                  | Label    |
        |-------------|--------------------------------------------------|----------|
        | boundary    | equals `boundary`                                | Artefact |
        | reject      | contains `reject`                                | Artefact |
        | arousal     | contains `arousal`                               | Arousal  |
        | respiratory | contains `hypopnea`/`obstructiveapnea`/`spo2desat`| Resp     |
        | movement    | contains `move`/`leg`, or equals `lklr`/`lkud`    | Move     |
        | snore       | contains `snor`/`jaw`                            | Snore    |

        Event timing comes from ``onsets`` and ``durations`` (both in samples;
        divided by ``dataset.sampling_rate``). A missing, ``None`` or
        non-numeric duration falls back to 1.0 s. Boundary events are the one
        exception -- see Notes.

        Returns
        -------
        total_count : int
            Number of annotations written, summed over EVERY rule above (this
            used to count only Artefact + Arousal).
        execution_time : float
            Wall-clock seconds spent.

        Notes
        -----
        **Why boundary events are masked.** EEGLAB writes a ``boundary`` event
        at each point where a segment of data was cut out. The remaining samples
        are spliced together, so the signal steps discontinuously at that
        instant. On sub-11vi (107 splices x 257 channels) the median step across
        a splice was 1.5 uV, but 7 boundaries stepped more than 30 uV on at
        least one channel -- large enough to seed a false slow wave. Masking
        +/- 2.0 s around every splice also guarantees that no detected event,
        and in particular no slow-oscillation/spindle coupling phase estimate,
        spans a discontinuity. The cost measured on that subject was about 1.6%
        of analysable time.

        **Why the boundary's own duration is ignored.** For a ``boundary``,
        EEGLAB's ``duration`` field is the length of the data that was REMOVED,
        expressed in the ORIGINAL time base, not a span in the surviving
        recording. Using ``onset + duration`` as the window end therefore
        rejects perfectly good post-splice data -- about 36 minutes on one
        measured subject. The window is a fixed +/- ``BOUNDARY_PAD_SECONDS``
        around the onset instead.

        The window is clipped to ``[0, recording length]``, which covers the
        boundaries some files place at latency 0 or at the very end of the
        recording; a window that collapses to zero length after clipping is
        skipped rather than written.
        """

        start_time = time.time()

        # Check if event information exists in header
        if 'event' not in self.dataset.header:
            print("No event information found in dataset header.")
            end_time = time.time()
            print(f"Processing time: {end_time - start_time:.4f} seconds")
            return 0, end_time - start_time

        event_info = self.dataset.header['event']
        onsets = np.array(event_info.get('onsets', []))
        types = event_info.get('types', [])
        durations = np.array(event_info.get('durations', []))
        #isreject = [str(t).lower() == 'reject' for t in types] if types else []
        # Check if we have any events
        if len(onsets) == 0:
            print("No events found in dataset.")

            return 0, time.time() - start_time

        s_freq = self.dataset.sampling_rate
        # float64 explicitly: `np.ones_like(onsets)` inherited the onsets dtype,
        # so an integer latency array produced an integer duration array and the
        # assignment below truncated every sub-second duration to 0 -- a 0.4 s
        # arousal became a zero-length annotation.
        onset_seconds = np.asarray(onsets, dtype=np.float64) / float(s_freq)
        n_events = len(onset_seconds)
        duration_seconds = np.ones(n_events, dtype=np.float64)

        # `durations` may be shorter than `onsets` (or absent, or hold None /
        # non-numeric entries). Anything unusable keeps the 1.0 s fallback
        # instead of raising IndexError or TypeError.
        for i, raw in enumerate(durations[:n_events]):
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isnan(value):
                continue
            duration_seconds[i] = value / float(s_freq)

        end_seconds = onset_seconds + duration_seconds

        # Pre-compile type checks
        types_arr = np.array([str(t).lower() if t else '' for t in types[:len(onsets)]])

        # Rule 0, evaluated first: EEGLAB splice markers. Matched on the exact
        # type so an unrelated type merely containing the word is not swept in.
        boundary_mask = np.array([t.strip() == 'boundary' for t in types_arr],
                                 dtype=bool)
        not_boundary = ~boundary_mask

        event_masks = {
            "Artefact": (np.char.find(types_arr, 'reject') != -1) & not_boundary,
            "Arousal": (np.char.find(types_arr, 'arousal') != -1) & not_boundary,
            "Resp": np.any([np.char.find(types_arr, x) != -1 for x in ['hypopnea', 'obstructiveapnea', 'spo2desat']], axis=0) & not_boundary,
            "Move": np.any([np.char.find(types_arr, x) != -1 for x in ['move', 'leg']] + [types_arr == x for x in ['lklr', 'lkud']], axis=0) & not_boundary,
            "Snore": np.any([np.char.find(types_arr, x) != -1 for x in ['snor', 'jaw']], axis=0) & not_boundary
        }

        event_counts = {key: 0 for key in event_masks}
        event_counts["Boundary"] = 0

        # Boundary windows: fixed +/- BOUNDARY_PAD_SECONDS around the onset,
        # ignoring the event's own duration (see Notes), clipped to the
        # recording, and dropped if the clip leaves nothing.
        boundary_indices = np.where(boundary_mask)[0]
        if len(boundary_indices) > 0:
            pad = float(self.BOUNDARY_PAD_SECONDS)
            rec_length = self._recording_length_seconds()

            b_onsets = onset_seconds[boundary_indices]
            b_starts = np.maximum(b_onsets - pad, 0.0)
            b_ends = b_onsets + pad
            if rec_length is not None:
                b_starts = np.minimum(b_starts, rec_length)
                b_ends = np.minimum(b_ends, rec_length)

            keep = b_ends > b_starts
            n_skipped = int(np.count_nonzero(~keep))
            if n_skipped:
                print(f"Skipped {n_skipped} boundary event(s) whose "
                      f"+/-{pad:g}s window fell outside the recording.")
            if np.any(keep):
                success = self.add_annotations_batch(
                    label="Artefact",
                    start_times=b_starts[keep],
                    end_times=b_ends[keep],
                    channels=None
                )
                if success:
                    event_counts["Boundary"] += int(np.count_nonzero(keep))

        # Batch process annotations
        for event_type, mask in event_masks.items():
            indices = np.where(mask)[0]
            if len(indices) > 0:
                # Add annotations for the event type
                success = self.add_annotations_batch(
                    label=event_type,
                    start_times=onset_seconds[indices],
                    end_times=end_seconds[indices],
                    channels=None
                )
                if success:
                    event_counts[event_type] += len(indices)

        # Count EVERY written type: a file holding only Resp/Move/Snore events
        # (or only boundaries) used to be added to the in-memory tree and then
        # never saved, because the save gate summed Artefact + Arousal alone.
        total_count = sum(event_counts.values())

        if total_count > 0:
            self.annotations.save()
            print(
                f"Added {event_counts['Artefact']} artefact annotations and "
                f"{event_counts['Arousal']} arousal annotations from event information. "
                f"{event_counts['Resp']} respiratory events, "
                f"{event_counts['Move']} movement events, "
                f"{event_counts['Snore']} snore events, "
                f"{event_counts['Boundary']} boundary masks "
                f"(+/-{float(self.BOUNDARY_PAD_SECONDS):g}s, written as Artefact)."
            )
        else:
            print("No artefacts, arousals or other exclusions found in event "
                  "information.")

        execution_time = time.time() - start_time
        print(f"Processing time: {execution_time:.4f} seconds")
        return total_count, execution_time



    def add_stages_from_header(self):
        """
        Import stages from header array into annotations using Wonambi's import_staging
        with Compumedics format.

        The rater name applied to the imported staging is taken from the instance
        attribute ``self.rater_name`` set at construction, not from an argument.

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Make sure we have a header with stages
            if not hasattr(self.dataset, 'header') or 'stages' not in self.dataset.header:
                print("No stages found in header")
                return False
                
            # Get stages from header
            stages = self.dataset.header['stages']
            
            # Make sure we have an annotations object
            if not hasattr(self, 'annotations'):
                print("No annotations object available")
                return False
            
            # Get epoch length - either from header or use default 30s
            epoch_length = 30 # default 30sec
            
            # Get recording start time
            if 'start_time' in self.dataset.header:
                rec_start = self.dataset.header['start_time']
            else:
                # Default to current date/time if not available.
                # `datetime` here is the MODULE (see the import at the top), so
                # this needs the class as well; `datetime.now()` raised
                # AttributeError, which the broad except below turned into a
                # silent "return False" -- i.e. a recording that HAS staging
                # but no header start_time lost its stages without a word.
                rec_start = datetime.datetime.now()
            
            # Create a temporary file with Compumedics format staging
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_filename = temp_file.name
                
                # Write stages directly in Compumedics format (one stage code per line)
                for stage_code in stages:
                    # Convert to string and write to file
                    temp_file.write(f"{stage_code}\n")
            
            try:
                # Import the staging using Wonambi's import_staging method
                self.annotations.import_staging(
                    filename=temp_filename,
                    source='compumedics',  # Use compumedics format
                    rater_name=self.rater_name,
                    rec_start=rec_start,
                    staging_start=None,  # Use default (no offset)
                    epoch_length=epoch_length,
                    poor=['Artefact'],  # Default poor quality markers
                    as_qual=False  # Don't import as quality markers
                )
                
                print(f"Successfully imported {len(stages)} stages from header as rater '{self.rater_name}'")
                return True
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_filename)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_filename}: {e}")
                    
        except Exception as e:
            print(f"Error importing stages from header: {e}")
            return False

    def add_annotations_batch(self, label, start_times, end_times, channels=None):
        """Add multiple annotations of one label at once.

        The event type is created if the annotation file does not already carry
        it. Nothing is written to disk here -- the caller is responsible for
        ``save()``.

        Parameters
        ----------
        label : str
            Event type name, e.g. ``'Artefact'``.
        start_times : sequence of float
            Start times in seconds.
        end_times : sequence of float
            End times in seconds, same length as ``start_times``. Pairs are
            zipped, so a shorter sequence silently truncates the batch.
        channels : sequence of str or None, optional
            Channel for each annotation. ``None`` (default) writes every
            annotation on ``'(all)'``.

        Returns
        -------
        bool
            True when the whole batch was added, False if any ``add_event``
            raised -- in which case annotations added before the failure remain
            in the in-memory tree, so the count is all-or-nothing only from the
            caller's point of view.
        """
        try:
            if label not in self.annotations.event_types:
                self.annotations.add_event_type(label)
                
            if channels is None:
                channels = ['(all)'] * len(start_times)
                
            # Add events in batch
            for start, end, chan in zip(start_times, end_times, channels):
                self.annotations.add_event(
                    name=label,
                    time=(float(start), float(end)),
                    chan=chan
                )
            return True
        except Exception as e:
            print(f"Error adding batch annotations: {e}")
            return False    


    def add_annotation(self, label, start_time, end_time, channel=None):
        """
        Add a single annotation to the annotations object.
        
        Parameters
        ----------
        label : str
            Label for the annotation
        start_time : float
            Start time in seconds
        end_time : float
            End time in seconds
        channel : str, list, or None
            Channel(s) associated with the annotation. 
            If None, uses '(all)' to indicate all channels.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Format the time as a tuple of float values
            time_tuple = (float(start_time), float(end_time))

            
            if channel is None:
                channel = '(all)'  # Wonambi standard for all channels

            # Make sure the event type exists
            if label not in self.annotations.event_types:
                self.annotations.add_event_type(label)

            # Add the event with proper rater specification
            self.annotations.add_event(
                name=label, 
                time=time_tuple,
                chan=channel
            )
            return True

        except Exception as e:
            print(f"Error adding annotation: {e}")
            return False


    def process_all(self):
        """Import artefact/arousal events and header staging in one pass.

        Both steps always run. Only the staging step has a pass/fail answer:
        :meth:`add_artefacts_from_events` returns ``(count, seconds)`` and a
        count of zero is a legitimate outcome (the recording simply carries no
        reject flags), so it cannot be folded into a success flag without
        reporting a clean recording as a failure.

        Returns
        -------
        bool
            The result of :meth:`add_stages_from_header`: ``True`` when header
            staging was imported, ``False`` when the recording header carries
            no staging (or the import failed) -- in which case the annotation
            XML is written WITHOUT sleep stages and every stage-filtered
            detection downstream will find nothing. This used to be hardcoded
            ``True``, so a non-GUI caller had no way to see that outcome;
            ``frontend/turtlewave_gui.py`` works around it by calling the two
            steps directly.
        """
        # Artefacts/arousals first: it reports its own counts and timing.
        self.add_artefacts_from_events()

        return self.add_stages_from_header()

    def save(self, filename=None):
        """Save the annotations as Wonambi XML.

        Uses ``Annotations.save()``, which serialises the whole annotation
        tree (raters, epochs, events) back to XML.

        **This used to call** ``Annotations.export(filename)``. Wonambi
        7.15's ``export`` defaults to ``xformat='csv'`` and writes a four-column
        epoch/stage CSV to the path it is given, so calling ``save()`` with the
        default ``annot_file`` OVERWROTE the annotation XML with a stage CSV --
        losing every event and rater, and leaving a file that
        ``Annotations(annot_file)`` can no longer parse. Nothing in this
        repository called it, which is why the trap survived.

        Parameters
        ----------
        filename : str or None
            Path to write to. ``None`` (default) uses the ``annot_file`` given
            at construction. A different path is written as a copy: the object
            keeps pointing at its original file afterwards, so a "save as" does
            not silently redirect every later write.

        Returns
        -------
        bool
            True on success, False if the write raised.
        """
        target = self.annot_file if filename is None else filename

        # Annotations.save() has no target argument -- it writes to
        # self.xml_file -- so a copy is made by retargeting it for the one
        # call and restoring it afterwards, even on failure.
        original = getattr(self.annotations, 'xml_file', None)
        try:
            if original is not None and str(target) != str(original):
                try:
                    self.annotations.xml_file = target
                    self.annotations.save()
                finally:
                    self.annotations.xml_file = original
            else:
                self.annotations.save()
            print(f"Annotations saved to {target}")
            return True
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False


class CustomAnnotations:
    """Helper class for reading and working with Wonambi annotations"""
    
    def __init__(self, annot_file):
        self.annot_file = annot_file
        self.wonb_annot = WonambiAnnotations(annot_file)
        
        # Try to explicitly select a rater if none is selected
        if self.wonb_annot.rater is None and len(self.wonb_annot.raters) > 0:
            self.wonb_annot.get_rater(self.wonb_annot.raters[0])
    @property
    def last_second(self):
        """Return the last second in the recording"""
        return self.wonb_annot.last_second
    
    @property
    def first_second(self):
        """Return the first second in the recording"""
        return self.wonb_annot.first_second
    
    @property
    def dataset(self):
        """Return the dataset associated with the annotations"""
        return self.wonb_annot.dataset
    
    @property
    def rater(self):
        """Return the current rater"""
        return self.wonb_annot.rater
    
    @property
    def raters(self):
        """Return all raters in the annotation file"""
        return self.wonb_annot.raters

    @property
    def epochs(self):
        """Get all epochs from the annotation file"""
        try:
            return list(self.wonb_annot.epochs)
        except IndexError:
            # If no rater is found, find all raters and use the first one
            if len(self.wonb_annot.raters) > 0:
                self.wonb_annot.get_rater(self.wonb_annot.raters[0])
                return list(self.wonb_annot.epochs)
            return []
        
    def get_epochs(self, *args, **kwargs):
        """
        Get epochs that match the specified criteria.
        This method matches the Wonambi API for compatibility.
        
        Returns
        -------
        list of dict
            list of epochs, which are dict with 'start' and 'end' times, plus
            additional parameters
        """
        # Delegate to the underlying Wonambi annotations object
        return self.wonb_annot.get_epochs(*args, **kwargs)

    def get_rater(self, rater):
        """
        Select one rater.
        
        Parameters
        ----------
        rater : str
            name of the rater
        """
        return self.wonb_annot.get_rater(rater)
    def add_rater(self, rater):
        """
        Add one rater.
        
        Parameters
        ----------
        rater : str
            name of the rater
        """
        return self.wonb_annot.add_rater(rater)
    
    def get_stages(self):
        """Extract just the stages from the epochs"""
        epochs = self.epochs
        if epochs:
            return [epoch['stage'] for epoch in epochs]
        return []
    
    def get_hypnogram(self):
        """Convert stages to numeric values for hypnogram plotting"""
        stage_map = {
            'Wake': 0,
            'NREM1': 1,
            'NREM2': 2, 
            'NREM3': 3,
            'REM': 4,
            'Artefact': -1,
            'Movement': -1,
            'Unknown': -1,
            'Undefined': -1
        }
        
        stages = self.get_stages()
        return [stage_map.get(stage, -1) for stage in stages]
    
    def save(self, filename=None):
        """Save the annotations as Wonambi XML.

        ``Annotations.save()`` takes no target and always writes to its own
        ``xml_file``, so this used to accept a ``filename``, ignore it, and
        then print "Annotations saved to <filename>" for a file it had not
        touched. The path is now honoured, and written as a copy: the object
        keeps pointing at its original file afterwards.

        Parameters
        ----------
        filename : str or None
            Path to write to. ``None`` (default) uses the ``annot_file`` given
            at construction.

        Returns
        -------
        bool
            True on success, False if the write raised.
        """
        target = self.annot_file if filename is None else filename

        original = getattr(self.wonb_annot, 'xml_file', None)
        try:
            if original is not None and str(target) != str(original):
                try:
                    self.wonb_annot.xml_file = target
                    self.wonb_annot.save()
                finally:
                    self.wonb_annot.xml_file = original
            else:
                self.wonb_annot.save()
            print(f"Annotations saved to {target}")
            return True
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False

    # Special method for fetch compatibility
    def create_epochs(self, times, epoch_length=30):
        """
        Create epochs from a sequence of time points.
        
        Parameters
        ----------
        times : list or ndarray
            List of time points (in seconds)
        epoch_length : float, optional
            Length of each epoch in seconds
        """
        times = np.asarray(times)
        return self.wonb_annot.create_epochs(times, epoch_length)
    
    # Add method to get time points for a specific stage
    def get_times(self, stage=None, cycle=None, exclude=None):
        """
        Return the times (start and end) for all epochs that match the parameters.
        
        Parameters
        ----------
        stage : str or None
            Stage to match with
        cycle : str or None
            Cycle to match with
        exclude : str or None
            Stage to exclude
            
        Returns
        -------
        list of tuple
            Each tuple contains the start and end time of an epoch
        """
        return self.wonb_annot.get_times(stage=stage, cycle=cycle, exclude=exclude)
            
    # Add any other methods you need to access from the original WonambiAnnotations
    def __getattr__(self, name):
        """Delegate any other method calls to the original WonambiAnnotations object"""
        return getattr(self.wonb_annot, name)