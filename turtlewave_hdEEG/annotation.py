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


    def add_artefacts_from_events(self):
        """
        Add artefact and arousal annotations from the dataset's event information.
        
        Uses the 'isreject' flag in events to identify artefacts.
        Also identifies arousal events if 'arousal' is in the event type (case-insensitive).
        
        Highly optimized for large datasets by pre-filtering relevant events.
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
        onset_seconds = onsets / s_freq
        duration_seconds = np.ones_like(onsets)
        valid_durations = durations[:len(onset_seconds)]
        duration_seconds[:len(valid_durations)] = np.where(
            valid_durations != None, 
            valid_durations / s_freq, 
            1.0
        )
        end_seconds = onset_seconds + duration_seconds

        # Pre-compile type checks
        types_arr = np.array([str(t).lower() if t else '' for t in types[:len(onsets)]])
    
        event_masks = {
            "Artefact": np.char.find(types_arr, 'reject') != -1,
            "Arousal": np.char.find(types_arr, 'arousal') != -1,
            "Resp": np.any([np.char.find(types_arr, x) != -1 for x in ['hypopnea', 'obstructiveapnea', 'spo2desat']], axis=0),
            "Move": np.any([np.char.find(types_arr, x) != -1 for x in ['move', 'leg']] + [types_arr == x for x in ['lklr', 'lkud']], axis=0),
            "Snore": np.any([np.char.find(types_arr, x) != -1 for x in ['snor', 'jaw']], axis=0)
        }

        event_counts = {key: 0 for key in event_masks}
        
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
            
        total_count = event_counts["Artefact"] + event_counts["Arousal"]
    
        if total_count > 0:
            self.annotations.save()
            print(
                f"Added {event_counts['Artefact']} artefact annotations and "
                f"{event_counts['Arousal']} arousal annotations from event information. "
                f"{event_counts['Resp']} respiratory events, "
                f"{event_counts['Move']} movement events, "
                f"{event_counts['Snore']} snore events."
            )
        else:
            print("No artefacts or arousals found in event information.")

        execution_time = time.time() - start_time
        print(f"Processing time: {execution_time:.4f} seconds")
        return total_count, execution_time



    def add_stages_from_header(self):
        """
        Import stages from header array into annotations using Wonambi's import_staging
        with Compumedics format.
        
        Parameters
        ----------
        rater_name : str
            Name of the rater to use for staging (default: "Automatic_Staging")
        
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
                # Default to current date/time if not available
                rec_start = datetime.now()
            
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
        """Add multiple annotations at once."""
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
        """
        Process all annotations - add artefacts and stages.
        """
        # Add artefacts
        self.add_artefacts_from_events()
        
        stages_from_header = self.add_stages_from_header()
        return True
    
    def save(self, filename=None):
        """
        Save annotations to the XML file in Wonambi format.
        
        Parameters
        ----------
        filename : str or None
            Path to save the file. If None, uses the annot_file from initialization.
        """
        if filename is None:
            filename = self.annot_file
            
        try:    
            self.annotations.export(filename)
            print(f"Annotations saved to {filename}")
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
        """
        Save annotations to the XML file in Wonambi format.
        
        Parameters
        ----------
        filename : str or None
            Path to save the file. If None, uses the annot_file from initialization.
        """
        if filename is None:
            filename = self.annot_file
            
        try:    
            self.wonb_annot.save()
            print(f"Annotations saved to {filename}")
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