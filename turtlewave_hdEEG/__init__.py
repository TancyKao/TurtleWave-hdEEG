"""
turtlewave_hdEEG - Extended Wonambi for large EEG datasets
"""

__version__ = '4.3.1'

# Import important classes to expose at the package level
from .dataset import LargeDataset
from .visualization import EventViewer
from .annotation import XLAnnotations, CustomAnnotations
from .eventprocessor import ParalEvents
from .swprocessor import ParalSWA
from .pacprocessor import ParalPAC
from .kcomplexprocessor import ParalKC
from .cycleprocessor import (ParalCycles, detect_cycles,
                             compute_stage_durations,
                             finalize_cycles_and_durations)
from .extensions import (ImprovedDetectSpindle, ImprovedDetectSlowWave,
                         ImprovedDetectKComplex)
from .dbwrite import (export_events_to_csv, default_csv_path, fmt_freq_token,
                      set_journal_mode, VALID_JOURNAL_MODES,
                      resolve_db_target, read_analysed_time,
                      assert_single_subject, subjects_in_database,
                      join_stage_token, split_stage_token, stage_components,
                      stage_tokens_covering, resolve_stage_tokens,
                      pooled_denominator, stage_format,
                      assert_stage_format_compatible)
from .density import event_density, format_density_table
from .utils import derive_subject, normalize_subject, read_channels_from_csv
from .rerun import (RerunGuardError, verify_rater_match, channel_clean_gate,
                    resolve_rerun_params, resolve_sw_amplitude_thresholds)

# Cycle plotting pulls in matplotlib; keep the import defensive so a missing or
# broken matplotlib never breaks `import turtlewave_hdEEG` (mirrors the optional
# GUI import below).
try:
    from .cycleplot import plot_hypnogram_cycles, plot_from_annotations
except ImportError:
    plot_hypnogram_cycles = None
    plot_from_annotations = None



try:
    from .frontend import event_review_main, EventReviewInterface
    EVENT_REVIEW_AVAILABLE = True
except ImportError as e:
    EVENT_REVIEW_AVAILABLE = False
    event_review_main = None
    EventReviewInterface = None