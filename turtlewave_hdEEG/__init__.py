"""
turtlewave_hdEEG - Extended Wonambi for large EEG datasets
"""

__version__ = '4.6.0'

# Opt-in only: with TURTLEWAVE_QUIET_WONAMBI=1 in the environment, silence the
# two Wonambi 7.15 DeprecationWarnings BEFORE Wonambi is imported below (the
# fooof notice cannot be stopped afterwards; see quiet_wonambi_warnings).
# Without the variable nothing is filtered at import. utils imports no Wonambi.
import os as _os
from .utils import quiet_wonambi_warnings, QUIET_WONAMBI_ENV
if _os.environ.get(QUIET_WONAMBI_ENV, '').strip().lower() in ('1', 'true',
                                                              'yes', 'on'):
    quiet_wonambi_warnings()

# Import important classes to expose at the package level
from .dataset import LargeDataset
from .eeglab_io import open_dataset, EEGLABFormatError
from .timeline import RecordingTimeline
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
from .extensions import THRESHOLD_UNITS
from .dbwrite import (ensure_detection_thresholds_schema,
                      store_detection_thresholds, read_detection_thresholds,
                      event_population_summary)
from .event_metrics import EventFigures, PeakFreq, event_figures
from .review_sampling import (draw_review_sample, top_up_region,
                              sample_progress, compute_review_precision,
                              label_agreement, ensure_review_sampling_schema,
                              preview_allocation, read_sample_labels,
                              prepare_population)
from .dbwrite import (export_events_to_csv, default_csv_path, fmt_freq_token,
                      set_journal_mode, VALID_JOURNAL_MODES,
                      resolve_db_target, read_analysed_time,
                      assert_single_subject, subjects_in_database,
                      join_stage_token, split_stage_token, stage_components,
                      stage_tokens_covering, resolve_stage_tokens,
                      pooled_denominator, stage_format,
                      assert_stage_format_compatible)
from .dbwrite import (ensure_event_reviews_schema, store_event_review,
                      delete_event_review,
                      read_event_reviews, ensure_reviewed_view,
                      rematch_orphaned_reviews, review_exclusion_clause,
                      review_category, REVIEW_DECISIONS, REVIEW_REASONS,
                      REVIEW_REASON_CATEGORY)
from .density import event_density, format_density_table
from .utils import (derive_subject, normalize_subject, read_channels_from_csv,
                    region_from_label,
                    DEFAULT_REJECT_TYPES, KNOWN_REJECT_TYPES,
                    resolve_reject_types, reject_key)
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