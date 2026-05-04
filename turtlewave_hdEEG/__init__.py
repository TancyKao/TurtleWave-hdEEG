"""
turtlewave_hdEEG - Extended Wonambi for large EEG datasets
"""

__version__ = '3.3.0'

# Import important classes to expose at the package level
from .dataset import LargeDataset
from .visualization import EventViewer
from .annotation import XLAnnotations, CustomAnnotations
from .eventprocessor import ParalEvents
from .swprocessor import ParalSWA
from .pacprocessor import ParalPAC
from .kcomplexprocessor import ParalKC
from .extensions import (ImprovedDetectSpindle, ImprovedDetectSlowWave,
                         ImprovedDetectKComplex)



try:
    from .frontend import event_review_main, EventReviewInterface
    EVENT_REVIEW_AVAILABLE = True
except ImportError as e:
    EVENT_REVIEW_AVAILABLE = False
    event_review_main = None
    EventReviewInterface = None