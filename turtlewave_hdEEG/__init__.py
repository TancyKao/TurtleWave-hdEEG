"""
turtlewave_hdEEG - Extended Wonambi for large EEG datasets
"""

__version__ = '1.1.0'

# Import important classes to expose at the package level
from .dataset import LargeDataset
from .visualization import EventViewer
from .annotation import XLAnnotations, CustomAnnotations
from .eventprocessor import ParalEvents
from .swprocessor import ParalSWA
from .extensions import ImprovedDetectSpindle, ImprovedDetectSlowWave
