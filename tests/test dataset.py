# Import your dataset module

from turtlewave_hdEEG import LargeDataset

# Path to your original data file (not the rebuilt one)
import os

## 1. Loading a clean EEG_processor dataset

root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/OSA_BL13PR/"
datafilename = "13PR_clean_rebuilt.set"





datafile = os.path.join(root_dir, datafilename)
data = LargeDataset(datafile, create_memmap=False)


# Optional: Create memory map for faster access
# dataset.create_memmap()

# Now you can read data segments
curdata = data.read_data(begtime=0, endtime=10)  # First 10 seconds
