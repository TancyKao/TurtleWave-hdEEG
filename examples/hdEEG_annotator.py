"""
hdEEG_annotator.py - High-Density EEG Annotation Generator
This script demonstrates how to generate annotations for high-density EEG (hdEEG) recordings
using the TurtleWave-hdEEG library built from wonambi. It creates structured annotations.

Features:
- Load large EEG datasets using the `LargeDataset` class.
- Automatically create a 'wonambi' directory for storing annotations.
- Generate annotations for artefacts and sleep stages using the `XLAnnotations` class.
- Save annotations to an XML file for further analysis or visualization.

Workflow:
1. Specify the root directory and the EEG data file name.
2. Load the clean EEG dataset from EEG_processor using the `LargeDataset` class.
3. Create a 'wonambi' subfolder for storing annotations.
4. Define the annotation file path and initialize the `XLAnnotations` class.
5. Process all annotations (e.g., artefacts, arousal, sleep stages) and save them to an XML file.

Outputs:
- An XML file containing the generated annotations, saved in the specified directory.

Example:
    root_dir = "/path/to/eeg_data/"
    datafilename = "subject_eeg.set"


Note:
- Terminal: pip install turtlewave_hdEEG
- TurtleWave-hdEEG library automatic installs the dependencies, inclduing wonambi=7.15.
- The generated annotations can be visualized using the Wonambi GUI or other compatible tools.

"""

from turtlewave_hdEEG import LargeDataset, XLAnnotations #EventViewer, 
import os

## 1. Loading a clean EEG_processor dataset

root_dir = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/turtleRef/"
datafilename = "13PR_clean_rebuilt.set"





datafile = os.path.join(root_dir, datafilename)
data = LargeDataset(datafile, create_memmap=False)

wonambi_dir = os.path.join(root_dir, "wonambi")

## 2. Create wonambi folder if it doesn't exist
if not os.path.exists(wonambi_dir):
    os.makedirs(wonambi_dir)

## 3. Create the annotation file name
# The annotation file name is based on the data file name, with the extension changed to .xml
# and saved in the 'wonambi' directory.
base_name = os.path.splitext(datafilename)[0]
annot_file = os.path.join(wonambi_dir, base_name + ".xml")

## 4. Create the XLAnnotations object
# The XLAnnotations class is used to handle annotations for the dataset.
# It takes the dataset and the annotation file path as arguments.
# The annotations will be saved in the specified XML file.
annotations = XLAnnotations(data, annot_file)
annotations.process_all()

print(f"Annotations saved to {annot_file}")
print(f"ALL DONE, you can check the annotations using wonambi GUI")


# Test event viewer
# annotation_file = "XXXX/synthetic_sleep_eeg_scores.xml"  
# viewer = EventViewer(data, annotation_file)
# viewer.show()