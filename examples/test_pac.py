# test_pac.py
import sys
import os

# Set the path to the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your main script
from hdEEG_pac_detector import main

# Monkey patch sys.argv
sys.argv = [
    'hdEEG_pac_detector.py',
    '--channel', 'E110',
    '--stages', 'NREM2NREM3',
    '--sw_method', 'Staresina2015',
    '--spindle_method', 'Moelle2011',
    '--sw_freq_range', '0.3', '2.0',
    '--spindle_freq_range', '9.0', '12.0',
    '--phase_freq', '0.5', '1.25',
    '--amp_freq', '12', '16'
]

# Call the main function
main()