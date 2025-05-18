# gen_dummyHdEEG.py

# Correcting the file saving process
import numpy as np
from scipy.io import savemat

# Generating EEG dataset
np.random.seed(42)
eeg_data = np.random.rand(10, 1000)  # 10 channels, 1000 data points
sampling_rate = 1000  # 1000 Hz

#%%
# Event structure
events = np.array([
    (100, 500, 'wake', 1, 0),
    (600, 500, 'eo', 2, 0),
    (1200, 1500, 'ec', 2, 0),
    (1800, 500, 'blink', 3, 0),
    (2500, 500, 'arousal', 2, 1)
], dtype=[('latency', 'O'), ('duration', 'O'), ('type', 'O'), ('id', 'O'), ('is_reject', 'O')])

#%%
# Sleep stages
stages = ['N2', 'N3', 'REM', 'Wake', 'N1']

# Saving as .mat file
mat_file_path = '/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/TurtleWave/tests/sample_eeg_dataset.mat'
savemat(mat_file_path := '/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/TurtleWave/tests/sample_eeg_dataset.mat', {
    'EEG': {
        'data': eeg_data,
        'srate': sampling_rate,
        'event': events,
        'stages': stages
    }
})