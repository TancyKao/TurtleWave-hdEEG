
# core_functionality.py

# %%
from wonambi.ioeeg.eeglab import EEGLAB

class Dataset:
    def __init__(self, filename):

        if filename.endswith('.set'):
            self.io = EEGLAB(filename)
        else:
            raise ValueError("Unsupported file format.")

        self.s_freq = self.io.header['s_freq']
        self.chan_name = self.io.header['chan_name']

    def read_data(self, begtime=None, endtime=None, chan=None):
        return self.io.return_dat(begtime, endtime, chan)


# usage example:
dataset = Dataset('/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/TurtleWave/tests/synthetic_sleep_eeg.set')
data, times = dataset.read_data(0, 10)
print(data.shape)
# %%
