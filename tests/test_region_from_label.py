#!/usr/bin/env python3
"""``turtlewave_hdEEG.utils.region_from_label`` maps 10-20 / 10-5 labels to regions.

The table is acceptance criterion 1 of ``_scratch/design/compumedics-channels-spec.md``
section 3: every label must map to exactly the region listed. Criterion 2
(counts over the 257 EEG channels of the reference Compumedics file) runs when
that file is reachable and is skipped, with a message, when it is not.

Run standalone: ``python tests/test_region_from_label.py``.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from turtlewave_hdEEG.utils import region_from_label  # noqa: E402

TABLE = {
    'frontal': ['Fp1', 'Fpz', 'FP1h', 'AFp3', 'AF7', 'AFz', 'AFF5h', 'F3', 'Fz', 'F11',
                'F9h', 'FFC3h', 'FFCz'],
    'central': ['FC3', 'FCz', 'FC5h', 'FCC3h', 'FCCz', 'C3', 'Cz', 'C5h', 'CCP5h', 'CP5',
                'CPz', 'CP1h'],
    'parietal': ['CPP3h', 'CPPz', 'P3', 'Pz', 'P9', 'PPO1h', 'PPOz', 'PO3', 'POz', 'PO9',
                 'PO10h'],
    'occipital': ['POO1', 'POOz', 'POO9h', 'O1', 'Oz', 'O1h', 'Iz', 'OCb1h', 'OCB1', 'OCb2'],
    'temporal': ['T7', 'T8h', 'FT9', 'FT11', 'FT7h', 'FFT7h', 'FFt9h', 'FTT9h', 'TTP7h',
                 'TP7', 'TP8h', 'TPP9h'],
    'neck': ['Cb1', 'Cb2', 'Cbz'],
    'other': ['M1', 'M2', 'A1', 'REF', 'Nz', 'E12', 'ECG', 'VEOG', 'EMGChin', 'do_not_use1',
              'SpO2_OSat', '', None],
}

BIPOLAR = {'EEG C3-M2': 'central', 'C4-A1': 'central'}

REFERENCE_FILE = ('/Volumes/Tancy_storage/Z_drug/sub-02dg/ses-1/'
                  'sub-02dg_ses-1_task-psg_run-1_desc-clean_eeg.set')
REFERENCE_COUNTS = {'frontal': 67, 'central': 64, 'parietal': 58, 'temporal': 44,
                    'occipital': 18, 'neck': 3, 'other': 3}


def test_label_table():
    """Every label in the spec table maps to its region."""
    print("\n1. Spec table (acceptance criterion 1):")
    n = 0
    for region, labels in TABLE.items():
        wrong = [(lab, region_from_label(lab)) for lab in labels
                 if region_from_label(lab) != region]
        assert not wrong, f"expected {region}, got {wrong}"
        n += len(labels)
        print(f"   [ok] {region}: {len(labels)} labels")
    for lab, region in BIPOLAR.items():
        assert region_from_label(lab) == region, (lab, region_from_label(lab))
    print(f"   [ok] EDF-style bipolar names {list(BIPOLAR)} read as their first electrode")
    print(f"   [ok] {n + len(BIPOLAR)} labels in total")


def test_parsing_rules():
    """Case, whitespace, prefix stripping and lookalike names."""
    print("\n2. Parsing rules:")
    same = [('cz', 'central'), ('CZ', 'central'), ('fPz', 'frontal'), ('  Cz  ', 'central'),
            ('eeg cz', 'central'), ('EEG Cz-Ref', 'central'), ('eeg Fz-M1', 'frontal'),
            ('FFt9h', 'temporal'), ('ffc3H', 'frontal'), ('Cz-', 'central')]
    for lab, region in same:
        assert region_from_label(lab) == region, (lab, region_from_label(lab), region)
    print(f"   [ok] {len(same)} spelling variants")

    other = ['EEG', 'EEG ', 'Z', 'C', 'Cz1x', 'C3a', '12', 'E1', 'E257', 'VREF', 'ECG_2',
             'BodyPosition_2', 'Chin1', 'Snore', 'Abdomen', 'Thorax', 'Flow', 'Pulse',
             'SpO2', 'M1h', 'A2', 'Nz']
    for lab in other:
        assert region_from_label(lab) == 'other', (lab, region_from_label(lab))
    print(f"   [ok] {len(other)} non-scalp / unparseable labels -> other")

    # non-string input is coerced, not fatal
    assert region_from_label(3) == 'other'
    print("   [ok] non-string input -> other")

    # only the vocabulary the review GUI uses ever comes back
    vocab = {'frontal', 'central', 'parietal', 'temporal', 'occipital', 'neck', 'other'}
    for labels in TABLE.values():
        assert {region_from_label(l) for l in labels} <= vocab
    print("   [ok] results stay within the seven-region vocabulary")


def test_reference_file_counts():
    """Criterion 2: the 257 EEG-typed labels of the reference file."""
    print("\n3. Reference file counts (acceptance criterion 2):")
    if not os.path.exists(REFERENCE_FILE):
        print(f"   [skip] {REFERENCE_FILE} is not reachable")
        return
    from turtlewave_hdEEG.eeglab_io import read_eeglab_channel_info
    info = read_eeglab_channel_info(REFERENCE_FILE)
    eeg = [lab for lab, typ in zip(info['labels'], info['types']) if typ == 'EEG']
    assert len(eeg) == 257, len(eeg)
    counts = {}
    others = []
    for lab in eeg:
        region = region_from_label(lab)
        counts[region] = counts.get(region, 0) + 1
        if region == 'other':
            others.append(lab)
    assert counts == REFERENCE_COUNTS, (counts, REFERENCE_COUNTS)
    assert sorted(others) == ['M1', 'M2', 'REF'], others
    print(f"   [ok] {counts}; other = {sorted(others)}")


if __name__ == "__main__":
    print("TESTING region_from_label")
    print("=========================")
    test_label_table()
    test_parsing_rules()
    test_reference_file_counts()
    print("\nAll region_from_label tests passed.")
