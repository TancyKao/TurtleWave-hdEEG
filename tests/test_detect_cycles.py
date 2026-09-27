"""Unit tests for ``detect_cycles`` (the pure hypnogram-in, cycles-out rule).

Codes: 0 Wake, 1 N1, 2 N2, 3 N3, 4 REM, -1 unscored. Thresholds are the
library defaults unless stated: ``wake_thresh=10``, ``nrem_min=30``,
``rem_min=10``, ``completion_min=10`` epochs.
"""
import math

import pytest

from turtlewave_hdEEG.cycleprocessor import detect_cycles


def _spans(cycles):
    return [(c['nrem_start_epoch'], c['nrem_end_epoch'],
             c['rem_start_epoch'], c['rem_end_epoch']) for c in cycles]


# --------------------------------------------------------------------------
# '2022' (modified rule)
# --------------------------------------------------------------------------

def test_2022_n2n3_onset_skips_leading_n1():
    hyp = [1] * 6 + [2] * 40 + [4] * 12 + [1] * 3 + [2] * 40 + [4] * 12
    n23 = detect_cycles(hyp, nrem_onset='n2n3')
    anyo = detect_cycles(hyp, nrem_onset='any')
    assert _spans(n23) == [(6, 45, 46, 60), (61, 100, 101, 112)]
    assert _spans(anyo) == [(0, 45, 46, 57), (58, 100, 101, 112)]
    # Leading N1 of period 2 belongs to segment 1 under N2/N3 onset.
    assert n23[0]['rem_sleep_min'] == 6.0 and n23[0]['rem_dur_min'] == 7.5


def test_2022_default_onset_is_n2n3():
    hyp = [1] * 6 + [2] * 40 + [4] * 12
    assert detect_cycles(hyp)[0]['nrem_start_epoch'] == 6


def test_2022_nrem_run_must_exceed_nrem_min():
    assert detect_cycles([2] * 30 + [4] * 12) == []
    assert len(detect_cycles([2] * 31 + [4] * 12)) == 1


def test_2022_n2n3_length_is_measured_from_the_n2_onset():
    # 5 N1 + 28 N2 = 33-epoch run passes under 'any' but the N2 part is only
    # 28 epochs, so it is dropped under 'n2n3'.
    hyp = [1] * 5 + [2] * 28 + [4] * 12
    assert len(detect_cycles(hyp, nrem_onset='any')) == 1
    assert detect_cycles(hyp, nrem_onset='n2n3') == []


def test_2022_wake_absorbed_up_to_threshold():
    base = [2] * 20 + [4] * 12
    absorbed = detect_cycles([2] * 20 + [0] * 10 + base)
    broken = detect_cycles([2] * 20 + [0] * 11 + base)
    assert _spans(absorbed) == [(0, 49, 50, 61)]
    assert absorbed[0]['nrem_sleep_min'] == 20.0        # wake subtracted
    assert absorbed[0]['nrem_dur_min'] == 25.0          # clock time
    assert broken == []                                  # two 20-epoch runs


def test_2022_last_segment_stops_at_last_sleep_epoch():
    hyp = [2] * 40 + [4] * 12 + [0] * 8
    c = detect_cycles(hyp)
    assert _spans(c) == [(0, 39, 40, 51)]
    assert c[0]['rem_end_sec'] == 52 * 30
    assert c[0]['wake_in_seg_min'] == 0.0


def test_2022_complete_flag_on_final_cycle():
    assert detect_cycles([2] * 40 + [4] * 10)[0]['complete'] is True
    assert detect_cycles([2] * 40 + [4] * 9)[0]['complete'] is False
    two = detect_cycles([2] * 40 + [4] * 3 + [2] * 40 + [4] * 3)
    assert [c['complete'] for c in two] == [True, False]


def test_2022_rem_class():
    c = detect_cycles([2] * 40 + [4] * 10 + [2] * 40 + [4] * 3 + [2] * 40)
    assert [x['rem_class'] for x in c] == ['full', 'short', 'none']
    assert c[2]['rem_start_epoch'] > c[2]['rem_end_epoch']   # empty segment


def test_2022_unscored_is_wake():
    hyp_wake = [2] * 20 + [0] * 5 + [2] * 20 + [4] * 12
    hyp_unsc = [2] * 20 + [-1] * 5 + [2] * 20 + [4] * 12
    hyp_nan = [2] * 20 + [math.nan] * 5 + [2] * 20 + [4] * 12
    assert _spans(detect_cycles(hyp_unsc)) == _spans(detect_cycles(hyp_wake))
    assert _spans(detect_cycles(hyp_nan)) == _spans(detect_cycles(hyp_wake))


# --------------------------------------------------------------------------
# '1979' (Feinberg & Floyd)
# --------------------------------------------------------------------------

def test_1979_wake_never_breaks_a_nrem_period():
    # 20-epoch wake inside NREM breaks '2022' but not Feinberg & Floyd.
    hyp = [2] * 20 + [0] * 20 + [2] * 20 + [4] * 12
    assert detect_cycles(hyp, method='2022') == []
    c = detect_cycles(hyp, method='1979')
    assert _spans(c) == [(0, 59, 60, 71)]
    assert c[0]['nrem_dur_min'] == 30.0 and c[0]['nrem_sleep_min'] == 20.0


def test_1979_merges_rem_runs_across_gaps_under_nrem_min():
    tail = [2] * 40 + [4] * 12
    merged = detect_cycles([2] * 40 + [4] * 6 + [2] * 29 + [4] * 6 + tail,
                           method='1979')
    split = detect_cycles([2] * 40 + [4] * 6 + [2] * 30 + [4] * 6 + tail,
                          method='1979')
    # Merged: one REM period of 12 REM epochs spanning the 29-epoch NREM gap.
    assert _spans(merged) == [(0, 39, 40, 80), (81, 120, 121, 132)]
    assert merged[0]['rem_sleep_min'] == 6.0
    # Split: the second 6-epoch REM run is too short after REMP1 and is
    # absorbed into NREM period 2; only two cycles, NREMP2 holds 3 REM min.
    assert _spans(split) == [(0, 39, 40, 45), (46, 121, 122, 133)]
    assert split[1]['rem_in_nremp_min'] == 3.0


def test_1979_wake_in_the_gap_does_not_count_as_nrem():
    # 29 NREM + 10 wake between two REM runs: still fewer than 30 NREM.
    hyp = [2] * 40 + [4] * 6 + [2] * 15 + [0] * 10 + [2] * 14 + [4] * 6 + [2] * 40 + [4] * 12
    c = detect_cycles(hyp, method='1979')
    assert len(c) == 2 and c[0]['rem_end_epoch'] == 90


def test_1979_first_rem_period_exempt_from_rem_min():
    hyp = [2] * 40 + [4] * 1 + [2] * 40 + [4] * 12
    c = detect_cycles(hyp, method='1979')
    assert _spans(c) == [(0, 39, 40, 40), (41, 80, 81, 92)]


def test_1979_short_rem_after_remp1_does_not_start_a_cycle():
    hyp = [2] * 40 + [4] * 12 + [2] * 40 + [4] * 9 + [2] * 40 + [4] * 12
    c = detect_cycles(hyp, method='1979')
    assert _spans(c) == [(0, 39, 40, 51), (52, 140, 141, 152)]
    assert c[1]['rem_in_nremp_min'] == 4.5


def test_1979_sleep_onset_rem_is_absorbed_and_flagged():
    hyp = [2] * 10 + [4] * 12 + [2] * 40 + [4] * 12
    c = detect_cycles(hyp, method='1979')
    assert _spans(c) == [(0, 61, 62, 73)]
    assert c[0]['sorem'] is True and c[0]['rem_in_nremp_min'] == 6.0


def test_1979_starts_at_stage_2_onset():
    hyp = [1] * 8 + [2] * 40 + [4] * 12
    c = detect_cycles(hyp, method='1979')
    assert c[0]['nrem_start_epoch'] == 8
    # 'nrem_onset' is ignored for '1979'.
    assert detect_cycles(hyp, method='1979', nrem_onset='any')[0]['nrem_start_epoch'] == 8


def test_1979_trailing_nrem_period_kept_only_if_long_enough():
    with_tail = detect_cycles([2] * 40 + [4] * 12 + [2] * 30, method='1979')
    short_tail = detect_cycles([2] * 40 + [4] * 12 + [2] * 29, method='1979')
    assert len(with_tail) == 2
    assert with_tail[1]['complete'] is False
    assert with_tail[1]['rem_start_epoch'] > with_tail[1]['rem_end_epoch']
    assert with_tail[0]['complete'] is True       # >= 10 NREM epochs follow
    assert len(short_tail) == 1
    assert short_tail[0]['rem_end_epoch'] == 80    # segment runs to last sleep
    assert short_tail[0]['complete'] is True


def test_1979_last_rem_period_incomplete_without_following_nrem():
    c = detect_cycles([2] * 40 + [4] * 12 + [2] * 9, method='1979')
    assert len(c) == 1 and c[0]['complete'] is False


def test_1979_nrem_period_needs_nrem_min_sleep_epochs():
    fx = [2] * 15 + [0] * 8 + [2] * 40 + [4] * 15 + [2] * 40 + [4] * 15 + [0] * 5
    assert detect_cycles(fx, method='1979', nrem_min=200) == []
    assert len(detect_cycles(fx, method='1979')) == 2


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------

def test_empty_and_no_sleep_inputs():
    for m in ('2022', '1979'):
        assert detect_cycles([], method=m) == []
        assert detect_cycles([0] * 50, method=m) == []
        assert detect_cycles([-1] * 50, method=m) == []


def test_bad_arguments():
    with pytest.raises(ValueError):
        detect_cycles([2] * 40, method='1994')
    with pytest.raises(ValueError):
        detect_cycles([2] * 40, nrem_onset='n3')
    with pytest.raises(ValueError):
        detect_cycles([2] * 40, epoch_starts=[0.0])


def test_epoch_starts_respected():
    starts = [i * 20.0 for i in range(52)]
    c = detect_cycles([2] * 40 + [4] * 12, epoch_length=20, epoch_starts=starts)
    assert c[0]['nrem_end_sec'] == 800.0 and c[0]['rem_end_sec'] == 1040.0
    assert c[0]['nrem_dur_min'] == pytest.approx(40 * 20 / 60, abs=1e-3)
