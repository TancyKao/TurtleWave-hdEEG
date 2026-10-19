# How to Upgrade to turtlewave-hdEEG 4.6.3

4.6.3 corrects four defaults and selections that changed results silently. No
schema changes. This guide covers which stored results to re-run.

## Before you start

```bash
cp neural_events.db neural_events.db.bak-pre-4.6.3
```

## What changed

- **PAC event selection matches the band exactly.** `analyze_pac` selects events
  with `freq_lower = lo AND freq_upper = hi` for `sw_freq_range` and
  `spindle_freq_range` in `event_opts`. A band that was never stored selects
  nothing and logs a warning listing the stored bands. Before 4.6.3 it pooled
  every stored band inside the range.
- **`generate_comodulogram` works** and returns the map and surrogate p-values.
  It failed on every call before.
- **No surrogates, no p-value.** `idpac` with `idpac[1] == 0` stores NaN in
  `median_mi_pval` instead of aborting the channel.
- **`ParalSWA.detect_slow_waves(frequency=None)` resolves the band per method:**
  Ngo2015 (0.5, 3.5) Hz, Staresina2015 (0.5, 1.25) Hz, Massimini family
  (0.1, 4) Hz. The old default (0.1, 4) ran Ngo2015 and Staresina2015 with a
  4 Hz low-pass. New helpers in `turtlewave_hdEEG.utils`: `published_sw_band`,
  `published_sw_duration`, `resolve_sw_frequency`, `PUBLISHED_BAND_SW_METHODS`,
  `DEFAULT_MASSIMINI_SW_BAND`.
- **`ParalKC.detect_kcomplexes` criteria default to `None`**, meaning the
  method's own: `AASM/Massimini2004` is unchanged ((0.25, 1.0) s, -40 µV,
  75 µV); `Massimini2004` now uses (0.3, 1.0) s, -80 µV, 140 µV. Before, the AASM
  numbers applied to every method.

## Re-run what is affected

1. **PAC results** from a database that holds more than one run of the same
   method at nested bands (for example spindles at 11-16 Hz and 12-15 Hz). Those
   runs were pooled into one estimate. Re-run `analyze_pac` with the band you
   intend:

    ```python
    event_opts = {
        'sw_method': 'Staresina2015', 'sw_freq_range': (0.5, 1.25),
        'spindle_method': 'Moelle2011', 'spindle_freq_range': (11, 16),
    }
    ```

    A database with one band per method is unaffected. Check what is stored:

    ```python
    import sqlite3

    conn = sqlite3.connect('neural_events.db')
    print(conn.execute(
        "SELECT event_type, method, freq_lower, freq_upper, COUNT(*) "
        "FROM events GROUP BY 1, 2, 3, 4").fetchall())
    ```

2. **Ngo2015 and Staresina2015 slow waves** detected with the default
   `frequency`. Re-detect to get the published low-pass, or keep the old
   behaviour by passing `frequency=(0.1, 4)`. Do not pool old and new counts.
3. **Massimini2004 K-complexes** detected with no explicit criteria. Re-detect;
   do not pool with older rows. `AASM/Massimini2004` runs are unchanged.
4. **Scripts that pass a PAC band** which was never stored (including one that
   only lies inside a stored band) now select no events. Pass the stored band.

## See also

- [Run PAC analysis](run-pac-analysis.md#select-events-by-the-exact-stored-band)
- [Detect slow waves](detect-slow-waves.md#choosing-the-detection-band)
- [Detect K-complexes](detect-kcomplexes.md)
- [Upgrade to 4.6](upgrade-to-4.6.md)
