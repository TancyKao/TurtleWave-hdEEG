"""Illustrative figure of slow-wave x spindle phase-amplitude coupling (PAC).

This is a self-contained synthetic demo of what
``turtlewave_hdEEG.pacprocessor.ParalPAC.analyze_pac`` measures when called with
``event_type='slow_wave', pair_with_spindles=True``: does the spindle-band
amplitude (11-16 Hz) cluster at a preferred phase of the slow oscillation
(0.5-1.25 Hz)?

No real data or events database is needed. We synthesise a slow oscillation with
a spindle whose envelope is locked to the SO up-state, recover phase and
amplitude the same way ``analyze_pac`` does (band-pass + Hilbert), bin the
spindle amplitude across SO phase with the same logic as
``ParalPAC._mean_amp``, and draw the polar coupling rose in the style of
``ParalPAC.compare_conditions``.

Run with the project venv:
    .venv/bin/python docs/images/sw_spindle_coupling.py

Outputs PNG (raster, for slides) and PDF (vector, for handouts).

Two faithfulness notes versus the library:
* ``analyze_pac`` filters via tensorpac; here we use a zero-phase Butterworth in
  second-order-sections form (``sosfiltfilt``), because a plain ``filtfilt`` is
  numerically unstable at the 0.5 Hz edge and diverges.
* ``analyze_pac`` defines its bin centres on ``[0, 2*pi)`` (``vecbin``) while it
  bins phase on ``[-pi, pi]`` -- a half-turn offset that shifts its reported
  ``preferred_phase_deg`` by 180 deg. We keep centres consistent with the bin
  edges so the preferred phase here reads as the true SO phase (up-state ~ 0 deg).
"""
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Parameters (mirror the library's PAC defaults) --------------------
FS = 500                      # Hz
DURATION = 60.0               # s  -> ~45 slow-oscillation cycles
SO_FREQ = 0.75                # Hz  slow oscillation
SO_AMP = 60.0                 # uV
SPINDLE_FREQ = 14.0           # Hz  carrier inside the 11-16 Hz spindle band
SPINDLE_AMP = 18.0            # uV  spindle amplitude at full coupling

PHASE_BAND = (0.5, 1.25)      # analyze_pac default phase_freq
AMP_BAND = (11.0, 16.0)       # analyze_pac default amp_freq
NBINS = 18                    # analyze_pac / _mean_amp default

# Coupling: the spindle envelope is concentrated near the SO up-state (the
# positive peak, where analytic SO phase ~ 0). KAPPA sets how tight the locking
# is; COUPLING_GAIN=0 gives the uncoupled negative control (flat rose, MI ~ 0).
KAPPA = 3.0
COUPLING_GAIN = 1.0
ENV_FLOOR = 0.15              # background spindle amplitude away from the up-state

RNG = np.random.default_rng(7)


def bandpass(sig, lo, hi, fs, order=4):
    """Zero-phase Butterworth band-pass via second-order sections.

    SOS form is used instead of (b, a) + filtfilt because a 4th-order band-pass
    with a 0.5 Hz lower edge at fs=500 has poles close to the unit circle and
    filtfilt diverges to overflow.
    """
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, sig)


def synth_signal():
    """Slow oscillation + spindle locked to the SO up-state + noise."""
    t = np.arange(0, DURATION, 1 / FS)
    wt = 2 * np.pi * SO_FREQ * t

    # Slow oscillation. With +sin, the analytic phase is 0 at the positive
    # (up-state) peak and +/-pi at the trough.
    so = SO_AMP * np.sin(wt)

    # Spindle envelope: von-Mises-shaped bump centred on the up-state (wt=pi/2,
    # the positive SO peak), riding on a small constant floor.
    env = SPINDLE_AMP * (
        ENV_FLOOR + COUPLING_GAIN * np.exp(KAPPA * (np.cos(wt - np.pi / 2) - 1.0))
    )
    spindle = env * np.sin(2 * np.pi * SPINDLE_FREQ * t)

    noise = 8.0 * RNG.standard_normal(t.size)
    return t, so + spindle + noise


def main():
    t, sig = synth_signal()

    # Recover phase and amplitude as analyze_pac does (band-pass + Hilbert).
    so_filt = bandpass(sig, *PHASE_BAND, FS)
    sp_filt = bandpass(sig, *AMP_BAND, FS)
    so_phase = np.angle(hilbert(so_filt))     # [-pi, pi]
    sp_amp = np.abs(hilbert(sp_filt))         # spindle envelope

    # --- Bin spindle amplitude across SO phase (ParalPAC._mean_amp logic) ---
    width = 2 * np.pi / NBINS
    edges = np.linspace(-np.pi, np.pi, NBINS + 1)
    idx = np.digitize(so_phase, edges) - 1
    idx[idx == NBINS] = 0
    ampbin = np.array([sp_amp[idx == i].mean() if np.any(idx == i) else 0.0
                       for i in range(NBINS)])

    # Normalise to a probability distribution (analyze_pac line 715).
    p = ampbin / ampbin.sum()

    # Bin centres consistent with the [-pi, pi] edges (see module docstring).
    centres = -np.pi + (np.arange(NBINS) + 0.5) * width

    # Preferred phase = amplitude-weighted circular mean; mean vector length.
    vec = np.sum(p * np.exp(1j * centres))
    pref = np.angle(vec)
    mvl = np.abs(vec)
    pref_deg = (np.degrees(pref) + 180) % 360 - 180   # report in [-180, 180]

    # Tort Modulation Index: MI = (log N - H) / log N, H = -sum p log p.
    H = -np.sum(p * np.log(p))
    mi = (np.log(NBINS) - H) / np.log(NBINS)

    # ---- Figure --------------------------------------------------------
    fig = plt.figure(figsize=(12, 4.5))
    axA = fig.add_subplot(1, 2, 1)
    axB = fig.add_subplot(1, 2, 2, projection="polar")

    # Panel A: ~3 s window over a couple of SO cycles with nested spindles.
    w0, w1 = 5.0, 8.0
    m = (t >= w0) & (t <= w1)
    tt, raw = t[m], sig[m]
    so_w = so_filt[m]
    env_w = sp_amp[m]

    axA.plot(tt, raw, lw=0.8, color="0.65", label="Broadband EEG")
    axA.plot(tt, so_w, lw=2.4, color="steelblue", label="Slow osc. (0.5-1.25 Hz)")
    axA.plot(tt, env_w, lw=1.8, color="crimson",
             label="Spindle envelope (11-16 Hz)")
    axA.plot(tt, -env_w, lw=1.8, color="crimson")

    # Mark an up-state (positive SO peak) inside the window.
    pk = tt[np.argmax(so_w)]
    axA.axvline(pk, color="darkgreen", lw=1, ls="--")
    axA.annotate("spindle nested\nat SO up-state",
                 xy=(pk, env_w.max()), xytext=(pk + 0.2, env_w.max() + 28),
                 fontsize=9, color="darkgreen",
                 arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2))

    axA.axhline(0, color="gray", lw=0.5)
    axA.set_xlim(w0, w1)
    axA.set_xlabel("Time (s)")
    axA.set_ylabel("Amplitude (µV)")
    axA.set_title("A. Slow wave with a nested spindle", loc="left", fontsize=11)
    axA.legend(loc="lower right", fontsize=8, frameon=False)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # Panel B: phase-amplitude coupling rose.
    axB.bar(centres, p, width=width, bottom=0.0,
            color="steelblue", edgecolor="white", alpha=0.85)
    axB.plot([pref, pref], [0, p.max() * 1.1], color="crimson", lw=2.5,
             solid_capstyle="round")
    axB.set_theta_zero_location("E")   # 0 deg (SO up-state) at the right
    axB.set_theta_direction(1)
    axB.set_yticklabels([])
    axB.set_xlabel("SO phase (0° = up-state, ±180° = down-state)", fontsize=9)
    axB.set_title("B. Spindle amplitude by SO phase", loc="left", fontsize=11)
    axB.text(-0.05, -0.02,
             f"pref. phase = {pref_deg:.0f}°\n"
             f"MVL = {mvl:.2f}\n"
             f"MI = {mi:.3f}",
             transform=axB.transAxes, fontsize=9, ha="left", va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "sw_spindle_coupling.png")
    pdf = os.path.join(OUT_DIR, "sw_spindle_coupling.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Preferred phase: {pref_deg:.1f} deg | MVL: {mvl:.3f} | MI: {mi:.4f}")


if __name__ == "__main__":
    main()
