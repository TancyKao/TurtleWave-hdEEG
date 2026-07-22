"""Workflow ecosystem diagram: raw EEG -> event detection -> statistical analysis.

Maps the full pipeline across three tools so the missing statistical-analysis
stage is unmistakable for an NCI design discussion:

    1. EEG_Processor      (MATLAB/EEGLAB, separate project)  -- preprocess + BIDS
    2. turtlewave_hdEEG   (this repo, Python)                -- detect + export
    3. Statistical analysis                                  -- mostly PROPOSED

Visual convention:
    solid box  = built today
    dashed box = proposed / to build (accent colour)

Run with the project venv:
    .venv/bin/python docs/images/workflow_ecosystem.py

Outputs PNG (raster, dpi=200) and PDF (vector) for both an overview and a
detailed ecosystem figure. No graphviz/mermaid dependency; pure matplotlib.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- palette ------------------------------------------------------------
BUILT_FC, BUILT_EC = "#dbe9f6", "#2b6cb0"        # blue  = built
PROP_FC, PROP_EC = "#fde6cf", "#c05621"          # orange = proposed
LANE_EC = "#9aa5b1"                              # lane container outline
HANDOFF = "#2f855a"                             # green = inter-tool handoff


def box(ax, cx, cy, w, h, text, proposed=False, fontsize=9, bold=False,
        fc=None, ec=None):
    """Rounded box centred at (cx, cy). Dashed accent if proposed."""
    fc = fc or (PROP_FC if proposed else BUILT_FC)
    ec = ec or (PROP_EC if proposed else BUILT_EC)
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        linewidth=1.6, edgecolor=ec, facecolor=fc,
        linestyle="--" if proposed else "-", zorder=2,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            zorder=3, fontweight="bold" if bold else "normal", color="#1a202c")
    return (cx, cy, w, h)


def arrow(ax, x1, y1, x2, y2, color="#4a5568", lw=1.8, label=None,
          ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, linestyle=ls, zorder=1,
        shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.4, label, ha="center",
                va="bottom", fontsize=7.5, color=color, style="italic")


def legend(ax, x, y):
    box(ax, x, y, 5.5, 3, "built", fontsize=8)
    box(ax, x + 14, y, 5.5, 3, "proposed", proposed=True, fontsize=8)
    ax.text(x + 3.4, y, "  = exists today", ha="left", va="center", fontsize=8)
    ax.text(x + 17.4, y, "  = to build", ha="left", va="center", fontsize=8)


# =========================================================================
# Figure 1 -- Overview (single left -> right flow)
# =========================================================================
fig1, ax = plt.subplots(figsize=(15, 4.2))
ax.set_xlim(0, 112)
ax.set_ylim(0, 30)
ax.axis("off")

y = 17
w, h = 15.5, 9
cx = [10, 30, 50, 70, 90, 104]
stages = [
    ("Preprocess\n\nEEG_Processor\n(MATLAB/EEGLAB)", False),
    ("Input\n\nLargeDataset +\nXLAnnotations", False),
    ("Detect\n\nspindle / slow wave\n/ PAC", False),
    ("Export\n\nparameters +\ndensity CSV", False),
    ("neural_events.db\n\nSQLite\n(events table)", False),
    ("Statistical\nanalysis\n\nSnPM_2025\n(MATLAB · perm/TFCE)", False),
]
# last box a touch narrower spacing handled via cx
cx = [9, 28.5, 48, 67.5, 87, 104]
boxes = []
for x, (txt, prop) in zip(cx, stages):
    bw = 16 if x != 104 else 14
    boxes.append(box(ax, x, y, bw, h, txt, proposed=prop, fontsize=8.5,
                     bold=True))

for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + boxes[i][2] / 2
    x2 = boxes[i + 1][0] - boxes[i + 1][2] / 2
    if i == 0:                                   # tool handoff: preproc -> detect
        col, lw, lbl, ls = HANDOFF, 2.6, ".set/.fdt +\nevents.tsv", "-"
    elif i == len(boxes) - 2:                    # the missing link to SnPM
        col, lw, lbl, ls = PROP_EC, 2.4, "importer\n(TBD)", "--"
    else:
        col, lw, lbl, ls = "#4a5568", 1.8, None, "-"
    arrow(ax, x1, y, x2, y, color=col, lw=lw, label=lbl, ls=ls)

ax.text(56, 28.5, "Sleep hd-EEG pipeline — overview",
        ha="center", fontsize=14, fontweight="bold")
legend(ax, 8, 4)
fig1.tight_layout()

p1_png = os.path.join(OUT_DIR, "workflow_overview.png")
p1_pdf = os.path.join(OUT_DIR, "workflow_overview.pdf")
fig1.savefig(p1_png, dpi=200, bbox_inches="tight")
fig1.savefig(p1_pdf, bbox_inches="tight")


# =========================================================================
# Figure 2 -- Detailed ecosystem (three tool-lanes)
# =========================================================================
fig2, ax = plt.subplots(figsize=(15, 12))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.text(50, 97.5, "Sleep hd-EEG ecosystem — raw signal to statistics",
        ha="center", fontsize=15, fontweight="bold")
legend(ax, 6, 93)


def lane(cy, h, title, compute):
    ax.add_patch(FancyBboxPatch(
        (3, cy - h / 2), 94, h,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        linewidth=1.4, edgecolor=LANE_EC, facecolor="#f7fafc", zorder=0))
    ax.text(5.5, cy + h / 2 - 2.6, title, ha="left", va="center",
            fontsize=12, fontweight="bold", color="#2d3748")
    ax.text(94.5, cy + h / 2 - 2.6, compute, ha="right", va="center",
            fontsize=9, style="italic", color="#718096")


# ---- Lane A: EEG_Processor ---------------------------------------------
A = 76
lane(A, 22, "1 · EEG_Processor", "runs LOCAL · MATLAB / EEGLAB")
ay = A - 1
box(ax, 14, ay, 17, 9,
    "Raw input\n\n.mff (EGI)\n.edf (Compumedics)\nGeoscan .txt coords\nhypnogram", fontsize=8)
box(ax, 35, ay, 15, 8, "Import +\nBIDS convert", fontsize=8.5, bold=True)
box(ax, 54, ay, 15, 8, "Mark bad\nchannels / epochs\n(FASTER / manual)", fontsize=8)
box(ax, 73, ay, 15, 8, "Filter / resample\nre-reference / ICA", fontsize=8.5, bold=True)
box(ax, 90, ay, 13, 9,
    "BIDS output\n\n.set / .fdt\nevents.tsv\nchannels.tsv\nelectrodes.tsv", fontsize=7.8)
for x1, x2 in [(22.5, 27.5), (42.5, 46.5), (61.5, 65.5), (80.5, 83.5)]:
    arrow(ax, x1, ay, x2, ay)

# ---- Lane B: turtlewave_hdEEG ------------------------------------------
B = 47
lane(B, 24, "2 · turtlewave_hdEEG", "runs LOCAL or NCI Gadi (PBS) · Python")
by = B + 1.5
box(ax, 14, by, 17, 9,
    "Input\n\nLargeDataset\n(.set/.fdt/.mat/v7.3)\n+ XLAnnotations\n(Wonambi XML)", fontsize=7.8)
box(ax, 38, by, 19, 11,
    "Detect (per channel)\n\nParalEvents — spindles\nParalSWA — slow waves\nParalPAC — coupling", fontsize=8, bold=True)
box(ax, 62, by, 15, 9,
    "Per-channel JSON\n\n{event}_{method}_\n{flo}-{fhi}Hz_{stages}", fontsize=7.6)
box(ax, 82, by, 15, 9,
    "Export CSV\n\nparameters +\ndensity\n(import to DB)", fontsize=8)
arrow(ax, 22.5, by, 28.5, by)
arrow(ax, 47.5, by, 54.5, by)
arrow(ax, 69.5, by, 74.5, by)
# DB box centred below export
box(ax, 82, by - 8.5, 15, 5.5,
    "neural_events.db\n(events / processing_status)", fontsize=7.8, bold=True,
    fc="#c6dcef")
arrow(ax, 82, by - 4.5, 82, by - 5.8)
# HPC note
ax.text(38, by - 8.2,
        "HPC batch: submit_all_*.sh → qsub per subject (subjects.txt) → *_GADI.py",
        ha="center", fontsize=7.8, style="italic", color="#718096")

# ---- Lane C: SnPM_2025 (statistical analysis) --------------------------
C = 15
lane(C, 24, "3 · SnPM_2025  (statistical analysis)",
     "runs LOCAL or HPC · MATLAB R2025a · SnPM / permutation")
# two input paths (top = built spectral, bottom = proposed event importer)
box(ax, 18, 22.5, 23, 7,
    "Spectral input  (works)\nload_spectral_dataset\n*_powerspect.mat → wide CSV", fontsize=7.4)
box(ax, 18, 12, 23, 8,
    "Event input — importer (TBD)\nevent CSV / neural_events.db\n+ electrodes.tsv → wide table",
    proposed=True, fontsize=7.4, bold=True)
# built SnPM pipeline
box(ax, 45, 17, 19, 12,
    "Analysis design\n\ncore_snpm_analysis (t / corr)\ncore_snpm_glm (anova1 /\nancova / regression /\nrmanova / mixed2way)\ncore_snpm_lmm (event-level)", fontsize=7.2)
box(ax, 66, 17, 15, 12,
    "Stat engine\n\nper-channel t / F / r →\npermutation null\n(sign-flip /\nFreedman-Lane)\n→ TFCE + cluster FWE", fontsize=7.2, bold=True)
box(ax, 85, 17, 15, 12,
    "Output\n\n.mat + .xlsx tables\n.html reports\ntopographic scalp\nmaps (.png)", fontsize=7.4)
arrow(ax, 29.5, 21.5, 35.3, 18.6, color=HANDOFF)
arrow(ax, 29.5, 12.5, 35.3, 15.4, color=PROP_EC, ls="--")
arrow(ax, 54.5, 17, 58.3, 17)
arrow(ax, 73.5, 17, 77.3, 17)
ax.text(60, 5.2,
        "Spectral path already works (load_spectral_dataset → wide CSV).  "
        "Still to modify: event-table importer (not yet coded), "
        "GUI wiring for GLM/LMM presets, normalized band power (upstream fix).",
        ha="center", fontsize=7.2, style="italic", color="#c05621")

# ---- inter-lane handoff arrows -----------------------------------------
# 1) EEG_Processor clean BIDS output -> turtlewave input
arrow(ax, 90, A - 6.0, 14, B + 7.2, color=HANDOFF, lw=2.8,
      label=".set/.fdt + events.tsv  (clean BIDS output)")
# 2) EEG_Processor spectral derivative -> SnPM (works today), via the lane-2 gap
arrow(ax, 28, A - 11.0, 19, 26.2, color=HANDOFF, lw=2.4,
      label="*_powerspect.mat\n(spectral power)")
# 3) turtlewave event tables -> SnPM (importer still to build)
arrow(ax, 82, B - 11.3, 20, 16.4, color=PROP_EC, lw=2.6, ls="--",
      label="event CSV / neural_events.db  (importer TBD)")

fig2.tight_layout()
p2_png = os.path.join(OUT_DIR, "workflow_ecosystem.png")
p2_pdf = os.path.join(OUT_DIR, "workflow_ecosystem.pdf")
fig2.savefig(p2_png, dpi=200, bbox_inches="tight")
fig2.savefig(p2_pdf, bbox_inches="tight")

for p in (p1_png, p1_pdf, p2_png, p2_pdf):
    print(f"Saved: {p}")
