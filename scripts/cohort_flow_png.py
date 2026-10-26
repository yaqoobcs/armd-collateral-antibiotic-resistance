"""Cohort-derivation flow diagram for the JAC-AMR manuscript.

Every count is read from results/full/run_report.json; nothing is
hand-entered. Boxes are auto-sized to their contents. Run from the repository root.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

R = json.load(open("results/full/run_report.json"))
C, CA = R["cohort"], R["causal"]

n_pat        = C["patients_available"]
n_results    = C["results_total"]
n_definitive = C["results_definitive"]
n_indet      = C["results_dropped_indeterminate"]
n_units      = C["culture_organism_units"]
n_orders     = C["culture_orders"]
n_pat_units  = C["patients_with_units"]
n_fq_units   = C["causal_valid_units"]
n_fq_orders  = C["causal_valid_orders"]
n_fq_pat     = CA["patients"]
n_exposed    = round(CA["exposure_rate"] * n_fq_orders)
n_unexposed  = n_fq_orders - n_exposed
n_no_follow  = n_units - n_fq_units

def g(n):
    "Thin-space thousands separator, matching siunitx output in the body text."
    return f"{n:,}".replace(",", " ")

MAIN, EDGE = "#e8eef4", "#33566e"
EXCL, EXCL_EDGE = "#f6f1e8", "#9a8358"
plt.rcParams["font.family"] = ["DejaVu Sans"]

HEAD_PAD, LINE_H, BOT_PAD, FS = 6.2, 5.4, 4.0, 9.0

def box_height(lines):
    return HEAD_PAD + LINE_H * (len(lines) - 1) + BOT_PAD + 2.0

def draw_box(ax, x, y_top, w, lines, fill=MAIN, edge=EDGE, fs=FS):
    """Draw a box whose TOP edge is y_top. Returns the bottom y."""
    h = box_height(lines)
    y = y_top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                linewidth=1.0, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y_top - HEAD_PAD + 0.6, lines[0], ha="center", va="center",
            fontsize=fs, fontweight="bold")
    for i, t in enumerate(lines[1:]):
        ax.text(x + w / 2, y_top - HEAD_PAD - 0.4 - (i + 1) * LINE_H + 1.4, t,
                ha="center", va="center", fontsize=fs)
    return y

def arrow(ax, p0, p1, color=EDGE):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.0, color=color, shrinkA=0, shrinkB=0))

fig, ax = plt.subplots(figsize=(7.2, 9.0))
ax.axis("off")

XL, W, CX = 2.0, 58.0, 31.0
XR, WR = 65.0, 36.0
GAP = 7.5

main = [
    ["ARMD cohort", f"{g(n_pat)} patients", "cultures ordered 2008–2023"],
    ["Positive cultures with an identified", "organism and a tested agent",
     f"{g(n_results)} susceptibility results"],
    ["Definitive results", f"{g(n_definitive)} susceptible or resistant"],
    ["Analytic units", f"{g(n_units)} culture–organism pairs",
     f"{g(n_orders)} culture orders", f"{g(n_pat_units)} patients"],
    ["Fluoroquinolone analysis set", f"{g(n_fq_units)} culture–organism units"],
    ["De-duplicated to the culture order",
     f"{g(n_fq_orders)} index orders", f"{g(n_fq_pat)} patients"],
]
excl = {
    1: ["Excluded", f"{g(n_indet)} intermediate or", "inconclusive results"],
    3: ["Excluded", f"{g(n_no_follow)} units with no next",
        "positive culture carrying a", "definitive result within 1 year"],
}

y_top, bottoms = 128.0, []
for i, lines in enumerate(main):
    y_bot = draw_box(ax, XL, y_top, W, lines)
    bottoms.append(y_bot)
    if i < len(main) - 1:
        nxt = y_bot - GAP
        arrow(ax, (CX, y_bot), (CX, nxt))
        if i in excl:
            y_mid = y_bot - GAP / 2
            e_lines = excl[i]
            e_top = y_mid + box_height(e_lines) / 2
            draw_box(ax, XR, e_top, WR, e_lines, fill=EXCL, edge=EXCL_EDGE, fs=8.4)
            arrow(ax, (CX, y_mid), (XR - 2.0, y_mid), color=EXCL_EDGE)
        y_top = nxt

y_last = bottoms[-1]
y_rail = y_last - 5.0
ax.add_patch(FancyArrowPatch((CX, y_last), (CX, y_rail), arrowstyle="-",
                             linewidth=1.0, color=EDGE, shrinkA=0, shrinkB=0))
lx, rx, bw = 3.0, 32.0, 26.0
ax.add_patch(FancyArrowPatch((lx + bw / 2, y_rail), (rx + bw / 2, y_rail), arrowstyle="-",
                             linewidth=1.0, color=EDGE, shrinkA=0, shrinkB=0))
for x, lines in ((lx, ["Exposed", f"{g(n_exposed)} orders"]),
                 (rx, ["Unexposed", f"{g(n_unexposed)} orders"])):
    arrow(ax, (x + bw / 2, y_rail), (x + bw / 2, y_rail - 5.0))
    draw_box(ax, x, y_rail - 5.0, bw, lines, fs=8.6)

ax.set_xlim(0, 101)
ax.set_ylim(y_rail - 5.0 - box_height(["a", "b"]) - 2, 130)
fig.savefig("outputs/figures/cohort_flow.png", dpi=300, bbox_inches="tight",
            facecolor="white", pad_inches=0.12)
print("written outputs/figures/cohort_flow.png")
print(f"checks: {n_units}-{n_fq_units}={n_no_follow} | {n_exposed}+{n_unexposed}={n_exposed+n_unexposed}"
      f" | {n_results}-{n_indet}={n_results-n_indet} (report says {n_definitive})")
