"""Emit the analysis overview as an editable draw.io file.

The decision rule takes its failure probabilities from the per-agent gradient boosting
models; the exposure block feeds prediction only; the acute severity block enters the
causal model only as a sensitivity analysis. Counts are read from
results/full/run_report.json.

Run from the repository root, then export with draw.io:
    python -m scripts.methodology_drawio
    draw.io -x -f png -s 2.5 -o outputs/figures/methodology.png outputs/figures/methodology.drawio
"""
import json
from xml.sax.saxutils import escape
import os

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

R = json.load(open("results/full/run_report.json"))
C, D = R["cohort"], R["decision"]
dropped = [c.lower() for c in D["classes_dropped_for_positivity"]]
kept = D["classes_with_effects"]
assert D["probabilities_from"] == "per-agent baseline"
assert len(kept) == 6 and len(dropped) == 5
assert R["feature_dims"]["exposure"] == 17
assert R["feature_dims"]["causal_primary"] == R["feature_dims"]["base"] + R["feature_dims"]["severity_pre_exposure"]


def n(x):
    return f"{x:,}"


BOX = ("rounded=1;arcSize=6;whiteSpace=wrap;html=1;fontSize=12;spacing=6;"
       "fillColor={f};strokeColor={s};fontColor=#12222e;")
DATA = BOX.format(f="#e8eef4", s="#33566e")
COV = BOX.format(f="#fbf3e6", s="#b7802f")
PRED = BOX.format(f="#ebf5ef", s="#2e7d57")
CAUS = BOX.format(f="#e8eef4", s="#1f6fb2")
NEG = BOX.format(f="#fbecea", s="#b5543f")
DEC = BOX.format(f="#1f6fb2", s="#16527f").replace("fontColor=#12222e", "fontColor=#ffffff")
GREYD = BOX.format(f="#f2f2f2", s="#8a8a8a") + "dashed=1;"
GROUP = ("rounded=0;whiteSpace=wrap;html=1;fillColor=#fafafa;strokeColor=#9a9a9a;"
         "verticalAlign=top;align=left;spacingLeft=8;fontSize=13;fontStyle=1;fontColor=#333333;")
GROUP_B = GROUP.replace("verticalAlign=top", "verticalAlign=bottom") + "spacingBottom=4;"
DIAMOND = ("rhombus;whiteSpace=wrap;html=1;fontSize=12;fillColor=#e8eef4;strokeColor=#1f6fb2;"
           "fontColor=#12222e;")
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#555555;strokeWidth=1.3;"
        "endArrow=block;endFill=1;endSize=6;fontSize=11;labelBackgroundColor=#ffffff;")

cells, k = [], [0]


def lab(head, *rest):
    return "<br>".join(["<b>" + escape(head) + "</b>"] + [escape(r) for r in rest])


def attr(s):
    return escape(s, {'"': "&quot;"})


def node(id_, label, style, x, y, w, h):
    cells.append(f'<mxCell id="{id_}" value="{attr(label)}" style="{style}" vertex="1" '
                 f'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'as="geometry"/></mxCell>')


def edge(src, tgt, label="", dashed=False, ex=None, ey=None, nx=None, ny=None, pts=()):
    k[0] += 1
    st = EDGE + ("dashed=1;" if dashed else "")
    if ex is not None:
        st += f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;"
    if nx is not None:
        st += f"entryX={nx};entryY={ny};entryDx=0;entryDy=0;"
    way = ""
    if pts:
        way = '<Array as="points">' + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts) + "</Array>"
    cells.append(f'<mxCell id="e{k[0]}" value="{attr(escape(label))}" style="{st}" edge="1" '
                 f'parent="1" source="{src}" target="{tgt}"><mxGeometry relative="1" '
                 f'as="geometry">{way}</mxGeometry></mxCell>')


node("D", lab("ARMD, Stanford Health Care, 2008–2023",
              f"{n(C['patients_available'])} patients; {n(C['results_definitive'])} definitive susceptibility results",
              f"{n(C['culture_organism_units'])} culture–organism units, {n(C['culture_orders'])} orders, "
              f"{n(C['patients_with_units'])} patients"),
     DATA, 300, 20, 520, 70)

node("G1", "Covariates, all measured before the culture unless stated", GROUP, 20, 120, 1080, 125)
node("BASE", lab("Base", "organism, specimen, care setting, ordering mode, age, sex, prior "
                 "procedures, organisms and resistant isolates"), COV, 35, 150, 250, 80)
node("EXPO", lab("Exposure", "receipt of each of 17 drug classes in the 90 days before the "
                 "culture"), COV, 300, 150, 250, 80)
node("SPRE", lab("Pre-exposure severity", "deprivation index, comorbidity count"),
     COV, 565, 150, 250, 80)
node("SACU", lab("Acute severity", "14-day laboratory and vital signs, measured after "
                 "exposure"), GREYD, 830, 150, 250, 80)

node("G2", "Susceptibility prediction (train before 2021, test 2021 onward)", GROUP_B,
     20, 280, 500, 385)
node("XGB", lab("Per-agent gradient boosting", "one XGBoost model per agent"),
     PRED, 40, 320, 200, 70)
node("ENC", lab("Shared neural encoder", "summary z of the base block"),
     PRED, 300, 320, 200, 60)
node("SH", lab("Susceptibility head", "reads z and exposure"), PRED, 300, 420, 200, 60)
node("NEG", lab("Reported negative results",
                "the shared network did not beat the per-agent models; z as the causal "
                "adjustment set widened the interval"), NEG, 270, 520, 230, 100)

node("G3", "Collateral resistance, estimated per drug class", GROUP_B, 560, 280, 540, 485)
node("DEDUP", lab("One row per culture order",
                  "outcome: resistance to the class at the next positive culture within "
                  "one year"), CAUS, 580, 315, 260, 75)
node("AIPW", lab("Cross-fitted AIPW",
                 "gradient boosting nuisance models, patient-grouped folds, intervals "
                 "clustered on patients"), CAUS, 580, 415, 260, 75)
node("GUARD", "Overlap check:<br>clipped orders<br>under 25%?", DIAMOND, 620, 515, 180, 110)
node("KEEP", lab(f"{len(kept)} classes estimable",
                 ", ".join(c.lower() for c in kept)), CAUS, 590, 655, 240, 75)
node("DROP", lab(f"{len(dropped)} classes not interpretable", ", ".join(dropped)),
     NEG, 860, 530, 225, 80)
node("DIAG", lab("Fluoroquinolone checks",
                 "balance, E-value, nitrofuran negative control, same organism, 30-day "
                 "window, censoring, acute block"), CAUS, 860, 400, 225, 105)

node("DEC", lab("Decision rule, test period",
                "score = λ × P(failure now) + (1 − λ) × collateral cost of the agent’s class",
                f"{D['eligible_agents']} eligible agents"), DEC, 120, 790, 420, 85)
node("FRONT", lab("Trade-off frontier", "λ swept from 0 to 1"), DEC, 640, 800, 200, 65)

edge("D", "G1", ex=0.5, ey=1, nx=0.5, ny=0)
edge("BASE", "XGB", ex=0.3, ey=1, nx=0.5, ny=0)
edge("EXPO", "XGB", ex=0.2, ey=1, nx=0.85, ny=0)
edge("BASE", "ENC", ex=0.8, ey=1, nx=0.3, ny=0, pts=((235, 264), (360, 264)))
edge("EXPO", "SH", ex=0.6, ey=1, nx=1, ny=0.5, pts=((450, 272), (540, 272), (540, 450)))
edge("ENC", "SH", ex=0.5, ey=1, nx=0.5, ny=0)
edge("BASE", "DEDUP", ex=0.95, ey=1, nx=0.25, ny=0, pts=((272, 256), (645, 256)))
edge("SPRE", "DEDUP", ex=0.5, ey=1, nx=0.6, ny=0)
edge("SACU", "DIAG", "sensitivity only", dashed=True, ex=0.5, ey=1, nx=0.5, ny=0)
edge("DEDUP", "AIPW", ex=0.5, ey=1, nx=0.5, ny=0)
edge("AIPW", "GUARD", ex=0.5, ey=1, nx=0.5, ny=0)
edge("AIPW", "DIAG", ex=1, ey=0.5, nx=0, ny=0.5)
edge("GUARD", "KEEP", "yes", ex=0.5, ey=1, nx=0.5, ny=0)
edge("GUARD", "DROP", "no", ex=1, ey=0.5, nx=0, ny=0.5)
edge("SH", "NEG", "compared", dashed=True, ex=0.5, ey=1, nx=0.5, ny=0)
edge("XGB", "DEC", "P(failure)", ex=0.5, ey=1, nx=0.25, ny=0)
edge("KEEP", "DEC", "cost per class", ex=0, ey=0.5, nx=0.85, ny=0)
edge("DEC", "FRONT", ex=1, ey=0.5, nx=0, ny=0.5)

xml = ('<mxfile host="drawio"><diagram id="methodology" name="Analysis overview">'
       '<mxGraphModel dx="1200" dy="900" grid="0" gridSize="10" guides="1" tooltips="1" '
       'connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="1120" '
       'pageHeight="900" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
open("outputs/figures/methodology.drawio", "w").write(xml)
print("written outputs/figures/methodology.drawio")
