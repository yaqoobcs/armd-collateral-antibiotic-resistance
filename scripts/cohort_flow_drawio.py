"""Emit the cohort-derivation flow chart as an editable draw.io (mxGraph) file.

Counts come from results/full/run_report.json, the same source the PNG
generator uses, so the two cannot drift apart. Boxes are native rounded rectangles
and the arrows are attached edges (source/target), not fixed waypoints, so dragging
a box in draw.io re-routes its connectors instead of stranding them.

Run from the repository root:
    python -m scripts.cohort_flow_drawio
"""
import json
from xml.sax.saxutils import escape
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

assert n_results - n_indet == n_definitive
assert n_exposed + n_unexposed == n_fq_orders
assert n_units - n_no_follow == n_fq_units

def g(n):
    "Thin-space thousands separator, matching siunitx output in the manuscript."
    return f"{n:,}".replace(",", " ")

MAIN = ("rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#e8eef4;strokeColor=#33566e;"
        "fontSize=13;fontColor=#12222e;verticalAlign=middle;spacing=4;")
EXCL = ("rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#f6f1e8;strokeColor=#9a8358;"
        "fontSize=12;fontColor=#3c3020;verticalAlign=middle;spacing=4;")
E_MAIN = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#33566e;strokeWidth=1.4;"
          "endArrow=block;endFill=1;endSize=6;exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
E_EXCL = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#9a8358;strokeWidth=1.2;"
          "endArrow=block;endFill=1;endSize=6;exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

def val(head, *rest):
    """Build the cell's HTML label.

    draw.io stores HTML inside an XML attribute, so the label is escaped twice: once
    here for the HTML layer, and again by attr() for the XML layer. Escaping only once
    produces a file draw.io refuses to open.
    """
    parts = ["<b>" + escape(head) + "</b>"] + [escape(r) for r in rest]
    return "<br>".join(parts)

def attr(html):
    "Escape an assembled HTML label for use as an XML attribute value."
    return escape(html, {'"': "&quot;"})

X, W = 40, 320
XE, WE = 420, 260

nodes = [
    ("n1",  40,  70, val("ARMD cohort", f"{g(n_pat)} patients",
                         "cultures ordered 2008–2023")),
    ("n2", 160,  70, val("Positive cultures with an identified organism and a tested agent",
                         f"{g(n_results)} susceptibility results")),
    ("n3", 280,  56, val("Definitive results",
                         f"{g(n_definitive)} susceptible or resistant")),
    ("n4", 386,  88, val("Analytic units", f"{g(n_units)} culture–organism pairs",
                         f"{g(n_orders)} culture orders", f"{g(n_pat_units)} patients")),
    ("n5", 524,  56, val("Fluoroquinolone analysis set",
                         f"{g(n_fq_units)} culture–organism units")),
    ("n6", 630,  70, val("De-duplicated to the culture order",
                         f"{g(n_fq_orders)} index orders", f"{g(n_fq_pat)} patients")),
]
excl_nodes = [
    ("x1", 160, 70, val("Excluded", f"{g(n_indet)} intermediate or",
                        "inconclusive results")),
    ("x2", 386, 88, val("Excluded", f"{g(n_no_follow)} units with no next positive",
                        "culture carrying a definitive", "result within 1 year")),
]
split = [
    ("s1",  40, 760, 145, 56, val("Exposed", f"{g(n_exposed)} orders")),
    ("s2", 215, 760, 145, 56, val("Unexposed", f"{g(n_unexposed)} orders")),
]
edges = [
    ("e1", "n1", "n2", E_MAIN), ("e2", "n2", "n3", E_MAIN),
    ("e3", "n3", "n4", E_MAIN), ("e4", "n4", "n5", E_MAIN),
    ("e5", "n5", "n6", E_MAIN),
    ("e6", "n2", "x1", E_EXCL), ("e7", "n4", "x2", E_EXCL),
    ("e8", "n6", "s1", E_MAIN), ("e9", "n6", "s2", E_MAIN),
]

out = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<mxfile host="app.diagrams.net" version="26.2.2">',
    '  <diagram name="Cohort derivation" id="armd-cohort-flow">',
    '    <mxGraphModel dx="1100" dy="900" grid="1" gridSize="10" guides="1" tooltips="1"'
    ' connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850"'
    ' pageHeight="1100" math="0" shadow="0">',
    '      <root>',
    '        <mxCell id="0" />',
    '        <mxCell id="1" parent="0" />',
]
for i, y, h, v in nodes:
    out.append(f'        <mxCell id="{i}" value="{attr(v)}" style="{MAIN}" parent="1" vertex="1">'
               f'<mxGeometry x="{X}" y="{y}" width="{W}" height="{h}" as="geometry" /></mxCell>')
for i, y, h, v in excl_nodes:
    out.append(f'        <mxCell id="{i}" value="{attr(v)}" style="{EXCL}" parent="1" vertex="1">'
               f'<mxGeometry x="{XE}" y="{y}" width="{WE}" height="{h}" as="geometry" /></mxCell>')
for i, x, y, w, h, v in split:
    out.append(f'        <mxCell id="{i}" value="{attr(v)}" style="{MAIN}" parent="1" vertex="1">'
               f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')
for i, s, t, st in edges:
    out.append(f'        <mxCell id="{i}" style="{st}" parent="1" source="{s}" target="{t}"'
               f' edge="1"><mxGeometry relative="1" as="geometry" /></mxCell>')
out += ['      </root>', '    </mxGraphModel>', '  </diagram>', '</mxfile>', '']

open("outputs/figures/cohort_flow.drawio", "w").write("\n".join(out))
print("written outputs/figures/cohort_flow.drawio")
print(f"checks ok: {n_results}-{n_indet}={n_definitive} | "
      f"{n_units}-{n_no_follow}={n_fq_units} | {n_exposed}+{n_unexposed}={n_fq_orders}")
