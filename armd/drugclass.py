"""Mapping antibiotic agents and exposure medications into one drug-class space.

The tested-agent column of the cohort table carries no class label, so a single mapping
has to serve both the susceptibility targets and the causal treatment; otherwise the
antibiotic that is predicted and the exposure that is studied would live in different
spaces. The exposure table *does* carry a native ``antibiotic_class`` column with a
coarser taxonomy, and :func:`audit_against_native` cross-checks the two so that
disagreements are reported rather than discovered later.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

CLASS_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Fluoroquinolone", ("floxacin", "quinolone")),
    ("Carbapenem", ("meropenem", "ertapenem", "imipenem", "doripenem", "penem")),
    ("Cephalosporin", ("cef", "ceph")),
    (
        "Penicillin",
        (
            "penicillin", "ampicillin", "amoxicillin", "piperacillin", "oxacillin",
            "nafcillin", "ticarcillin", "ticacarcillin", "dicloxacillin", "methicillin",
        ),
    ),
    ("Monobactam", ("aztreonam",)),
    ("Aminoglycoside", ("gentamicin", "tobramycin", "amikacin", "streptomycin", "neomycin", "plazomicin")),
    ("Glycopeptide", ("vancomycin", "teicoplanin", "dalbavancin", "oritavancin", "telavancin")),
    ("Macrolide", ("erythromycin", "azithromycin", "clarithromycin")),
    ("Lincosamide", ("clindamycin", "lincomycin")),
    ("Glycylcycline", ("tigecycline", "eravacycline", "omadacycline")),
    ("Tetracycline", ("tetracycline", "doxycycline", "minocycline")),
    ("Oxazolidinone", ("linezolid", "tedizolid")),
    ("Lipopeptide", ("daptomycin",)),
    ("Streptogramin", ("quinupristin", "dalfopristin")),
    ("Folate Pathway Inhibitor", ("trimethoprim", "sulfamethoxazole", "sulfa")),
    ("Nitrofuran", ("nitrofurantoin", "nitrofuran")),
    ("Polymyxin", ("colistin", "polymyxin")),
    ("Ansamycin", ("rifampin", "rifaximin", "rifabutin", "rifapentine")),
    ("Nitroimidazole", ("metronidazole", "tinidazole")),
    ("Phosphonic", ("fosfomycin",)),
    ("Phenicol", ("chloramphenicol",)),
)


def agent_to_class(name: object) -> Optional[str]:
    """Map an agent or medication name to a canonical drug class, or None."""
    if not isinstance(name, str):
        return None
    s = name.lower()
    for cls, keys in CLASS_RULES:
        if any(k in s for k in keys):
            return cls
    return None


def map_series(names: Iterable[object]) -> pd.Series:
    """Vectorised-ish wrapper that caches the mapping over the distinct names."""
    s = pd.Series(list(names), dtype="object")
    lookup = {n: agent_to_class(n) for n in s.dropna().unique()}
    return s.map(lookup)


def unmapped_agents(names: Iterable[object]) -> list[str]:
    """Agent names that no rule matches. A non-empty result is a coverage gap."""
    out = {n for n in names if isinstance(n, str) and agent_to_class(n) is None}
    return sorted(out)


def audit_against_native(
    exposure: pd.DataFrame,
    name_col: str = "medication_name",
    native_col: str = "antibiotic_class",
) -> pd.DataFrame:
    """Compare the hand-written mapping with the dataset's own ``antibiotic_class``.

    Returns one row per (medication, native class, derived class) with a count, so that
    systematic disagreements are visible. The two taxonomies are cut differently -- the
    native one merges beta-lactams into "Beta Lactam" and macrolides with lincosamides --
    so disagreement is expected; silence about it is not.
    """
    if native_col not in exposure.columns or name_col not in exposure.columns:
        return pd.DataFrame(columns=[name_col, native_col, "derived_class", "n"])
    tmp = exposure[[name_col, native_col]].copy()
    tmp["derived_class"] = map_series(tmp[name_col]).values
    out = (
        tmp.groupby([name_col, native_col, "derived_class"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    return out.reset_index(drop=True)
