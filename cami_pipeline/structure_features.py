import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from casmi_pipeline.images import fetch_pubchem_cid
from casmi_pipeline.io_utils import safe_float, safe_int, sanitize_short_text, write_json_atomic
from casmi_pipeline.settings import PRIMARY_NEGATIVE_ADDUCTS, PUBCHEM_PROPERTY_URL


FORMULA_TOKEN_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")

ACID_PATTERNS = (
    "C(=O)O",
    "C(O)=O",
    "S(=O)(=O)O",
    "S(O)(=O)=O",
    "OS(=O)(=O)",
    "OP(=O)(O)",
    "P(=O)(O)O",
    "P(O)(O)=O",
    "[O-]",
    "[S-]",
)

BASIC_PATTERNS = (
    "[nH]",
    "N(",
    "N=",
    "CN",
    "NC",
    "N1",
    "N2",
    "N3",
    "N4",
)


def fetch_pubchem_properties(
    session: requests.Session,
    cid: int,
    timeout: int,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cid}.json"
        if cache_path.exists() and not force:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)

    try:
        response = session.get(PUBCHEM_PROPERTY_URL.format(cid=cid), timeout=timeout)
        response.raise_for_status()
        record = response.json().get("PropertyTable", {}).get("Properties", [{}])[0]
    except Exception:
        record = {}

    if cache_path is not None and record:
        write_json_atomic(cache_path, record)
    return record


def parse_formula_counts(formula: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not formula or formula == "unknown":
        return counts
    for element, count in FORMULA_TOKEN_PATTERN.findall(formula):
        counts[element] = counts.get(element, 0) + int(count or "1")
    return counts


def build_heteroatom_signature(formula_counts: Dict[str, int]) -> str:
    if not formula_counts:
        return "unknown"
    halogens = sum(formula_counts.get(elem, 0) for elem in ("F", "Cl", "Br", "I"))
    return (
        f"N{formula_counts.get('N', 0)} "
        f"O{formula_counts.get('O', 0)} "
        f"S{formula_counts.get('S', 0)} "
        f"P{formula_counts.get('P', 0)} "
        f"hal{halogens}"
    )


def classify_xlogp(xlogp: Optional[float]) -> str:
    if xlogp is None:
        return "unknown"
    if xlogp < 1.0:
        return "low"
    if xlogp < 3.0:
        return "medium"
    return "high"


def guess_adduct_compatibility(smiles: str) -> list[str]:
    if not smiles:
        return ["unknown"]
    acidic = any(pattern in smiles for pattern in ACID_PATTERNS)
    basic = any(pattern in smiles for pattern in BASIC_PATTERNS)
    if not acidic and not basic:
        hetero_rich = sum(smiles.count(token) for token in ("N", "O", "S", "P", "F", "Cl", "Br", "I"))
        if hetero_rich >= 2:
            return ["[M+H]+", "[M+Na]+", "[M-H]-"]
        return ["unknown"]

    forms = []
    if acidic:
        forms.extend(PRIMARY_NEGATIVE_ADDUCTS)
    if basic:
        forms.extend(["[M+H]+", "[M+NH4]+", "[M+Na]+"])

    deduped = []
    seen = set()
    for form in forms:
        if form not in seen:
            deduped.append(form)
            seen.add(form)
    return deduped or ["unknown"]


def build_rdkit_structure_features(smiles: str) -> Dict[str, Any]:
    """Build deterministic structure features locally, without PubChem I/O."""
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception as exc:
        raise RuntimeError("RDKit is required for MassSpecGym candidate processing.") from exc

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "feature_source": "rdkit_parse_failed",
            "pubchem_cid": None,
            "molecular_formula": "unknown",
            "exact_mass": None,
            "hbd": None,
            "hba": None,
            "xlogp": None,
            "xlogp_class": "unknown",
            "heteroatom_signature": "unknown",
            "formula_counts": {},
            "adduct_compatibility": guess_adduct_compatibility(smiles),
        }

    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)
    formula_counts = parse_formula_counts(molecular_formula)
    xlogp = float(Crippen.MolLogP(mol))
    return {
        "feature_source": "rdkit",
        "pubchem_cid": None,
        "molecular_formula": molecular_formula,
        "exact_mass": float(Descriptors.ExactMolWt(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "xlogp": xlogp,
        "xlogp_class": classify_xlogp(xlogp),
        "heteroatom_signature": build_heteroatom_signature(formula_counts),
        "formula_counts": formula_counts,
        "adduct_compatibility": guess_adduct_compatibility(smiles),
    }


def build_structure_features(
    session: requests.Session,
    smiles: str,
    image_meta: Dict[str, Any],
    timeout: int,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    cid = image_meta.get("pubchem_cid")
    if cid is None and smiles:
        cid = fetch_pubchem_cid(session, smiles, timeout)

    properties: Dict[str, Any] = {}
    if cid is not None:
        properties = fetch_pubchem_properties(
            session=session,
            cid=int(cid),
            timeout=timeout,
            cache_dir=cache_dir,
            force=force,
        )

    molecular_formula = sanitize_short_text(properties.get("MolecularFormula"))
    formula_counts = parse_formula_counts(molecular_formula)
    exact_mass = safe_float(properties.get("ExactMass"))
    hbd = safe_int(properties.get("HBondDonorCount"))
    hba = safe_int(properties.get("HBondAcceptorCount"))
    xlogp = safe_float(properties.get("XLogP"))

    return {
        "pubchem_cid": int(cid) if cid is not None else None,
        "molecular_formula": molecular_formula,
        "exact_mass": exact_mass,
        "hbd": hbd,
        "hba": hba,
        "xlogp": xlogp,
        "xlogp_class": classify_xlogp(xlogp),
        "heteroatom_signature": build_heteroatom_signature(formula_counts),
        "formula_counts": formula_counts,
        "adduct_compatibility": guess_adduct_compatibility(smiles),
    }
