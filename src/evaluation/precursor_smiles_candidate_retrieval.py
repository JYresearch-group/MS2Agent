#!/usr/bin/env python3
"""
Precursor m/z -> high-confidence SMILES candidate retrieval.

This script intentionally does not parse MGF. It expects structured JSON records
that already contain fields such as pepmass/precursor_mz, charge, ionmode, and
optionally smiles/inchi/name.

Retrieval strategy:
1. Infer the precursor adduct and neutral monoisotopic mass from PEPMASS.
2. Retrieve neutral molecule candidates from HMDB, PubChem, and ChemSpider.
3. Apply rigid filters: ppm mass error, precursor charge/adduct consistency,
   and stable/metastable structure status.
4. Deduplicate by canonical structure key.
5. Rank by mass match, stability, and database authority.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import requests
except Exception:  # pragma: no cover - handled at runtime
    requests = None  # type: ignore[assignment]

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except Exception:  # pragma: no cover - RDKit is optional
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]


PROTON_MASS = 1.007276466812
ELECTRON_MASS = 0.000548579909
WATER_MASS = 18.010564684
ACN_MASS = 41.026549101
CALCIUM_2PLUS_DELTA = 39.961493

MONOISOTOPIC_MASSES = {
    "H": 1.00782503223,
    "B": 11.00930536,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Na": 22.9897692820,
    "Mg": 23.985041697,
    "Al": 26.98153853,
    "Si": 27.97692653465,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "K": 38.9637064864,
    "Ca": 39.962590863,
    "Fe": 55.93493633,
    "Cu": 62.92959772,
    "Zn": 63.92914201,
    "Br": 78.9183376,
    "I": 126.9044719,
    "Se": 79.9165218,
}

SOURCE_PRIORITY = {
    "HMDB": 0,
    "PubChem": 1,
    "ChemSpider": 2,
    "InputSMILES": 3,
}

STABILITY_PRIORITY = {
    "stable": 0,
    "metastable": 1,
    "unknown": 2,
    "unstable": 9,
}


@dataclass(frozen=True)
class AdductSpec:
    name: str
    charge: int
    mass_delta: float
    multiplier: int = 1
    polarity: str = "positive"
    aliases: Tuple[str, ...] = ()

    def neutral_mass_from_mz(self, observed_mz: float) -> float:
        return (observed_mz * abs(self.charge) - self.mass_delta) / self.multiplier

    def mz_from_neutral_mass(self, neutral_mass: float) -> float:
        return (self.multiplier * neutral_mass + self.mass_delta) / abs(self.charge)


ADDUCT_SPECS: Dict[str, AdductSpec] = {
    "[M+H]+": AdductSpec("[M+H]+", 1, PROTON_MASS, polarity="positive", aliases=("M+H",)),
    "[M+Na]+": AdductSpec("[M+Na]+", 1, 22.989218, polarity="positive", aliases=("M+NA",)),
    "[M+NH4]+": AdductSpec("[M+NH4]+", 1, 18.033823, polarity="positive", aliases=("M+NH4",)),
    "[M+K]+": AdductSpec("[M+K]+", 1, 38.963158, polarity="positive", aliases=("M+K",)),
    "[M]+": AdductSpec("[M]+", 1, 0.0, polarity="positive", aliases=("M", "M+")),
    "[M+H-H2O]+": AdductSpec(
        "[M+H-H2O]+",
        1,
        PROTON_MASS - WATER_MASS,
        polarity="positive",
        aliases=("[M-H2O+H]+", "M+H-H2O", "M-H2O+H"),
    ),
    "[M+H-2H2O]+": AdductSpec(
        "[M+H-2H2O]+",
        1,
        PROTON_MASS - 2 * WATER_MASS,
        polarity="positive",
        aliases=("[M-2H2O+H]+", "M+H-2H2O", "M-2H2O+H"),
    ),
    "[M+H-3H2O]+": AdductSpec(
        "[M+H-3H2O]+",
        1,
        PROTON_MASS - 3 * WATER_MASS,
        polarity="positive",
        aliases=("[M-3H2O+H]+", "M+H-3H2O", "M-3H2O+H"),
    ),
    "[M+H-4H2O]+": AdductSpec(
        "[M+H-4H2O]+",
        1,
        PROTON_MASS - 4 * WATER_MASS,
        polarity="positive",
        aliases=("[M-4H2O+H]+", "M+H-4H2O", "M-4H2O+H"),
    ),
    "[M+H-5H2O]+": AdductSpec(
        "[M+H-5H2O]+",
        1,
        PROTON_MASS - 5 * WATER_MASS,
        polarity="positive",
        aliases=("[M-5H2O+H]+", "M+H-5H2O", "M-5H2O+H"),
    ),
    "[M+2H]2+": AdductSpec("[M+2H]2+", 2, 2 * PROTON_MASS, polarity="positive", aliases=("M+2H",)),
    "[M+3H]3+": AdductSpec("[M+3H]3+", 3, 3 * PROTON_MASS, polarity="positive", aliases=("M+3H",)),
    "[M+Ca]2+": AdductSpec("[M+Ca]2+", 2, CALCIUM_2PLUS_DELTA, polarity="positive", aliases=("M+CA",)),
    "[M+Ca-H]+": AdductSpec(
        "[M+Ca-H]+",
        1,
        CALCIUM_2PLUS_DELTA - PROTON_MASS,
        polarity="positive",
        aliases=("M+CA-H",),
    ),
    "[M+2H-3H2O]2+": AdductSpec(
        "[M+2H-3H2O]2+",
        2,
        2 * PROTON_MASS - 3 * WATER_MASS,
        polarity="positive",
        aliases=("[M-3H2O+2H]2+", "M-3H2O+2H", "M+2H-3H2O"),
    ),
    "[2M+H]+": AdductSpec("[2M+H]+", 1, PROTON_MASS, multiplier=2, polarity="positive", aliases=("2M+H",)),
    "[2M+Na]+": AdductSpec("[2M+Na]+", 1, 22.989218, multiplier=2, polarity="positive", aliases=("2M+NA",)),
    "[2M+NH4]+": AdductSpec("[2M+NH4]+", 1, 18.033823, multiplier=2, polarity="positive", aliases=("2M+NH4",)),
    "[2M+K]+": AdductSpec("[2M+K]+", 1, 38.963158, multiplier=2, polarity="positive", aliases=("2M+K",)),
    "[2M+Ca]2+": AdductSpec(
        "[2M+Ca]2+",
        2,
        CALCIUM_2PLUS_DELTA,
        multiplier=2,
        polarity="positive",
        aliases=("2M+CA",),
    ),
    "[2M+Ca-H]+": AdductSpec(
        "[2M+Ca-H]+",
        1,
        CALCIUM_2PLUS_DELTA - PROTON_MASS,
        multiplier=2,
        polarity="positive",
        aliases=("2M+CA-H",),
    ),
    "[2M+H-H2O]+": AdductSpec(
        "[2M+H-H2O]+",
        1,
        PROTON_MASS - WATER_MASS,
        multiplier=2,
        polarity="positive",
        aliases=("[2M-H2O+H]+", "2M+H-H2O", "2M-H2O+H"),
    ),
    "[2M+H-2H2O]+": AdductSpec(
        "[2M+H-2H2O]+",
        1,
        PROTON_MASS - 2 * WATER_MASS,
        multiplier=2,
        polarity="positive",
        aliases=("[2M-2H2O+H]+", "2M+H-2H2O", "2M-2H2O+H"),
    ),
    "[2M-H+2Na]+": AdductSpec(
        "[2M-H+2Na]+",
        1,
        2 * 22.989218 - PROTON_MASS,
        multiplier=2,
        polarity="positive",
        aliases=("2M-H+2NA",),
    ),
    "[3M+H]+": AdductSpec("[3M+H]+", 1, PROTON_MASS, multiplier=3, polarity="positive", aliases=("3M+H",)),
    "[3M+Na]+": AdductSpec("[3M+Na]+", 1, 22.989218, multiplier=3, polarity="positive", aliases=("3M+NA",)),
    "[3M+NH4]+": AdductSpec("[3M+NH4]+", 1, 18.033823, multiplier=3, polarity="positive", aliases=("3M+NH4",)),
    "[3M+K]+": AdductSpec("[3M+K]+", 1, 38.963158, multiplier=3, polarity="positive", aliases=("3M+K",)),
    "[3M+Ca]2+": AdductSpec(
        "[3M+Ca]2+",
        2,
        CALCIUM_2PLUS_DELTA,
        multiplier=3,
        polarity="positive",
        aliases=("3M+CA",),
    ),
    "[3M+Ca-H]+": AdductSpec(
        "[3M+Ca-H]+",
        1,
        CALCIUM_2PLUS_DELTA - PROTON_MASS,
        multiplier=3,
        polarity="positive",
        aliases=("3M+CA-H",),
    ),
    "[4M+Ca]2+": AdductSpec(
        "[4M+Ca]2+",
        2,
        CALCIUM_2PLUS_DELTA,
        multiplier=4,
        polarity="positive",
        aliases=("4M+CA",),
    ),
    "[5M+Ca]2+": AdductSpec(
        "[5M+Ca]2+",
        2,
        CALCIUM_2PLUS_DELTA,
        multiplier=5,
        polarity="positive",
        aliases=("5M+CA",),
    ),
    "[M+ACN+H]+": AdductSpec("[M+ACN+H]+", 1, ACN_MASS + PROTON_MASS, polarity="positive", aliases=("M+ACN+H",)),
    "[M+ACN+NH4]+": AdductSpec(
        "[M+ACN+NH4]+",
        1,
        ACN_MASS + 18.033823,
        polarity="positive",
        aliases=("M+ACN+NH4",),
    ),
    "[M-e]+": AdductSpec("[M-e]+", 1, -ELECTRON_MASS, polarity="positive", aliases=("M-E",)),
    "[M-H+2Na]+": AdductSpec(
        "[M-H+2Na]+",
        1,
        2 * 22.989218 - PROTON_MASS,
        polarity="positive",
        aliases=("M-H+2NA",),
    ),
    "[M-H]-": AdductSpec("[M-H]-", -1, -PROTON_MASS, polarity="negative", aliases=("M-H",)),
    "[M+Cl]-": AdductSpec("[M+Cl]-", -1, 34.969402, polarity="negative", aliases=("M+CL",)),
    "[M+FA-H]-": AdductSpec(
        "[M+FA-H]-",
        -1,
        44.998201,
        polarity="negative",
        aliases=("[M+HCOOH-H]-", "M+HCOOH-H", "M+FA-H"),
    ),
    "[M+CH3COO]-": AdductSpec(
        "[M+CH3COO]-",
        -1,
        59.013851,
        polarity="negative",
        aliases=("[M+CH3COOH-H]-", "M+CH3COOH-H", "M+CH3COO"),
    ),
}


def normalize_token(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value.strip().upper())


ADDUCT_ALIASES: Dict[str, str] = {}
for canonical, spec in ADDUCT_SPECS.items():
    ADDUCT_ALIASES[normalize_token(canonical)] = canonical
    stripped = canonical.replace("[", "").replace("]", "")
    ADDUCT_ALIASES[normalize_token(stripped)] = canonical
    for alias in spec.aliases:
        ADDUCT_ALIASES[normalize_token(alias)] = canonical


@dataclass
class QuerySpec:
    record_index: int
    observed_mz: float
    charge: int
    ion_mode: str
    adduct: AdductSpec
    neutral_mass: float
    neutral_mass_window_da: float
    ppm_tolerance: float
    absolute_da_reference: float
    adduct_inference_source: str
    record_id: str
    raw_record: Dict[str, Any]


@dataclass
class Candidate:
    source: str
    source_id: str = ""
    name: str = ""
    smiles: str = ""
    canonical_smiles: str = ""
    inchi: str = ""
    inchikey: str = ""
    formula: str = ""
    exact_mass: Optional[float] = None
    record_charge: Optional[int] = None
    stability: str = "unknown"
    stability_reason: str = ""
    source_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve high-confidence SMILES candidates from precursor m/z JSON records."
    )
    parser.add_argument("--input", required=False, help="Structured JSON input file. Object or list is supported.")
    parser.add_argument("--output", required=False, help="Output JSON path.")
    parser.add_argument("--top-k", type=int, default=50, help="Maximum candidates per query after filtering.")
    parser.add_argument("--ppm-tolerance", type=float, default=5.0, help="Rigid precursor m/z tolerance in ppm.")
    parser.add_argument(
        "--absolute-da-reference",
        type=float,
        default=0.01,
        help="Reference Da window reported as ppm; final filtering still uses --ppm-tolerance.",
    )
    parser.add_argument("--hmdb-file", help="Optional local HMDB export file: JSON/JSONL/CSV/TSV.")
    parser.add_argument("--pubchem-file", help="Optional local PubChem export file: JSON/JSONL/CSV/TSV.")
    parser.add_argument("--chemspider-file", help="Optional local ChemSpider export file: JSON/JSONL/CSV/TSV.")
    parser.add_argument(
        "--no-online-pubchem",
        action="store_true",
        help="Disable PubChem online exact-mass retrieval.",
    )
    parser.add_argument(
        "--pubchem-retmax",
        type=int,
        default=200,
        help="Maximum PubChem CIDs to request before local filtering.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds for online database calls.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between online database calls to be polite to public APIs.",
    )
    parser.add_argument(
        "--include-input-smiles",
        action="store_true",
        help="Also evaluate the input record's own smiles/inchi as a candidate. Useful for sanity checks only.",
    )
    parser.add_argument(
        "--allow-common-adducts",
        action="store_true",
        help="If the record has no explicit adduct, evaluate all common adducts matching ion mode and charge.",
    )
    parser.add_argument(
        "--chemspider-api-key",
        default=os.environ.get("CHEMSPIDER_API_KEY"),
        help="Reserved for ChemSpider API integrations. Local file input is currently recommended.",
    )
    return parser.parse_args(argv)


def lower_key_map(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key).lower(): value for key, value in record.items()}


def first_present(record: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    lowered = lower_key_map(record)
    for key in keys:
        if key.lower() in lowered:
            value = lowered[key.lower()]
            if value not in (None, ""):
                return value
    return default


def parse_float(value: Any, field_name: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    elif isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value.strip())
        if not match:
            raise ValueError(f"Cannot parse float for {field_name!r}: {value!r}")
        number = float(match.group(0))
    else:
        raise ValueError(f"Cannot parse float for {field_name!r}: {value!r}")
    if not math.isfinite(number):
        raise ValueError(f"Non-finite float for {field_name!r}: {value!r}")
    return number


def parse_charge(value: Any, ion_mode: str = "") -> int:
    if value in (None, ""):
        return -1 if ion_mode.lower().startswith("neg") else 1
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = int(value)
    else:
        text = str(value).strip()
        suffix_match = re.search(r"(\d+)\s*([+-])", text)
        signed_match = re.search(r"([+-]\d+|\d+)", text)
        if suffix_match:
            parsed = int(suffix_match.group(1))
            if suffix_match.group(2) == "-":
                parsed *= -1
        elif signed_match:
            parsed = int(signed_match.group(1))
        else:
            parsed = -1 if text.endswith("-") else 1
    if parsed == 0:
        return -1 if ion_mode.lower().startswith("neg") else 1
    if ion_mode.lower().startswith("neg") and parsed > 0:
        return -parsed
    if ion_mode.lower().startswith("pos") and parsed < 0:
        return abs(parsed)
    return parsed


def infer_ion_mode(record: Mapping[str, Any], charge: Optional[int] = None) -> str:
    value = first_present(record, ("ionmode", "ion_mode", "Ion_Mode", "polarity", "mode"), "")
    text = str(value).strip().lower()
    if text.startswith("pos") or text in {"+", "positive"}:
        return "Positive"
    if text.startswith("neg") or text in {"-", "negative"}:
        return "Negative"
    if charge is not None and charge < 0:
        return "Negative"
    return "Positive"


def infer_record_id(record: Mapping[str, Any], record_index: int) -> str:
    value = first_present(
        record,
        ("record_id", "spectrumid", "SpectrumID", "usi", "filename", "name", "id"),
        f"record_{record_index}",
    )
    text = str(value).strip()
    return text or f"record_{record_index}"


def extract_observed_mz(record: Mapping[str, Any]) -> float:
    value = first_present(
        record,
        (
            "pepmass",
            "PEPMASS",
            "precursor",
            "Precursor",
            "precursormass",
            "PRECURSORMASS",
            "precursor_mz",
            "parent_mz",
            "Parent m/z",
        ),
    )
    if value is None:
        raise ValueError("Missing precursor m/z. Expected pepmass/precursor/precursormass/parent_mz.")
    if isinstance(value, list):
        if not value:
            raise ValueError("Precursor m/z list is empty.")
        value = value[0]
    return parse_float(value, "precursor_mz")


def detect_adduct_from_text(text: str, ion_mode: str) -> Optional[AdductSpec]:
    if not text:
        return None
    upper_text = text.upper()

    # Prefer explicit bracketed adducts. Brackets prevent accidental matches
    # across formula/name boundaries, e.g. C52H82O22_ M+H must not become 2M+H.
    for canonical, spec in sorted(ADDUCT_SPECS.items(), key=lambda item: len(item[0]), reverse=True):
        if canonical.upper() in upper_text and ion_mode.lower().startswith(spec.polarity[:3]):
            return spec

    alias_pairs: List[Tuple[str, str]] = []
    for canonical, spec in ADDUCT_SPECS.items():
        stripped = canonical.replace("[", "").replace("]", "")
        alias_pairs.append((stripped, canonical))
        for alias in spec.aliases:
            if alias.startswith("[") and alias.endswith("]"):
                alias_pairs.append((alias[1:-1], canonical))
            else:
                alias_pairs.append((alias, canonical))

    for alias, canonical in sorted(alias_pairs, key=lambda item: len(item[0]), reverse=True):
        if len(alias) <= 1:
            continue
        pattern = rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])"
        if re.search(pattern, upper_text):
            spec = ADDUCT_SPECS[canonical]
            if ion_mode.lower().startswith(spec.polarity[:3]):
                return spec
    return None


def infer_adducts(record: Mapping[str, Any], charge: int, ion_mode: str, allow_common: bool) -> List[Tuple[AdductSpec, str]]:
    adduct_fields = (
        "adduct",
        "precursor_type",
        "precursortype",
        "ion_type",
        "iontype",
        "species",
        "name",
        "title",
        "compound_name",
    )
    for key in adduct_fields:
        value = first_present(record, (key,))
        if value is None:
            continue
        spec = detect_adduct_from_text(str(value), ion_mode)
        if spec and abs(spec.charge) == abs(charge):
            return [(spec, f"field:{key}")]

    compatible = [
        spec
        for spec in ADDUCT_SPECS.values()
        if abs(spec.charge) == abs(charge) and spec.polarity.lower().startswith(ion_mode.lower()[:3])
    ]
    if allow_common and compatible:
        return [(spec, "common_adduct_hypothesis") for spec in compatible]

    if ion_mode.lower().startswith("neg"):
        default_name = "[M-H]-"
    elif abs(charge) == 2:
        default_name = "[M+2H]2+"
    elif abs(charge) == 3:
        default_name = "[M+3H]3+"
    else:
        default_name = "[M+H]+"
    return [(ADDUCT_SPECS[default_name], "default_by_ion_mode_charge")]


def ppm_error(observed: float, theoretical: float) -> float:
    return (theoretical - observed) / observed * 1_000_000.0


def ppm_to_da(mass: float, ppm: float) -> float:
    return abs(mass) * ppm / 1_000_000.0


def formula_exact_mass(formula: str) -> Optional[float]:
    if not formula:
        return None
    formula = formula.strip()
    formula = re.sub(r"^[A-Za-z0-9]+=", "", formula)
    formula = re.sub(r"[\+\-]\d*$", "", formula)
    if not formula:
        return None

    # This lightweight parser supports normal molecular formulas such as
    # C14H16O8. Complex grouped formulas should be supplied with ExactMass.
    total = 0.0
    position = 0
    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    for match in pattern.finditer(formula):
        if match.start() != position:
            return None
        element = match.group(1)
        count = int(match.group(2) or "1")
        if element not in MONOISOTOPIC_MASSES:
            return None
        total += MONOISOTOPIC_MASSES[element] * count
        position = match.end()
    if position != len(formula):
        return None
    return total


def formula_from_inchi(inchi: str) -> str:
    if not inchi:
        return ""
    match = re.match(r"^InChI=[^/]+/([^/]+)", inchi.strip())
    return match.group(1) if match else ""


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return parse_float(value, "exact_mass")
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def rdkit_mol_from_smiles(smiles: str) -> Any:
    if Chem is None or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def rdkit_canonical_smiles(smiles: str) -> str:
    mol = rdkit_mol_from_smiles(smiles)
    if mol is None or Chem is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return ""


def rdkit_exact_mass(smiles: str) -> Optional[float]:
    mol = rdkit_mol_from_smiles(smiles)
    if mol is None or Descriptors is None:
        return None
    try:
        return float(Descriptors.ExactMolWt(mol))
    except Exception:
        return None


def rdkit_formula(smiles: str) -> str:
    mol = rdkit_mol_from_smiles(smiles)
    if mol is None or rdMolDescriptors is None:
        return ""
    try:
        return str(rdMolDescriptors.CalcMolFormula(mol))
    except Exception:
        return ""


def infer_structure_stability(candidate: Candidate) -> Tuple[str, str]:
    existing = str(candidate.metadata.get("stability") or candidate.stability or "").strip().lower()
    if existing in {"stable", "metastable", "unstable"}:
        return existing, "provided_by_database"

    if candidate.smiles and Chem is not None:
        mol = rdkit_mol_from_smiles(candidate.smiles)
        if mol is None:
            return "unstable", "rdkit_parse_failed"
        try:
            radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
            formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            fragments = Chem.GetMolFrags(mol)
        except Exception:
            return "metastable", "rdkit_sanitized_with_warnings"
        if radicals:
            return "unstable", "radical_electrons_detected"
        if len(fragments) > 1:
            return "metastable", "multi_fragment_salt_or_mixture"
        if formal_charge != 0:
            return "metastable", "formal_charge_detected"
        return "stable", "rdkit_valid_neutral_single_fragment"

    if candidate.smiles:
        return "metastable", "rdkit_not_available_assumed_from_smiles"
    if candidate.inchi or candidate.inchikey:
        return "metastable", "structure_identifier_available_without_smiles"
    return "unstable", "missing_structure_identifier"


def canonical_structure_key(candidate: Candidate) -> str:
    if candidate.inchikey:
        # The first block collapses stereoisomer/protonation variants less
        # aggressively than raw SMILES but still catches same-structure strings.
        return f"inchikey:{candidate.inchikey.strip().upper()}"
    if candidate.canonical_smiles:
        return f"smiles:{candidate.canonical_smiles}"
    if candidate.smiles:
        return f"raw_smiles:{candidate.smiles.strip()}"
    return f"{candidate.source}:{candidate.source_id}"


def normalize_candidate_row(row: Mapping[str, Any], source: str) -> Candidate:
    source_id = str(
        first_present(
            row,
            (
                "database_id",
                "source_id",
                "accession",
                "hmdb_id",
                "cid",
                "pubchem_cid",
                "csid",
                "chemspider_id",
                "id",
                "identifier",
            ),
            "",
        )
    )
    name = str(first_present(row, ("name", "title", "compound_name", "iupac_name", "synonym"), ""))
    smiles = str(
        first_present(
            row,
            (
                "smiles",
                "SMILES",
                "isomeric_smiles",
                "isomericsmiles",
                "canonical_smiles",
                "canonicalsmiles",
            ),
            "",
        )
    )
    inchi = str(first_present(row, ("inchi", "InChI"), ""))
    inchikey = str(first_present(row, ("inchikey", "inchi_key", "InChIKey"), ""))
    formula = str(first_present(row, ("formula", "molecular_formula", "MolecularFormula"), ""))
    if not formula and inchi:
        formula = formula_from_inchi(inchi)

    exact_mass = safe_float(
        first_present(
            row,
            (
                "exact_mass",
                "ExactMass",
                "monoisotopic_mass",
                "MonoisotopicMass",
                "monisotopic_molecular_weight",
                "monoisotopic_molecular_weight",
            ),
        )
    )
    if exact_mass is None and smiles:
        exact_mass = rdkit_exact_mass(smiles)
    if exact_mass is None and formula:
        exact_mass = formula_exact_mass(formula)

    canonical_smiles = str(first_present(row, ("canonical_smiles", "canonicalsmiles"), ""))
    if not canonical_smiles and smiles:
        canonical_smiles = rdkit_canonical_smiles(smiles)
    if not formula and smiles:
        formula = rdkit_formula(smiles)

    candidate = Candidate(
        source=source,
        source_id=source_id,
        name=name,
        smiles=smiles,
        canonical_smiles=canonical_smiles,
        inchi=inchi,
        inchikey=inchikey,
        formula=formula,
        exact_mass=exact_mass,
        record_charge=safe_int(first_present(row, ("charge", "Charge", "formal_charge", "net_charge"))),
        stability=str(first_present(row, ("stability", "structure_stability"), "unknown")).lower(),
        source_url=str(first_present(row, ("url", "source_url", "database_url"), "")),
        metadata=dict(row),
    )
    candidate.stability, candidate.stability_reason = infer_structure_stability(candidate)
    if source == "PubChem" and candidate.source_id and not candidate.source_url:
        candidate.source_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{candidate.source_id}"
    return candidate


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, Mapping)]
    if isinstance(data, Mapping):
        for key in ("records", "candidates", "data", "items", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        return [dict(data)]
    raise ValueError(f"Unsupported JSON root in {path}: expected object or list.")


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if isinstance(value, Mapping):
                records.append(dict(value))
    return records


def load_table_records(path: Path) -> List[Dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [dict(row) for row in reader]


def load_local_records(path_value: Optional[str]) -> List[Dict[str, Any]]:
    if not path_value:
        return []
    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local database file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_records(path)
    if suffix == ".jsonl":
        return load_jsonl_records(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        return load_table_records(path)
    raise ValueError(f"Unsupported local database file type: {path}")


def local_database_candidates(path_value: Optional[str], source: str) -> List[Candidate]:
    return [normalize_candidate_row(row, source) for row in load_local_records(path_value)]


def pubchem_esearch_exact_mass(
    neutral_mass: float,
    tolerance_da: float,
    retmax: int,
    session: Any,
    timeout: float,
) -> List[str]:
    if requests is None:
        raise RuntimeError("requests is not installed; PubChem online retrieval is unavailable.")
    low = neutral_mass - tolerance_da
    high = neutral_mass + tolerance_da
    term = f"{low:.8f}:{high:.8f}[ExactMass]"
    response = session.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pccompound", "term": term, "retmode": "json", "retmax": str(retmax)},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    ids = payload.get("esearchresult", {}).get("idlist", [])
    return [str(item) for item in ids]


def chunked(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def pubchem_fetch_properties(cids: Sequence[str], session: Any, timeout: float) -> List[Dict[str, Any]]:
    if not cids:
        return []
    property_sets = [
        "CanonicalSMILES,IsomericSMILES,InChI,InChIKey,MolecularFormula,ExactMass,MonoisotopicMass,Charge",
        "CanonicalSMILES,IsomericSMILES,InChI,InChIKey,MolecularFormula,ExactMass,MonoisotopicMass",
    ]
    rows: List[Dict[str, Any]] = []
    for batch in chunked(list(cids), 100):
        cid_text = ",".join(batch)
        last_error: Optional[Exception] = None
        for properties in property_sets:
            try:
                response = session.get(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_text}/property/{properties}/JSON",
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                rows.extend(payload.get("PropertyTable", {}).get("Properties", []))
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - depends on remote API
                last_error = exc
        if last_error:
            raise last_error
    return rows


def online_pubchem_candidates(
    query: QuerySpec,
    retmax: int,
    session: Any,
    timeout: float,
    sleep_seconds: float,
    warnings: List[str],
) -> List[Candidate]:
    if requests is None:
        warnings.append("PubChem online retrieval skipped: requests is not installed.")
        return []
    try:
        cids = pubchem_esearch_exact_mass(
            query.neutral_mass,
            query.neutral_mass_window_da,
            retmax=retmax,
            session=session,
            timeout=timeout,
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        rows = pubchem_fetch_properties(cids, session=session, timeout=timeout)
    except Exception as exc:
        warnings.append(f"PubChem online retrieval failed: {exc}")
        return []

    candidates: List[Candidate] = []
    for row in rows:
        row = dict(row)
        if "CID" in row and "cid" not in row:
            row["cid"] = row["CID"]
        candidate = normalize_candidate_row(row, "PubChem")
        candidates.append(candidate)
    return candidates


def input_record_candidate(record: Mapping[str, Any]) -> Optional[Candidate]:
    smiles = first_present(record, ("smiles", "SMILES"))
    inchi = first_present(record, ("inchi", "InChI"))
    if not smiles and not inchi:
        return None
    row = dict(record)
    row.setdefault("database_id", infer_record_id(record, 0))
    row.setdefault("name", first_present(record, ("name", "title", "compound_name"), "input_record"))
    row.setdefault("formula", formula_from_inchi(str(inchi or "")))
    return normalize_candidate_row(row, "InputSMILES")


def query_specs_from_record(
    record: Mapping[str, Any],
    record_index: int,
    ppm_tolerance: float,
    absolute_da_reference: float,
    allow_common_adducts: bool,
) -> List[QuerySpec]:
    observed_mz = extract_observed_mz(record)
    preliminary_mode = infer_ion_mode(record)
    charge = parse_charge(first_present(record, ("charge", "Charge")), preliminary_mode)
    ion_mode = infer_ion_mode(record, charge)
    record_id = infer_record_id(record, record_index)
    specs: List[QuerySpec] = []
    for adduct, source in infer_adducts(record, charge, ion_mode, allow_common_adducts):
        neutral_mass = adduct.neutral_mass_from_mz(observed_mz)
        neutral_window = ppm_to_da(neutral_mass, ppm_tolerance)
        specs.append(
            QuerySpec(
                record_index=record_index,
                observed_mz=observed_mz,
                charge=charge,
                ion_mode=ion_mode,
                adduct=adduct,
                neutral_mass=neutral_mass,
                neutral_mass_window_da=neutral_window,
                ppm_tolerance=ppm_tolerance,
                absolute_da_reference=absolute_da_reference,
                adduct_inference_source=source,
                record_id=record_id,
                raw_record=dict(record),
            )
        )
    return specs


def passes_charge_adduct_filter(candidate: Candidate, query: QuerySpec) -> Tuple[bool, str]:
    if abs(query.adduct.charge) != abs(query.charge):
        return False, "query_charge_adduct_mismatch"
    if query.ion_mode.lower().startswith("pos") and query.adduct.charge <= 0:
        return False, "positive_mode_requires_positive_adduct"
    if query.ion_mode.lower().startswith("neg") and query.adduct.charge >= 0:
        return False, "negative_mode_requires_negative_adduct"

    # Candidate database charge is molecular formal charge, not precursor
    # charge. Strongly charged standalone structures are kept only if the DB
    # explicitly reports them as stable/metastable and the mass still matches.
    if candidate.record_charge not in (None, 0) and candidate.stability == "unstable":
        return False, "charged_unstable_candidate"
    return True, "pass"


def filter_candidate(candidate: Candidate, query: QuerySpec) -> Tuple[bool, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {}
    if candidate.exact_mass is None:
        diagnostics["reject_reason"] = "missing_exact_mass"
        return False, diagnostics
    if candidate.stability not in {"stable", "metastable"}:
        diagnostics["reject_reason"] = "unstable_structure"
        return False, diagnostics

    charge_ok, charge_reason = passes_charge_adduct_filter(candidate, query)
    diagnostics["charge_adduct_check"] = charge_reason
    if not charge_ok:
        diagnostics["reject_reason"] = charge_reason
        return False, diagnostics

    theoretical_mz = query.adduct.mz_from_neutral_mass(candidate.exact_mass)
    error_ppm = ppm_error(query.observed_mz, theoretical_mz)
    diagnostics.update(
        {
            "theoretical_mz": theoretical_mz,
            "ppm_error": error_ppm,
            "abs_ppm_error": abs(error_ppm),
        }
    )
    if abs(error_ppm) > query.ppm_tolerance:
        diagnostics["reject_reason"] = "mass_ppm_out_of_tolerance"
        return False, diagnostics

    diagnostics["reject_reason"] = ""
    return True, diagnostics


def candidate_sort_key(candidate_payload: Dict[str, Any]) -> Tuple[float, int, int, str]:
    return (
        float(candidate_payload["abs_ppm_error"]),
        STABILITY_PRIORITY.get(str(candidate_payload["stability"]), 8),
        SOURCE_PRIORITY.get(str(candidate_payload["source"]), 99),
        str(candidate_payload.get("canonical_smiles") or candidate_payload.get("smiles") or ""),
    )


def candidate_to_payload(candidate: Candidate, query: QuerySpec, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": candidate.source,
        "source_priority": SOURCE_PRIORITY.get(candidate.source, 99),
        "database_id": candidate.source_id,
        "name": candidate.name,
        "smiles": candidate.smiles,
        "canonical_smiles": candidate.canonical_smiles,
        "inchi_key": candidate.inchikey,
        "inchi": candidate.inchi,
        "formula": candidate.formula,
        "exact_mass": candidate.exact_mass,
        "assigned_adduct": query.adduct.name,
        "precursor_charge": query.adduct.charge,
        "ion_mode": query.ion_mode,
        "observed_mz": query.observed_mz,
        "theoretical_mz": diagnostics.get("theoretical_mz"),
        "ppm_error": diagnostics.get("ppm_error"),
        "abs_ppm_error": diagnostics.get("abs_ppm_error"),
        "stability": candidate.stability,
        "stability_reason": candidate.stability_reason,
        "source_url": candidate.source_url,
        "dedupe_key": canonical_structure_key(candidate),
    }


def dedupe_candidates(payloads: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[str, Dict[str, Any]] = {}
    duplicates_by_key: Dict[str, List[Dict[str, Any]]] = {}
    for payload in payloads:
        key = str(payload["dedupe_key"])
        existing = best_by_key.get(key)
        if existing is None or candidate_sort_key(payload) < candidate_sort_key(existing):
            if existing is not None:
                duplicates_by_key.setdefault(key, []).append(existing)
            best_by_key[key] = payload
        else:
            duplicates_by_key.setdefault(key, []).append(payload)

    results = list(best_by_key.values())
    for payload in results:
        duplicates = duplicates_by_key.get(str(payload["dedupe_key"]), [])
        payload["duplicate_count_removed"] = len(duplicates)
        payload["duplicate_sources_removed"] = sorted({str(item.get("source")) for item in duplicates})
    return sorted(results, key=candidate_sort_key)


def collect_candidates_for_query(
    query: QuerySpec,
    args: argparse.Namespace,
    session: Any,
    local_candidates: Mapping[str, List[Candidate]],
    warnings: List[str],
) -> List[Candidate]:
    candidates: List[Candidate] = []
    # Database authority order is kept explicit for traceability.
    candidates.extend(local_candidates.get("HMDB", []))
    candidates.extend(local_candidates.get("PubChem", []))
    if not args.no_online_pubchem:
        candidates.extend(
            online_pubchem_candidates(
                query=query,
                retmax=args.pubchem_retmax,
                session=session,
                timeout=args.timeout,
                sleep_seconds=args.sleep_seconds,
                warnings=warnings,
            )
        )
    candidates.extend(local_candidates.get("ChemSpider", []))

    if args.chemspider_api_key:
        warnings.append(
            "ChemSpider API key detected, but direct ChemSpider remote retrieval is not enabled in this script. "
            "Use --chemspider-file with an exported table to keep the retrieval reproducible."
        )

    if args.include_input_smiles:
        own_candidate = input_record_candidate(query.raw_record)
        if own_candidate is not None:
            candidates.append(own_candidate)
    return candidates


def run_single_query(
    query: QuerySpec,
    args: argparse.Namespace,
    session: Any,
    local_candidates: Mapping[str, List[Candidate]],
) -> Dict[str, Any]:
    warnings: List[str] = []
    candidates = collect_candidates_for_query(query, args, session, local_candidates, warnings)

    accepted_payloads: List[Dict[str, Any]] = []
    rejection_counts: Dict[str, int] = {}
    for candidate in candidates:
        accepted, diagnostics = filter_candidate(candidate, query)
        if not accepted:
            reason = str(diagnostics.get("reject_reason") or "unknown")
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        accepted_payloads.append(candidate_to_payload(candidate, query, diagnostics))

    deduped = dedupe_candidates(accepted_payloads)
    for rank, payload in enumerate(deduped[: args.top_k], start=1):
        payload["rank"] = rank

    absolute_reference_ppm = (
        query.absolute_da_reference / query.observed_mz * 1_000_000.0 if query.observed_mz else None
    )
    return {
        "record_id": query.record_id,
        "record_index": query.record_index,
        "query": {
            "observed_mz": query.observed_mz,
            "charge": query.charge,
            "ion_mode": query.ion_mode,
            "adduct": query.adduct.name,
            "adduct_inference_source": query.adduct_inference_source,
            "neutral_mass": query.neutral_mass,
            "ppm_tolerance": query.ppm_tolerance,
            "neutral_mass_window_da": query.neutral_mass_window_da,
            "absolute_da_reference": query.absolute_da_reference,
            "absolute_da_reference_as_ppm": absolute_reference_ppm,
        },
        "retrieval_summary": {
            "candidate_count_before_filter": len(candidates),
            "candidate_count_after_rigid_filter": len(accepted_payloads),
            "candidate_count_after_dedup": len(deduped),
            "returned_top_k": min(args.top_k, len(deduped)),
            "rejection_counts": rejection_counts,
            "warnings": warnings,
        },
        "candidates": deduped[: args.top_k],
    }


def read_input_records(input_path: str) -> List[Dict[str, Any]]:
    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input JSON file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        return load_jsonl_records(path)
    return load_json_records(path)


def build_local_candidate_cache(args: argparse.Namespace) -> Dict[str, List[Candidate]]:
    return {
        "HMDB": local_database_candidates(args.hmdb_file, "HMDB"),
        "PubChem": local_database_candidates(args.pubchem_file, "PubChem"),
        "ChemSpider": local_database_candidates(args.chemspider_file, "ChemSpider"),
    }


def write_json(path_value: str, payload: Any) -> None:
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    records = read_input_records(args.input)
    local_candidates = build_local_candidate_cache(args)

    session = requests.Session() if requests is not None else None
    if session is not None:
        session.headers.update(
            {
                "User-Agent": "llm-ms-slime-precursor-smiles-candidate-retrieval/1.0 "
                "(exact-mass candidate retrieval; contact: local-user)"
            }
        )

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for record_index, record in enumerate(records):
        try:
            query_specs = query_specs_from_record(
                record=record,
                record_index=record_index,
                ppm_tolerance=args.ppm_tolerance,
                absolute_da_reference=args.absolute_da_reference,
                allow_common_adducts=args.allow_common_adducts,
            )
            for query in query_specs:
                results.append(run_single_query(query, args, session, local_candidates))
        except Exception as exc:
            errors.append(
                {
                    "record_index": record_index,
                    "record_id": infer_record_id(record, record_index) if isinstance(record, Mapping) else "",
                    "error": str(exc),
                }
            )

    output_payload = {
        "schema": "precursor_smiles_candidate_retrieval.v1",
        "configuration": {
            "top_k": args.top_k,
            "ppm_tolerance": args.ppm_tolerance,
            "absolute_da_reference": args.absolute_da_reference,
            "sources": {
                "HMDB": {"local_file": args.hmdb_file, "priority": SOURCE_PRIORITY["HMDB"]},
                "PubChem": {
                    "local_file": args.pubchem_file,
                    "online_enabled": not args.no_online_pubchem,
                    "priority": SOURCE_PRIORITY["PubChem"],
                },
                "ChemSpider": {"local_file": args.chemspider_file, "priority": SOURCE_PRIORITY["ChemSpider"]},
            },
            "rdkit_available": Chem is not None,
        },
        "local_database_counts": {source: len(items) for source, items in local_candidates.items()},
        "result_count": len(results),
        "error_count": len(errors),
        "results": results,
        "errors": errors,
    }
    write_json(args.output, output_payload)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser()),
                "records": len(records),
                "results": len(results),
                "errors": len(errors),
                "rdkit_available": Chem is not None,
            },
            ensure_ascii=False,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
