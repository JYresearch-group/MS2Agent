from typing import Any, Dict, List, Optional, Tuple

from casmi_pipeline.io_utils import first_present, normalize_space, safe_float, safe_int
from casmi_pipeline.settings import (
    ADDUCT_ALIASES,
    ADDUCT_SPECS,
    PRIMARY_NEGATIVE_ADDUCTS,
    PRIMARY_POSITIVE_ADDUCTS,
)


def extract_record_id(record: Dict[str, Any], idx: int) -> str:
    for key in ("spectrum_id", "SpectrumID", "fn"):
        value = normalize_space(record.get(key))
        if value:
            return value
    return f"record_{idx:06d}"


def normalize_peaks(raw_peaks: Any) -> List[Tuple[float, float]]:
    peaks: List[Tuple[float, float]] = []
    if not isinstance(raw_peaks, list):
        return peaks
    for item in raw_peaks:
        if not isinstance(item, list) or len(item) < 2:
            continue
        mz = safe_float(item[0])
        intensity = safe_float(item[1])
        if mz is None or intensity is None:
            continue
        peaks.append((mz, intensity))
    peaks.sort(key=lambda x: x[0])
    return peaks


def normalize_input_record(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
    raw_adduct = normalize_space(
        first_present(record, ["adduct", "Adduct", "precursor_type", "Precursor_Type"], "")
    )
    adduct_spec = get_adduct_spec(raw_adduct)
    return {
        "record_id": extract_record_id(record, idx),
        "source_fn": normalize_space(first_present(record, ["fn"], "")),
        "parent_mz": safe_float(first_present(record, ["parent_mz"])),
        "charge": safe_int(first_present(record, ["Charge", "charge"])),
        "ms_level": safe_int(first_present(record, ["ms_level"])),
        "ion_mode": normalize_space(first_present(record, ["Ion_Mode", "ion_mode"])),
        "ion_source": normalize_space(first_present(record, ["Ion_Source", "ion_source"])),
        "instrument": normalize_space(first_present(record, ["Instrument", "instrument"])),
        "scan": safe_int(first_present(record, ["scan"])),
        "smiles": normalize_space(first_present(record, ["smiles", "SMILES"])),
        "adduct": adduct_spec["name"] if adduct_spec is not None else raw_adduct,
        "peaks": normalize_peaks(first_present(record, ["ms"], [])),
    }


def infer_ms_mass_anchor(
    parent_mz: Optional[float],
    ion_mode: str,
    charge: Optional[int],
    adduct: str = "",
) -> Dict[str, Any]:
    likely_ion_forms: List[str] = []
    neutral_mass_hypotheses: List[float] = []
    normalized_adduct = normalize_space(adduct)
    explicit_spec = get_adduct_spec(normalized_adduct) if normalized_adduct else None

    if parent_mz is None:
        return {
            "likely_ion_forms": [explicit_spec["name"] if explicit_spec else normalized_adduct or "unknown"],
            "neutral_mass_hypotheses": [],
            "adduct_source": "input" if normalized_adduct else "unavailable",
        }

    if explicit_spec is not None:
        adducts = [explicit_spec["name"]]
        adduct_source = "input"
    elif normalized_adduct:
        return {
            "likely_ion_forms": [normalized_adduct],
            "neutral_mass_hypotheses": [],
            "adduct_source": "input_unrecognized",
        }
    elif charge in (None, 0):
        return {
            "likely_ion_forms": ["unknown"],
            "neutral_mass_hypotheses": [],
            "adduct_source": "unavailable",
        }
    else:
        adducts = default_adducts_for_ion_mode(ion_mode, charge=charge, include_oligomers=False)
        adduct_source = "inferred_from_ion_mode_and_charge"

    for adduct in adducts:
        spec = get_adduct_spec(adduct)
        if spec is None:
            continue
        ion_charge = abs(int(spec["charge"]))
        multiplier = int(spec["multiplier"])
        delta = float(spec["delta"])
        neutral_mass = ((parent_mz * ion_charge) - delta) / multiplier
        if neutral_mass <= 0:
            continue
        likely_ion_forms.append(adduct)
        neutral_mass_hypotheses.append(round(neutral_mass, 6))

    if not likely_ion_forms:
        likely_ion_forms = ["unknown"]

    return {
        "likely_ion_forms": likely_ion_forms,
        "neutral_mass_hypotheses": neutral_mass_hypotheses,
        "adduct_source": adduct_source,
    }


def classify_precursor_dominance(relative_intensity: Optional[float]) -> str:
    if relative_intensity is None:
        return "unknown"
    if relative_intensity >= 90.0:
        return "very_high"
    if relative_intensity >= 50.0:
        return "high"
    if relative_intensity >= 20.0:
        return "medium"
    return "low"


def classify_fragmentation_richness(fragment_peak_count: int) -> str:
    if fragment_peak_count <= 2:
        return "very_low"
    if fragment_peak_count <= 6:
        return "low"
    if fragment_peak_count <= 15:
        return "medium"
    return "high"


def get_adduct_spec(adduct: str) -> Optional[Dict[str, Any]]:
    canonical = ADDUCT_ALIASES.get(adduct, adduct)
    spec = ADDUCT_SPECS.get(canonical)
    if spec is None:
        return None
    return {"name": canonical, **spec}


def default_adducts_for_ion_mode(
    ion_mode: str,
    charge: Optional[int] = 1,
    include_oligomers: bool = False,
) -> List[str]:
    normalized_mode = normalize_space(ion_mode).lower()
    if normalized_mode == "positive":
        pool = PRIMARY_POSITIVE_ADDUCTS if not include_oligomers else [
            key for key, spec in ADDUCT_SPECS.items() if spec["mode"] == "positive"
        ]
    elif normalized_mode == "negative":
        pool = PRIMARY_NEGATIVE_ADDUCTS if not include_oligomers else [
            key for key, spec in ADDUCT_SPECS.items() if spec["mode"] == "negative"
        ]
    else:
        return []

    if charge in (None, 0):
        return list(pool)

    return [
        adduct for adduct in pool
        if abs(int(ADDUCT_SPECS[ADDUCT_ALIASES.get(adduct, adduct)]["charge"])) == abs(int(charge))
    ]


def summarize_peaks(
    peaks: List[Tuple[float, float]],
    parent_mz: Optional[float],
    top_k: int,
    precursor_tolerance: float = 0.01,
) -> Dict[str, Any]:
    if not peaks:
        return {
            "peak_count": 0,
            "total_ion_current": None,
            "base_peak": None,
            "precursor_peak": None,
            "top_peaks": [],
            "fragment_peak_count": 0,
        }

    total_ion_current = sum(intensity for _, intensity in peaks)
    base_mz, base_intensity = max(peaks, key=lambda x: x[1])
    top_peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)[:top_k]
    top_peaks = []
    for mz, intensity in top_peaks_sorted:
        rel = 100.0 * intensity / base_intensity if base_intensity else 0.0
        top_peaks.append(
            {
                "mz": round(mz, 6),
                "intensity": round(intensity, 4),
                "relative_intensity": round(rel, 4),
            }
        )

    precursor_peak = None
    if parent_mz is not None:
        nearest = min(peaks, key=lambda x: abs(x[0] - parent_mz))
        if abs(nearest[0] - parent_mz) <= precursor_tolerance:
            precursor_peak = {
                "mz": round(nearest[0], 6),
                "intensity": round(nearest[1], 4),
                "relative_intensity": round(100.0 * nearest[1] / base_intensity, 4),
                "mz_error": round(nearest[0] - parent_mz, 6),
            }

    if parent_mz is not None:
        fragment_peak_count = sum(1 for mz, _ in peaks if abs(mz - parent_mz) > precursor_tolerance)
    else:
        fragment_peak_count = len(peaks)

    fragment_top_peaks = []
    neutral_losses: List[float] = []
    for peak in top_peaks:
        mz = safe_float(peak.get("mz"))
        if mz is None:
            continue
        if parent_mz is not None and abs(mz - parent_mz) <= precursor_tolerance:
            continue
        fragment_top_peaks.append(peak)
        if parent_mz is not None and mz < parent_mz:
            loss = parent_mz - mz
            if 0.5 <= loss <= 300:
                rounded_loss = round(loss, 4)
                if rounded_loss not in neutral_losses:
                    neutral_losses.append(rounded_loss)

    return {
        "peak_count": len(peaks),
        "total_ion_current": round(total_ion_current, 4),
        "base_peak": {
            "mz": round(base_mz, 6),
            "intensity": round(base_intensity, 4),
            "relative_intensity": 100.0,
        },
        "precursor_peak": precursor_peak,
        "top_peaks": top_peaks,
        "top_fragment_peaks": fragment_top_peaks[:top_k],
        "fragment_peak_count": fragment_peak_count,
        "neutral_losses": neutral_losses[: min(top_k, 6)],
        "precursor_dominance": classify_precursor_dominance(
            safe_float(precursor_peak.get("relative_intensity")) if precursor_peak else None
        ),
        "fragmentation_richness": classify_fragmentation_richness(fragment_peak_count),
    }
