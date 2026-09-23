from typing import Any, Dict, Sequence

from casmi_pipeline.io_utils import (
    ensure_float_list,
    ensure_string_list,
    format_optional_number,
    safe_float,
    sanitize_short_text,
)


def normalize_ms_analysis(raw: Dict[str, Any]) -> Dict[str, Any]:
    mass_anchor = raw.get("mass_anchor") if isinstance(raw.get("mass_anchor"), dict) else {}
    spectrum_profile = raw.get("spectrum_profile") if isinstance(raw.get("spectrum_profile"), dict) else {}
    return {
        "analysis_type": "ms_retrieval_signature",
        "likely_ion_forms": ensure_string_list(mass_anchor.get("likely_ion_forms"), max_items=4),
        "neutral_mass_hypotheses": ensure_float_list(mass_anchor.get("neutral_mass_hypotheses"), max_items=4),
        "precursor_dominance": sanitize_short_text(spectrum_profile.get("precursor_dominance")),
        "fragmentation_richness": sanitize_short_text(spectrum_profile.get("fragmentation_richness")),
        "top_fragments_mz": ensure_float_list(spectrum_profile.get("top_fragments_mz"), max_items=6, ndigits=4),
        "neutral_losses": ensure_float_list(spectrum_profile.get("neutral_losses"), max_items=6, ndigits=4),
        "chemical_hints": ensure_string_list(raw.get("chemical_hints"), max_items=6),
        "retrieval_terms": ensure_string_list(raw.get("retrieval_terms"), max_items=6),
    }


def normalize_structure_analysis(raw: Dict[str, Any]) -> Dict[str, Any]:
    structure_profile = raw.get("structure_profile") if isinstance(raw.get("structure_profile"), dict) else {}
    fragmentation_priors = raw.get("fragmentation_priors") if isinstance(raw.get("fragmentation_priors"), dict) else {}
    return {
        "analysis_type": "structure_retrieval_signature",
        "consistency_check": sanitize_short_text(raw.get("consistency_check")),
        "scaffold": sanitize_short_text(structure_profile.get("scaffold")),
        "ring_system": sanitize_short_text(structure_profile.get("ring_system")),
        "heteroatom_signature": sanitize_short_text(structure_profile.get("heteroatom_signature")),
        "functional_groups": ensure_string_list(structure_profile.get("functional_groups"), max_items=8),
        "acid_base_class": sanitize_short_text(structure_profile.get("acid_base_class")),
        "shape_class": sanitize_short_text(structure_profile.get("shape_class")),
        "likely_losses": ensure_string_list(fragmentation_priors.get("likely_losses"), max_items=6),
        "fragile_bonds": ensure_string_list(fragmentation_priors.get("fragile_bonds"), max_items=6),
        "retrieval_terms": ensure_string_list(raw.get("retrieval_terms"), max_items=6),
    }


def float_list_text(values: Sequence[float], ndigits: int = 4) -> str:
    return "; ".join(format_optional_number(safe_float(value), ndigits) for value in values) if values else "unknown"


def make_ms_summary_text(
    record: Dict[str, Any],
    peak_summary: Dict[str, Any],
    ms_mass_anchor: Dict[str, Any],
    analysis: Dict[str, Any],
) -> str:
    likely_ion_forms = analysis["likely_ion_forms"] or ms_mass_anchor.get("likely_ion_forms") or ["unknown"]
    neutral_mass_hypotheses = analysis["neutral_mass_hypotheses"] or ms_mass_anchor.get("neutral_mass_hypotheses") or []
    top_fragments_mz = analysis["top_fragments_mz"] or [
        safe_float(peak.get("mz")) for peak in peak_summary.get("top_fragment_peaks", [])
    ]
    neutral_losses = analysis["neutral_losses"] or peak_summary.get("neutral_losses", [])

    lines = [
        "analysis_type: ms_retrieval_signature",
        f"precursor_mz: {format_optional_number(record['parent_mz'], 4)}",
        f"charge: {record['charge'] if record['charge'] is not None else 'unknown'}",
        f"ion_mode: {sanitize_short_text(record['ion_mode'])}",
        f"observed_adduct: {sanitize_short_text(record.get('adduct'))}",
        "likely_ion_forms: " + ("; ".join(likely_ion_forms) or "unknown"),
        f"neutral_mass_hypotheses: {float_list_text(neutral_mass_hypotheses)}",
        f"peak_count: {peak_summary['peak_count']}",
        f"fragment_peak_count: {peak_summary['fragment_peak_count']}",
        f"precursor_dominance: {analysis['precursor_dominance'] if analysis['precursor_dominance'] != 'unknown' else peak_summary['precursor_dominance']}",
        f"fragmentation_richness: {analysis['fragmentation_richness'] if analysis['fragmentation_richness'] != 'unknown' else peak_summary['fragmentation_richness']}",
        f"top_fragments_mz: {float_list_text([value for value in top_fragments_mz if value is not None])}",
        f"neutral_losses: {float_list_text([value for value in neutral_losses if value is not None])}",
        "chemical_hints: " + ("; ".join(analysis["chemical_hints"]) or "unknown"),
        "retrieval_terms: " + ("; ".join(analysis["retrieval_terms"]) or "unknown"),
    ]
    return "\n".join(lines)


def make_structure_summary_text(
    record: Dict[str, Any],
    structure_features: Dict[str, Any],
    analysis: Dict[str, Any],
) -> str:
    heteroatom_signature = analysis["heteroatom_signature"]
    if heteroatom_signature == "unknown":
        heteroatom_signature = structure_features["heteroatom_signature"]

    lines = [
        "analysis_type: structure_retrieval_signature",
        f"smiles: {record['smiles'] or 'unknown'}",
        f"exact_mass: {format_optional_number(structure_features['exact_mass'], 4)}",
        f"molecular_formula: {structure_features['molecular_formula']}",
        f"heteroatom_signature: {heteroatom_signature}",
        f"hbd: {structure_features['hbd'] if structure_features['hbd'] is not None else 'unknown'}",
        f"hba: {structure_features['hba'] if structure_features['hba'] is not None else 'unknown'}",
        f"xlogp_class: {structure_features['xlogp_class']}",
        "adduct_compatibility: " + ("; ".join(structure_features["adduct_compatibility"]) or "unknown"),
        f"consistency_check: {analysis['consistency_check']}",
        f"scaffold: {analysis['scaffold']}",
        f"ring_system: {analysis['ring_system']}",
        "functional_groups: " + ("; ".join(analysis["functional_groups"]) or "unknown"),
        f"acid_base_class: {analysis['acid_base_class']}",
        f"shape_class: {analysis['shape_class']}",
        "likely_losses: " + ("; ".join(analysis["likely_losses"]) or "unknown"),
        "fragile_bonds: " + ("; ".join(analysis["fragile_bonds"]) or "unknown"),
        "retrieval_terms: " + ("; ".join(analysis["retrieval_terms"]) or "unknown"),
    ]
    return "\n".join(lines)


def make_combined_summary_text(ms_summary_text: str, structure_summary_text: str) -> str:
    return ms_summary_text + "\n\n" + structure_summary_text
