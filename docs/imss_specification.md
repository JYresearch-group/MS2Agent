# IMSS — Interpretable Mass Spectrometry Standard

IMSS is the structured reporting specification used by MS2Agent. A "standardized report" is a compact machine-readable retrieval signature, not an unrestricted narrative report.

## Numerical Representation

- **Relative intensity**: `100.0 * I_j / I_base` on [0, 100] scale, base peak = 100.0
- **m/z precision**: preprocessing retains 6 decimal places; summary text uses 4 decimal places
- **Neutral-loss values**: retained to 4 decimal places
- **Precursor matching tolerance**: 0.01 Da
- **Candidate mass-filter parameter**: 0.05 Da
- **Neutral-loss range**: 0.5 ≤ Δ(m/z) ≤ 300 Da (without charge/adduct correction)

## MS Report Template

```json
{
  "analysis_type": "ms_retrieval_signature",
  "mass_anchor": {
    "likely_ion_forms": ["<ion_form>"],
    "neutral_mass_hypotheses": []
  },
  "spectrum_profile": {
    "precursor_dominance": "very_high|high|medium|low|unknown",
    "fragmentation_richness": "very_low|low|medium|high|unknown",
    "top_fragments_mz": [],
    "neutral_losses": []
  },
  "chemical_hints": ["<chemical_hint>"],
  "retrieval_terms": ["<retrieval_term>"]
}
```

## Structure Report Template

```json
{
  "analysis_type": "structure_retrieval_signature",
  "consistency_check": "consistent|partially_consistent|uncertain|inconsistent",
  "structure_profile": {
    "scaffold": "<scaffold_label>",
    "ring_system": "<ring_system_label>",
    "heteroatom_signature": "<heteroatom_counts>",
    "functional_groups": ["<functional_group>"],
    "acid_base_class": "strong_acid|weak_acid|neutral|basic|zwitterionic|unknown",
    "shape_class": "fused_aromatic|planar_aromatic|flexible_aliphatic|mixed|unknown"
  },
  "fragmentation_priors": {
    "likely_losses": ["<loss_label>"],
    "fragile_bonds": ["<bond_label>"]
  },
  "retrieval_terms": ["<retrieval_term>"]
}
```

## Reranker Output Templates

### Choice Mode (single best match)

```json
{
  "selected_record_id": "<candidate_id>",
  "selected_smiles": "<smiles>",
  "confidence": "low|medium|high",
  "decision_tags": ["<evidence_tag>"]
}
```

### Ranking Mode (complete ordering)

```json
{
  "ranked_record_ids": ["<candidate_id_1>", "<candidate_id_2>"],
  "top_choice_record_id": "<candidate_id_1>",
  "confidence": "low|medium|high",
  "ranking_tags": ["<evidence_tag>"]
}
```

## Scoring Criteria

See Supplementary Information Tables S7 (MS report) and S8 (Structure report) for the detailed expert scoring criteria (full score: 100).
