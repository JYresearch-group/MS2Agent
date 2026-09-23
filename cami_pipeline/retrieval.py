from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from casmi_pipeline.io_utils import safe_float
from casmi_pipeline.preprocess import default_adducts_for_ion_mode, get_adduct_spec


def normalize_embeddings(matrix: Any) -> Any:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def score_matrix(query_embeddings: Any, candidate_embeddings: Any) -> Any:
    query = normalize_embeddings(query_embeddings)
    candidate = normalize_embeddings(candidate_embeddings)
    return np.matmul(query, candidate.T)


def expected_mz(exact_mass: float, adduct: str) -> float | None:
    spec = get_adduct_spec(adduct)
    if spec is None:
        return None
    multiplier = int(spec["multiplier"])
    delta = float(spec["delta"])
    charge = abs(int(spec["charge"]))
    return (exact_mass * multiplier + delta) / charge


def candidate_mask_for_query(
    query_row: Dict[str, Any],
    candidate_rows: Sequence[Dict[str, Any]],
    mass_tolerance: float,
) -> np.ndarray:
    query_mz = safe_float(query_row.get("normalized_record", {}).get("parent_mz"))
    normalized_record = query_row.get("normalized_record", {})
    ion_mode = normalized_record.get("ion_mode", "")
    charge = normalized_record.get("charge")
    observed_adduct = normalized_record.get("adduct", "")
    observed_adduct_spec = get_adduct_spec(observed_adduct)
    if observed_adduct_spec is not None:
        adducts = [observed_adduct_spec["name"]]
    else:
        adducts = default_adducts_for_ion_mode(ion_mode, charge=charge, include_oligomers=False)
    mask = np.ones(len(candidate_rows), dtype=bool)

    if query_mz is None or not adducts:
        return mask

    any_exact_mass = False
    valid = np.zeros(len(candidate_rows), dtype=bool)
    for idx, row in enumerate(candidate_rows):
        exact_mass = safe_float(row.get("structure_features", {}).get("exact_mass"))
        if exact_mass is None:
            valid[idx] = True
            continue
        any_exact_mass = True
        for adduct in adducts:
            mz = expected_mz(exact_mass, adduct)
            if mz is not None and abs(mz - query_mz) <= mass_tolerance:
                valid[idx] = True
                break

    if any_exact_mass and valid.any():
        return valid
    return mask


def summarize_hits(hit_ranks: Iterable[int], topk: Sequence[int]) -> Dict[int, float]:
    ranks = list(hit_ranks)
    if not ranks:
        return {k: 0.0 for k in topk}
    return {k: float(sum(rank <= k for rank in ranks)) / len(ranks) for k in topk}


def evaluate_retrieval(
    ms_embeddings: Any,
    structure_embeddings: Any,
    analysis_rows: Sequence[Dict[str, Any]],
    topk: Sequence[int] = (1, 3, 5, 10),
    mass_tolerance: float = 0.05,
    res_index_data=None,
) -> Dict[str, Any]:
    similarity = score_matrix(ms_embeddings, structure_embeddings)

    raw_ranks: List[int] = []
    filtered_ranks: List[int] = []
    candidate_counts: List[int] = []
    per_query_rankings: List[Dict[str, Any]] = []

    for idx, row in enumerate(analysis_rows):
        raw_order = np.argsort(-similarity[idx]).tolist()
        raw_rank = raw_order.index(idx) + 1
        raw_ranks.append(raw_rank)

        mask = candidate_mask_for_query(row, analysis_rows, mass_tolerance)
        candidate_counts.append(int(mask.sum()))
        filtered_scores = similarity[idx].copy()
        filtered_scores[~mask] = -np.inf
        filtered_order = np.argsort(-filtered_scores).tolist()
        filtered_rank = filtered_order.index(idx) + 1
        filtered_ranks.append(filtered_rank)

        if res_index_data:
            raw_order = res_index_data[idx]['index']

        raw_ranked_record_ids = [analysis_rows[j]["record_id"] for j in raw_order]
        filtered_ranked_record_ids = [analysis_rows[j]["record_id"] for j in filtered_order if mask[j]]
        per_query_rankings.append(
            {
                "query_record_id": row["record_id"],
                "target_record_id": analysis_rows[idx]["record_id"],
                "raw_rank": raw_rank,
                "filtered_rank": filtered_rank,
                "candidate_count_after_mass_filter": int(mask.sum()),
                "raw_ranked_record_ids": raw_ranked_record_ids,
                "filtered_ranked_record_ids": filtered_ranked_record_ids,
            }
        )

    return {
        "embedding_only": summarize_hits(raw_ranks, topk),
        "mass_filtered": summarize_hits(filtered_ranks, topk),
        "rank_stats": {
            "raw_mean_rank": float(np.mean(raw_ranks)),
            "raw_median_rank": float(np.median(raw_ranks)),
            "mass_filtered_mean_rank": float(np.mean(filtered_ranks)),
            "mass_filtered_median_rank": float(np.median(filtered_ranks)),
            "mean_candidate_count_after_mass_filter": float(np.mean(candidate_counts)),
            "median_candidate_count_after_mass_filter": float(np.median(candidate_counts)),
        },
        "per_query_rankings": per_query_rankings,
    }
