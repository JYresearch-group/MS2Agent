#!/usr/bin/env python3
"""
Attach precursor-mass retrieval results to each structured MS2 JSON record.

Input:
  A JSON list of records with pepmass/charge/ionmode/name/smiles/inchi fields.

Output:
  The same records, each augmented with:
    - adduct
    - retrieval_query
    - retrieval_candidate_count
    - retrieval_top50

This script does not parse MGF. It uses the input file itself as the local
candidate library.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import evaluate_precursor_smiles_recall as eval_recall
import precursor_smiles_candidate_retrieval as retrieval


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach top-k precursor retrieval candidates to JSON records.")
    parser.add_argument("--input", required=True, help="Input structured JSON file.")
    parser.add_argument("--output", required=True, help="Output augmented JSON file.")
    parser.add_argument("--top-k", type=int, default=50, help="Maximum retrieval candidates to attach.")
    parser.add_argument("--ppm-tolerance", type=float, default=10.0, help="Mass tolerance in ppm.")
    parser.add_argument(
        "--candidate-window-ppm",
        type=float,
        default=10.0,
        help="Mass index pre-window; final filtering still uses --ppm-tolerance.",
    )
    parser.add_argument(
        "--absolute-da-reference",
        type=float,
        default=0.01,
        help="Reference Da window stored in retrieval_query diagnostics.",
    )
    parser.add_argument(
        "--allow-common-adducts",
        action="store_true",
        help="If no explicit adduct exists, evaluate all compatible common adducts and keep the richest result.",
    )
    return parser.parse_args(argv)


def compact_candidate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep retrieval results useful but not unnecessarily huge."""
    return {
        "rank": payload.get("rank"),
        "candidate_id": payload.get("database_id"),
        "source": payload.get("source"),
        "name": payload.get("name"),
        "smiles": payload.get("smiles"),
        "inchi_key": payload.get("inchi_key"),
        "formula": payload.get("formula"),
        "exact_mass": payload.get("exact_mass"),
        "adduct": payload.get("assigned_adduct"),
        "observed_mz": payload.get("observed_mz"),
        "theoretical_mz": payload.get("theoretical_mz"),
        "ppm_error": payload.get("ppm_error"),
        "abs_ppm_error": payload.get("abs_ppm_error"),
        "stability": payload.get("stability"),
    }


def choose_query_result(
    record: Dict[str, Any],
    record_index: int,
    candidates: Sequence[retrieval.Candidate],
    masses: Sequence[float],
    mass_indices: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    query_specs = retrieval.query_specs_from_record(
        record=record,
        record_index=record_index,
        ppm_tolerance=args.ppm_tolerance,
        absolute_da_reference=args.absolute_da_reference,
        allow_common_adducts=args.allow_common_adducts,
    )

    best: Optional[Dict[str, Any]] = None
    for query in query_specs:
        pool = eval_recall.fetch_mass_window_candidates(
            query,
            masses,
            mass_indices,
            candidates,
            candidate_window_ppm=args.candidate_window_ppm,
        )
        ranked, rejection_counts = eval_recall.rank_candidates_for_query(query, pool, top_k=args.top_k)
        result = {
            "query": query,
            "candidate_pool_count": len(pool),
            "ranked": ranked,
            "rejection_counts": rejection_counts,
        }
        if best is None:
            best = result
            continue
        # Prefer the adduct hypothesis that returns more valid candidates; tie by
        # smaller top-candidate ppm error.
        current_top_ppm = abs(float(ranked[0].get("ppm_error", math.inf))) if ranked else math.inf
        best_ranked = best["ranked"]
        best_top_ppm = abs(float(best_ranked[0].get("ppm_error", math.inf))) if best_ranked else math.inf
        if len(ranked) > len(best_ranked) or (len(ranked) == len(best_ranked) and current_top_ppm < best_top_ppm):
            best = result

    if best is None:
        raise ValueError("No query spec could be generated.")
    return best


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    records = retrieval.read_input_records(args.input)
    candidates, library_stats = eval_recall.build_candidate_library(records)
    masses, mass_indices = eval_recall.build_mass_index(candidates)

    augmented: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        output_record = dict(record)
        try:
            result = choose_query_result(record, idx, candidates, masses, mass_indices, args)
            query: retrieval.QuerySpec = result["query"]
            ranked = result["ranked"]
            output_record["adduct"] = query.adduct.name
            output_record["retrieval_query"] = {
                "record_id": query.record_id,
                "observed_mz": query.observed_mz,
                "charge": query.charge,
                "ion_mode": query.ion_mode,
                "adduct": query.adduct.name,
                "adduct_inference_source": query.adduct_inference_source,
                "neutral_mass": query.neutral_mass,
                "ppm_tolerance": query.ppm_tolerance,
                "candidate_window_ppm": args.candidate_window_ppm,
                "neutral_mass_window_da": query.neutral_mass_window_da,
            }
            output_record["retrieval_candidate_count"] = len(ranked)
            output_record["retrieval_pool_count_before_filter"] = result["candidate_pool_count"]
            output_record["retrieval_rejection_counts"] = result["rejection_counts"]
            output_record["retrieval_top50"] = [compact_candidate_payload(item) for item in ranked[: args.top_k]]
        except Exception as exc:
            output_record["adduct"] = None
            output_record["retrieval_query"] = None
            output_record["retrieval_candidate_count"] = 0
            output_record["retrieval_top50"] = []
            output_record["retrieval_error"] = str(exc)
            error_rows.append(
                {
                    "record_index": idx,
                    "record_id": retrieval.infer_record_id(record, idx),
                    "error": str(exc),
                }
            )
        augmented.append(output_record)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(augmented, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input": str(Path(args.input).expanduser()),
        "output": str(output_path),
        "record_count": len(records),
        "error_count": len(error_rows),
        "top_k": args.top_k,
        "ppm_tolerance": args.ppm_tolerance,
        "library_stats": library_stats,
        "mean_retrieval_candidate_count": sum(row.get("retrieval_candidate_count", 0) for row in augmented)
        / len(augmented)
        if augmented
        else 0.0,
        "max_retrieval_candidate_count": max((row.get("retrieval_candidate_count", 0) for row in augmented), default=0),
        "errors": error_rows[:20],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not error_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
