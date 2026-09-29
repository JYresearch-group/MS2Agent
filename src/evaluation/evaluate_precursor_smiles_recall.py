#!/usr/bin/env python3
"""
Evaluate precursor-mass SMILES candidate recall on structured JSON records.

This evaluation is deliberately offline and reproducible:
- Input records act as the candidate structure library.
- Each query uses only precursor m/z, charge, ion mode, and name/adduct fields.
- The target answer is the record's own SMILES.

The script reports recall@K and writes per-query candidate/rank/failure details.
It does not parse MGF.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import precursor_smiles_candidate_retrieval as retrieval


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMILES recall for precursor-mass candidate retrieval.")
    parser.add_argument("--input", required=True, help="Structured JSON records. Object/list/JSONL supported.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary and per-query outputs.")
    parser.add_argument("--top-k", type=int, default=50, help="Maximum candidates stored per query.")
    parser.add_argument("--ppm-tolerance", type=float, default=5.0, help="Mass tolerance used by rigid filtering.")
    parser.add_argument(
        "--absolute-da-reference",
        type=float,
        default=0.01,
        help="Reference Da window reported as ppm for diagnostics.",
    )
    parser.add_argument(
        "--allow-common-adducts",
        action="store_true",
        help="If no explicit adduct is found, evaluate all compatible common adducts.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional limit for fast debugging. 0 means all records.",
    )
    parser.add_argument(
        "--candidate-window-ppm",
        type=float,
        default=6.0,
        help="Mass index pre-window. Should be >= --ppm-tolerance.",
    )
    return parser.parse_args(argv)


def normalize_smiles_text(smiles: Any) -> str:
    return str(smiles or "").strip()


def is_same_target_smiles(candidate_payload: Dict[str, Any], target_smiles: str) -> bool:
    target = normalize_smiles_text(target_smiles)
    if not target:
        return False
    return normalize_smiles_text(candidate_payload.get("smiles")) == target


def build_candidate_library(records: Sequence[Dict[str, Any]]) -> Tuple[List[retrieval.Candidate], Dict[str, int]]:
    candidates: List[retrieval.Candidate] = []
    stats = {
        "records": len(records),
        "candidate_with_smiles": 0,
        "candidate_with_exact_mass": 0,
        "candidate_missing_exact_mass": 0,
        "candidate_stable_or_metastable": 0,
    }
    for idx, record in enumerate(records):
        row = dict(record)
        row.setdefault("database_id", retrieval.infer_record_id(record, idx))
        candidate = retrieval.normalize_candidate_row(row, "GNPS")
        candidates.append(candidate)
        if candidate.smiles:
            stats["candidate_with_smiles"] += 1
        if candidate.exact_mass is None:
            stats["candidate_missing_exact_mass"] += 1
        else:
            stats["candidate_with_exact_mass"] += 1
        if candidate.stability in {"stable", "metastable"}:
            stats["candidate_stable_or_metastable"] += 1
    return candidates, stats


def build_mass_index(candidates: Sequence[retrieval.Candidate]) -> Tuple[List[float], List[int]]:
    mass_index: List[Tuple[float, int]] = []
    for idx, candidate in enumerate(candidates):
        if candidate.exact_mass is None or not math.isfinite(candidate.exact_mass):
            continue
        mass_index.append((float(candidate.exact_mass), idx))
    mass_index.sort(key=lambda item: item[0])
    return [item[0] for item in mass_index], [item[1] for item in mass_index]


def fetch_mass_window_candidates(
    query: retrieval.QuerySpec,
    masses: Sequence[float],
    indices: Sequence[int],
    candidates: Sequence[retrieval.Candidate],
    candidate_window_ppm: float,
) -> List[retrieval.Candidate]:
    window_da = retrieval.ppm_to_da(query.neutral_mass, max(candidate_window_ppm, query.ppm_tolerance))
    left = bisect.bisect_left(masses, query.neutral_mass - window_da)
    right = bisect.bisect_right(masses, query.neutral_mass + window_da)
    return [candidates[indices[pos]] for pos in range(left, right)]


def rank_candidates_for_query(
    query: retrieval.QuerySpec,
    candidates: Sequence[retrieval.Candidate],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    accepted_payloads: List[Dict[str, Any]] = []
    rejection_counts: Dict[str, int] = {}
    for candidate in candidates:
        accepted, diagnostics = retrieval.filter_candidate(candidate, query)
        if not accepted:
            reason = str(diagnostics.get("reject_reason") or "unknown")
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        accepted_payloads.append(retrieval.candidate_to_payload(candidate, query, diagnostics))

    deduped = retrieval.dedupe_candidates(accepted_payloads)
    for rank, payload in enumerate(deduped, start=1):
        payload["rank"] = rank
    return deduped[:top_k], rejection_counts


def target_filter_diagnostic(target_candidate: retrieval.Candidate, query: retrieval.QuerySpec) -> Dict[str, Any]:
    accepted, diagnostics = retrieval.filter_candidate(target_candidate, query)
    return {
        "target_passes_rigid_filter": accepted,
        "target_filter_reason": diagnostics.get("reject_reason") or "pass",
        "target_theoretical_mz": diagnostics.get("theoretical_mz"),
        "target_ppm_error": diagnostics.get("ppm_error"),
        "target_abs_ppm_error": diagnostics.get("abs_ppm_error"),
        "target_exact_mass": target_candidate.exact_mass,
        "target_stability": target_candidate.stability,
        "target_stability_reason": target_candidate.stability_reason,
    }


def find_target_rank(candidates: Sequence[Dict[str, Any]], target_smiles: str) -> Optional[int]:
    for payload in candidates:
        if is_same_target_smiles(payload, target_smiles):
            return int(payload["rank"])
    return None


def classify_failure(
    record: Dict[str, Any],
    query: Optional[retrieval.QuerySpec],
    target_candidate: Optional[retrieval.Candidate],
    diagnostic: Optional[Dict[str, Any]],
    rank: Optional[int],
    full_candidate_count: int,
    top_k: int,
    query_error: str = "",
) -> str:
    if query_error:
        return "query_parse_error"
    if not normalize_smiles_text(record.get("smiles")):
        return "missing_target_smiles"
    if target_candidate is None:
        return "target_candidate_build_failed"
    if target_candidate.exact_mass is None:
        return "target_exact_mass_unavailable"
    if diagnostic and not diagnostic.get("target_passes_rigid_filter"):
        reason = diagnostic.get("target_filter_reason") or "rigid_filter_failed"
        return f"target_{reason}"
    if full_candidate_count == 0:
        return "no_candidates_after_mass_filter"
    if rank is None:
        return "target_removed_by_dedup_or_smiles_mismatch"
    if rank > top_k:
        return "target_rank_gt_top_k"
    return "hit"


def summarize_hits(rows: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Dict[str, Any]:
    total = len(rows)
    output: Dict[str, Any] = {"total": total}
    for k in ks:
        hits = sum(1 for row in rows if row.get("target_rank") is not None and int(row["target_rank"]) <= k)
        output[f"recall@{k}"] = hits / total if total else 0.0
        output[f"hits@{k}"] = hits
    miss_rows = [row for row in rows if row.get("target_rank") is None or int(row["target_rank"]) > max(ks)]
    output["miss_count"] = len(miss_rows)
    output["miss_rate"] = len(miss_rows) / total if total else 0.0
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    records = retrieval.read_input_records(args.input)
    if args.max_records and args.max_records > 0:
        records = records[: args.max_records]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates, library_stats = build_candidate_library(records)
    masses, mass_indices = build_mass_index(candidates)

    per_query_path = output_dir / "per_query_retrieval.jsonl"
    failures_path = output_dir / "failures.jsonl"
    summary_path = output_dir / "summary.json"

    rows_for_summary: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    query_error_count = 0
    total_candidate_pool_count = 0
    total_after_filter_count = 0

    with per_query_path.open("w", encoding="utf-8") as per_query_handle, failures_path.open(
        "w", encoding="utf-8"
    ) as failures_handle:
        for idx, record in enumerate(records):
            record_id = retrieval.infer_record_id(record, idx)
            target_smiles = normalize_smiles_text(record.get("smiles"))
            query: Optional[retrieval.QuerySpec] = None
            target_candidate: Optional[retrieval.Candidate] = candidates[idx] if idx < len(candidates) else None
            diagnostic: Optional[Dict[str, Any]] = None
            ranked_candidates: List[Dict[str, Any]] = []
            rejection_counts: Dict[str, int] = {}
            query_error = ""

            try:
                query_specs = retrieval.query_specs_from_record(
                    record=record,
                    record_index=idx,
                    ppm_tolerance=args.ppm_tolerance,
                    absolute_da_reference=args.absolute_da_reference,
                    allow_common_adducts=args.allow_common_adducts,
                )
                # For evaluation we keep the best adduct hypothesis for the target
                # SMILES. With default settings this is usually exactly one query.
                best_rank: Optional[int] = None
                best_payload: Optional[Dict[str, Any]] = None
                best_ranked: List[Dict[str, Any]] = []
                best_rejections: Dict[str, int] = {}
                best_diagnostic: Optional[Dict[str, Any]] = None
                best_query: Optional[retrieval.QuerySpec] = None
                best_pool_count = 0
                for current_query in query_specs:
                    pool = fetch_mass_window_candidates(
                        current_query,
                        masses,
                        mass_indices,
                        candidates,
                        candidate_window_ppm=args.candidate_window_ppm,
                    )
                    current_ranked, current_rejections = rank_candidates_for_query(
                        current_query,
                        pool,
                        top_k=max(args.top_k, 50),
                    )
                    current_rank = find_target_rank(current_ranked, target_smiles)
                    current_diag = (
                        target_filter_diagnostic(target_candidate, current_query) if target_candidate is not None else None
                    )
                    better = False
                    if best_query is None:
                        better = True
                    elif current_rank is not None and (best_rank is None or current_rank < best_rank):
                        better = True
                    elif current_rank is not None and best_rank is not None and current_rank == best_rank:
                        better = len(current_ranked) < len(best_ranked)
                    elif current_rank is None and best_rank is None and len(current_ranked) > len(best_ranked):
                        better = True
                    if better:
                        best_query = current_query
                        best_rank = current_rank
                        best_ranked = current_ranked
                        best_rejections = current_rejections
                        best_diagnostic = current_diag
                        best_pool_count = len(pool)
                        best_payload = current_ranked[0] if current_ranked else None

                query = best_query
                ranked_candidates = best_ranked
                rejection_counts = best_rejections
                diagnostic = best_diagnostic
                rank = best_rank
                candidate_pool_count = best_pool_count
            except Exception as exc:
                query_error = str(exc)
                query_error_count += 1
                rank = None
                candidate_pool_count = 0
                best_payload = None

            total_candidate_pool_count += candidate_pool_count
            total_after_filter_count += len(ranked_candidates)
            failure_reason = classify_failure(
                record=record,
                query=query,
                target_candidate=target_candidate,
                diagnostic=diagnostic,
                rank=rank,
                full_candidate_count=len(ranked_candidates),
                top_k=args.top_k,
                query_error=query_error,
            )

            output_row: Dict[str, Any] = {
                "record_index": idx,
                "record_id": record_id,
                "spectrumid": record.get("spectrumid"),
                "name": record.get("name"),
                "target_smiles": target_smiles,
                "query_error": query_error,
                "observed_mz": query.observed_mz if query else record.get("pepmass"),
                "charge": query.charge if query else record.get("charge"),
                "ion_mode": query.ion_mode if query else record.get("ionmode"),
                "assigned_adduct": query.adduct.name if query else None,
                "adduct_inference_source": query.adduct_inference_source if query else None,
                "neutral_mass": query.neutral_mass if query else None,
                "target_rank": rank,
                "hit_top_1": rank is not None and rank <= 1,
                "hit_top_3": rank is not None and rank <= 3,
                "hit_top_5": rank is not None and rank <= 5,
                "hit_top_10": rank is not None and rank <= 10,
                "hit_top_50": rank is not None and rank <= 50,
                "failure_reason": failure_reason,
                "candidate_pool_count_before_rigid_filter": candidate_pool_count,
                "candidate_count_returned": len(ranked_candidates),
                "rejection_counts": rejection_counts,
                "target_diagnostic": diagnostic,
                "top_candidates": ranked_candidates[: args.top_k],
            }
            per_query_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

            summary_row = {
                "record_index": idx,
                "record_id": record_id,
                "target_rank": rank,
                "failure_reason": failure_reason,
                "assigned_adduct": output_row["assigned_adduct"],
                "target_ppm_error": diagnostic.get("target_ppm_error") if diagnostic else None,
                "candidate_count_returned": len(ranked_candidates),
            }
            rows_for_summary.append(summary_row)
            if failure_reason != "hit":
                failure_rows.append(output_row)
                failures_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    ks = sorted({1, 3, 5, 10, 50, args.top_k})
    recall_summary = summarize_hits(rows_for_summary, ks)
    failure_counts: Dict[str, int] = {}
    adduct_counts: Dict[str, int] = {}
    for row in rows_for_summary:
        reason = str(row["failure_reason"])
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
        adduct = str(row.get("assigned_adduct"))
        adduct_counts[adduct] = adduct_counts.get(adduct, 0) + 1

    target_ppm_errors = [
        abs(float(row["target_ppm_error"]))
        for row in rows_for_summary
        if row.get("target_ppm_error") is not None and math.isfinite(float(row["target_ppm_error"]))
    ]
    summary = {
        "schema": "precursor_smiles_recall_evaluation.v1",
        "input": str(Path(args.input).expanduser()),
        "output_dir": str(output_dir),
        "configuration": {
            "top_k": args.top_k,
            "ppm_tolerance": args.ppm_tolerance,
            "candidate_window_ppm": args.candidate_window_ppm,
            "absolute_da_reference": args.absolute_da_reference,
            "allow_common_adducts": args.allow_common_adducts,
        },
        "library_stats": library_stats,
        "query_stats": {
            "records_evaluated": len(records),
            "query_error_count": query_error_count,
            "mean_candidate_pool_count_before_rigid_filter": total_candidate_pool_count / len(records)
            if records
            else 0.0,
            "mean_candidate_count_returned": total_after_filter_count / len(records) if records else 0.0,
            "adduct_counts": dict(sorted(adduct_counts.items(), key=lambda item: item[0])),
        },
        "recall": recall_summary,
        "failure_counts": dict(sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))),
        "target_abs_ppm_error_stats": {
            "count": len(target_ppm_errors),
            "mean": sum(target_ppm_errors) / len(target_ppm_errors) if target_ppm_errors else None,
            "max": max(target_ppm_errors) if target_ppm_errors else None,
            "over_tolerance_count": sum(1 for value in target_ppm_errors if value > args.ppm_tolerance),
        },
        "files": {
            "per_query_retrieval": str(per_query_path),
            "failures": str(failures_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
