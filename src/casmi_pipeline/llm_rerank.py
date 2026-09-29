from pathlib import Path
import statistics
from typing import Any, Dict, List, Sequence
import time
import requests
from tqdm import tqdm
from casmi_pipeline.io_utils import ensure_string_list, sanitize_short_text
from casmi_pipeline.llm_client import build_llm_messages, cached_llm_json
from casmi_pipeline.prompts import build_rerank_prompts
from casmi_pipeline.retrieval import summarize_hits


def normalize_choice_response(raw: Dict[str, Any], candidate_ids: Sequence[str], candidate_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    selected_record_id = sanitize_short_text(raw.get("selected_record_id"))
    if selected_record_id not in candidate_ids:
        selected_record_id = candidate_ids[0] if candidate_ids else "unknown"

    smiles_by_id = {
        row['candidate_id']: row['res_smile_ana']['analysis_row'].get("normalized_record", {}).get("smiles") or "unknown"
        for row in candidate_rows
    }
    selected_smiles = sanitize_short_text(smiles_by_id.get(selected_record_id))

    return {
        "selected_record_id": selected_record_id,
        "selected_smiles": selected_smiles,
        "confidence": sanitize_short_text(raw.get("confidence")),
        "decision_tags": ensure_string_list(raw.get("decision_tags"), max_items=6),
    }


def normalize_ranking_response(raw: Dict[str, Any], candidate_ids: Sequence[str]) -> Dict[str, Any]:
    ranked_ids = ensure_string_list(raw.get("ranked_record_ids"), max_items=len(candidate_ids))
    ordered: List[str] = []
    seen = set()
    for record_id in ranked_ids:
        if record_id in candidate_ids and record_id not in seen:
            ordered.append(record_id)
            seen.add(record_id)
    for record_id in candidate_ids:
        if record_id not in seen:
            ordered.append(record_id)
            seen.add(record_id)

    top_choice = sanitize_short_text(raw.get("top_choice_record_id"))
    if top_choice not in ordered:
        top_choice = ordered[0] if ordered else "unknown"

    return {
        "ranked_record_ids": ordered,
        "top_choice_record_id": top_choice,
        "confidence": sanitize_short_text(raw.get("confidence")),
        "ranking_tags": ensure_string_list(raw.get("ranking_tags"), max_items=6),
    }


def find_rank(record_id: str, ranked_ids: Sequence[str]) -> int:
    try:
        return list(ranked_ids).index(record_id) + 1
    except ValueError:
        return len(ranked_ids) + 1


def source_key_to_ids(ranking_row: Dict[str, Any], rerank_source: str) -> List[str]:
    if rerank_source == "mass_filtered":
        return list(ranking_row.get("filtered_ranked_record_ids", []))
    return list(ranking_row.get("raw_ranked_record_ids", []))


def evaluate_llm_rerank_for_source(
    session: requests.Session,
    cache_dir: Path,
    api_url: str,
    api_key: str,
    model: str,
    analysis_rows: Sequence[Dict[str, Any]],
    retrieval_results: Dict[str, Any],
    rerank_k: int,
    rerank_source: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    force: bool,
    topk: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    # rows_by_id = {row["record_id"]: row for row in analysis_rows}
    per_query_rankings = retrieval_results.get("per_query_rankings", [])

    choice_hits: List[int] = []
    ranking_ranks: List[int] = []
    pool_hits: List[int] = []
    per_query_results: List[Dict[str, Any]] = []

    for ranking_row, text_row in tqdm(zip(per_query_rankings, analysis_rows), desc='process1 ...'):
        st_time = time.time()
        query_record_id = ranking_row["query_record_id"]
        target_record_id = ranking_row["target_record_id"]
        source_ids = source_key_to_ids(ranking_row, rerank_source)
        candidate_ids = list(source_ids[:rerank_k])
        rows_by_id = {x['candidate_id']: x for x in text_row['retrieval_top50_res_smile_ana']}

        candidate_rows = [rows_by_id[record_id] for record_id in candidate_ids]
        if not candidate_rows:
            continue

        pool_hits.append(1 if target_record_id in candidate_ids else 0)
        query_row = text_row

        prompts = build_rerank_prompts(query_row, candidate_rows)

        choice_messages = build_llm_messages(
            system_prompt=prompts["choice_system_prompt"],
            user_prompt=prompts["choice_user_prompt"],
        )
        ranking_messages = build_llm_messages(
            system_prompt=prompts["ranking_system_prompt"],
            user_prompt=prompts["ranking_user_prompt"],
        )

        source_cache_dir = cache_dir / rerank_source
        choice_raw_json, choice_raw_content = cached_llm_json(
            session=session,
            cache_dir=source_cache_dir / "choice",
            api_url=api_url,
            api_key=api_key,
            model=model,
            messages=choice_messages,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            force=force,
        )
        ranking_raw_json, ranking_raw_content = cached_llm_json(
            session=session,
            cache_dir=source_cache_dir / "ranking",
            api_url=api_url,
            api_key=api_key,
            model=model,
            messages=ranking_messages,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            force=force,
        )

        choice_result = normalize_choice_response(choice_raw_json, candidate_ids, candidate_rows)
        ranking_result = normalize_ranking_response(ranking_raw_json, candidate_ids)

        choice_correct = int(choice_result["selected_record_id"] == target_record_id)
        target_rank = find_rank(target_record_id, ranking_result["ranked_record_ids"])
        choice_hits.append(choice_correct)
        ranking_ranks.append(target_rank)

        per_query_results.append(
            {
                "query_record_id": query_record_id,
                "target_record_id": target_record_id,
                "rerank_source": rerank_source,
                "candidate_record_ids": candidate_ids,
                "target_in_candidate_pool": bool(target_record_id in candidate_ids),
                "choice_result": choice_result,
                "ranking_result": ranking_result,
                "llm_choice_correct": bool(choice_correct),
                "llm_target_rank": target_rank,
                "choice_raw_content": choice_raw_content,
                "ranking_raw_content": ranking_raw_content,
            }
        )

        print(f'time: {time.time() - st_time}')

    choice_accuracy = float(sum(choice_hits) / len(choice_hits)) if choice_hits else 0.0
    ranking_metrics = summarize_hits(ranking_ranks, topk)
    pool_oracle = float(sum(pool_hits) / len(pool_hits)) if pool_hits else 0.0

    return {
        "config": {
            "rerank_model": model,
            "rerank_k": rerank_k,
            "rerank_source": rerank_source,
        },
        "candidate_pool_oracle_accuracy": pool_oracle,
        "choice_top1_accuracy": choice_accuracy,
        "ranking_metrics": ranking_metrics,
        "rank_stats": {
            "mean_llm_rank": float(sum(ranking_ranks) / len(ranking_ranks)) if ranking_ranks else 0.0,
            "median_llm_rank": float(statistics.median(ranking_ranks)) if ranking_ranks else 0.0,
        },
        "per_query_results": per_query_results,
    }


def evaluate_llm_rerank(
    session: requests.Session,
    cache_dir: Path,
    api_url: str,
    api_key: str,
    model: str,
    analysis_rows: Sequence[Dict[str, Any]],
    retrieval_results: Dict[str, Any],
    rerank_k: int,
    rerank_source: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    force: bool,
    topk: Sequence[int] = (1, 3, 5, 10, 20, 50),
) -> Dict[str, Any]:
    if rerank_source == "both":
        sources = ["embedding_only", "mass_filtered"]
    else:
        sources = [rerank_source]

    results: Dict[str, Any] = {}
    for source in sources:
        print("source ...", source)
        results[source] = evaluate_llm_rerank_for_source(
            session=session,
            cache_dir=cache_dir,
            api_url=api_url,
            api_key=api_key,
            model=model,
            analysis_rows=analysis_rows,
            retrieval_results=retrieval_results,
            rerank_k=rerank_k,
            rerank_source=source,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            force=force,
            topk=topk,
        )

    return results
