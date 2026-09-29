import argparse
import concurrent.futures
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import requests
from tqdm import tqdm
import json
from casmi_pipeline.embedding_utils import embed_texts, load_embeddings, save_embeddings
from casmi_pipeline.images import get_structure_image
from casmi_pipeline.io_utils import read_json_array, read_jsonl, write_json, write_jsonl
from casmi_pipeline.llm_client import build_llm_messages, cached_llm_json
from casmi_pipeline.llm_rerank import evaluate_llm_rerank
from casmi_pipeline.preprocess import infer_ms_mass_anchor, normalize_input_record, summarize_peaks
from casmi_pipeline.prompts import build_ms_prompt_input, build_prompts, build_structure_prompt_input
from casmi_pipeline.retrieval import evaluate_retrieval
from casmi_pipeline.settings import (
    CHAT_API_URL,
    DEFAULT_IMAGE_DIR,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    PROMPT_VERSION,
)
from casmi_pipeline.structure_features import build_structure_features
from casmi_pipeline.summaries import (
    make_combined_summary_text,
    make_ms_summary_text,
    make_structure_summary_text,
    normalize_ms_analysis,
    normalize_structure_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate retrieval-oriented MS and structure signatures, then build embeddings and evaluate retrieval."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSON array path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Image directory.")
    parser.add_argument("--api-url", default=CHAT_API_URL, help="Chat completion API URL.")
    parser.add_argument(
        "--api-key",
        default="",
        help="API key. If omitted, COMMONSTACK_API_KEY from environment will be used.",
    )
    parser.add_argument("--ms-model", default="Qwen3.5-0.8B-GRPO", help="Instrument prompt model.")
    parser.add_argument(
        "--structure-model",
        default="Qwen3.5-0.8B-GRPO",
        help="SMILES + image prompt model.",
    )
    parser.add_argument("--embedding-model", default="/data3/xj/pre_model/Qwen3-Embedding-8B", help="Embedding model.")
    parser.add_argument("--top-k-peaks", type=int, default=25, help="Top peaks to include.")
    parser.add_argument("--temperature", type=float, default=0.9, help="LLM temperature.")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries for chat completion.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Delay between LLM calls.")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Number of worker processes for per-record preprocessing and LLM generation.",
    )
    parser.add_argument(
        "--worker-backend",
        default="auto",
        choices=["auto", "process", "thread", "sequential"],
        help="Parallel backend for per-record processing. auto prefers process and falls back to thread.",
    )
    parser.add_argument("--mass-filter-tolerance", type=float, default=0.05, help="Mass filter tolerance in Da.")
    parser.add_argument("--llm-rerank-k", type=int, default=20, help="If > 0, ask LLM to rerank the top-k candidates.")
    parser.add_argument(
        "--llm-rerank-source",
        default="both",
        choices=["embedding_only", "mass_filtered", "both"],
        help="Candidate source for LLM reranking.",
    )
    parser.add_argument("--rerank-model", default="", help="Optional LLM model for reranking. Defaults to ms-model.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit.")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM and embedding calls.")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding and similarity.")
    parser.add_argument("--force", action="store_true", help="Ignore cache where possible.")
    return parser.parse_args()


def resolve_api_key(args: argparse.Namespace) -> str:
    return os.environ.get("COMMONSTACK_API_KEY", "") or args.api_key


def load_run_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def expected_record_count(input_path: Path, limit: int | None) -> int:
    records = read_json_array(input_path)
    if limit is None:
        return len(records)
    return min(len(records), limit)


def can_reuse_analysis_cache(args: argparse.Namespace, input_path: Path, output_dir: Path) -> bool:
    if args.force or args.dry_run:
        return False
    analyses_path = output_dir / "analyses.jsonl"
    run_summary_path = output_dir / "run_summary.json"
    if not analyses_path.exists() or not run_summary_path.exists():
        return False
    run_summary = load_run_summary(run_summary_path)
    expected_count = expected_record_count(input_path, args.limit)
    return (
        run_summary.get("prompt_version") == PROMPT_VERSION
        and run_summary.get("input_path") == str(input_path)
        and run_summary.get("ms_model") == args.ms_model
        and run_summary.get("structure_model") == args.structure_model
        # and int(run_summary.get("record_count", -1)) == expected_count
    )


def embedding_cache_paths(output_dir: Path, embedding_model: str) -> Dict[str, Path]:
    filename = os.path.basename(str(embedding_model).rstrip("/")) or "embedding_model"
    return {
        "ms": output_dir / f"ms_embeddings_{filename}.npy",
        "structure": output_dir / f"structure_embeddings_{filename}.npy",
        "combined": output_dir / f"combined_embeddings_{filename}.npy",
    }


def embeddings_match_record_count(paths: Dict[str, Path], record_count: int) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        for path in paths.values():
            matrix = load_embeddings(path)
            # if getattr(matrix, "shape", (0,))[0] != record_count:
            #     return False
    except Exception:
        return False
    return True


def process_record_task(task: Dict[str, Any]) -> Dict[str, Any]:

    try:

        idx = task["idx"]
        raw_record = task["raw_record"]
        session = requests.Session()

        record = normalize_input_record(raw_record, idx)
        peak_summary = summarize_peaks(record["peaks"], record["parent_mz"], task["top_k_peaks"])
        ms_mass_anchor = infer_ms_mass_anchor(record["parent_mz"], record["ion_mode"], record["charge"])
        image_meta = get_structure_image(
            session=session,
            smiles=record["smiles"],
            image_dir=Path(task["image_dir"]),
            timeout=task["timeout"],
            force=task["force"],
        )
        structure_features = build_structure_features(
            session=session,
            smiles=record["smiles"],
            image_meta=image_meta,
            timeout=task["timeout"],
            cache_dir=Path(task["cache_dir"]) / "structure_properties",
            force=task["force"],
        )
        prompts = build_prompts(record, peak_summary, ms_mass_anchor, image_meta, structure_features)

        image_row = {"record_id": record["record_id"], "smiles": record["smiles"], **image_meta}
        prompt_row = {
            "record_id": record["record_id"],
            "ms_system_prompt": prompts["ms_system_prompt"],
            "ms_user_prompt": prompts["ms_user_prompt"],
            "structure_system_prompt": prompts["structure_system_prompt"],
            "structure_user_prompt": prompts["structure_user_prompt"],
        }

        if task["dry_run"]:
            analysis_row = {
                "record_id": record["record_id"],
                "normalized_record": record,
                "peak_summary": peak_summary,
                "ms_mass_anchor": ms_mass_anchor,
                "image_meta": image_meta,
                "structure_features": structure_features,
                "ms_prompt_input": build_ms_prompt_input(record, peak_summary, ms_mass_anchor),
                "structure_prompt_input": build_structure_prompt_input(record, image_meta, structure_features),
            }
            return {
                "idx": idx,
                "image_row": image_row,
                "prompt_row": prompt_row,
                "analysis_row": analysis_row,
            }

        api_key = task["api_key"]
        if not api_key:
            raise ValueError("API key is required unless --dry-run is enabled.")

        if raw_record['ms'] is not None:
            ms_messages = build_llm_messages(
                system_prompt=prompts["ms_system_prompt"],
                user_prompt=prompts["ms_user_prompt"],
            )
        else:
            ms_messages = None

        structure_image_path = Path(image_meta["image_path"]) if image_meta["image_available"] else None

        if raw_record['smiles'] is not None:
            structure_messages = build_llm_messages(
                system_prompt=prompts["structure_system_prompt"],
                user_prompt=prompts["structure_user_prompt"],
                image_path=structure_image_path,
            )
        else:
            structure_messages = None

        if raw_record['ms'] is not None:
            ms_raw_json, ms_raw_content = cached_llm_json(
                session=session,
                cache_dir=Path(task["cache_dir"]) / "instrument",
                api_url=task["api_url"],
                api_key=api_key,
                model=task["ms_model"],
                messages=ms_messages,
                temperature=task["temperature"],
                timeout=task["timeout"],
                max_retries=task["max_retries"],
                force=task["force"],
            )
        else:
            ms_raw_json, ms_raw_content = None, None

        if raw_record['smiles'] is not None:
            structure_raw_json, structure_raw_content = cached_llm_json(
                session=session,
                cache_dir=Path(task["cache_dir"]) / "structure",
                api_url=task["api_url"],
                api_key=api_key,
                model=task["structure_model"],
                messages=structure_messages,
                temperature=task["temperature"],
                timeout=task["timeout"],
                max_retries=task["max_retries"],
                force=task["force"],
            )
        else:
            structure_raw_json, structure_raw_content = None, None

        if raw_record['ms'] is not None:
            ms_analysis = normalize_ms_analysis(ms_raw_json)
            ms_summary_text = make_ms_summary_text(record, peak_summary, ms_mass_anchor, ms_analysis)
        else:
            ms_analysis = None
            ms_summary_text = None

        if raw_record['smiles'] is not None:
            structure_analysis = normalize_structure_analysis(structure_raw_json)
            structure_summary_text = make_structure_summary_text(record, structure_features, structure_analysis)

        else:
            structure_analysis = None
            structure_summary_text = None

        # combined_summary_text = make_combined_summary_text(ms_summary_text, structure_summary_text)
        combined_summary_text = None

        analysis_row = {
            "record_id": record["record_id"],
            "source_fn": record["source_fn"],
            "normalized_record": record,
            "peak_summary": peak_summary,
            "ms_mass_anchor": ms_mass_anchor,
            "image_meta": image_meta,
            "structure_features": structure_features,
            "ms_analysis": ms_analysis,
            "structure_analysis": structure_analysis,
            "ms_summary_text": ms_summary_text,
            "structure_summary_text": structure_summary_text,
            "combined_summary_text": combined_summary_text,
            "ms_raw_content": ms_raw_content,
            "structure_raw_content": structure_raw_content,
        }

        if task["sleep_seconds"] > 0:
            time.sleep(task["sleep_seconds"])

        return {
            "idx": idx,
            "image_row": image_row,
            "prompt_row": prompt_row,
            "analysis_row": analysis_row,
        }

    except:
        print("error ...", task["idx"])
        return None

def run_tasks_with_executor(
    tasks: List[Dict[str, Any]],
    worker_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    num_workers: int,
    desc: str,
    worker_backend: str,
) -> Tuple[List[Dict[str, Any]], str]:
    if num_workers <= 1 or worker_backend == "sequential":
        return [worker_fn(task) for task in tqdm(tasks, total=len(tasks), desc=desc)], "sequential"

    if worker_backend in ("auto", "process"):
        try:
            ctx = mp.get_context("spawn")
            results: List[Dict[str, Any]] = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                future_to_idx = {executor.submit(worker_fn, task): task["idx"] for task in tasks}
                for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(tasks), desc=desc):
                    results.append(future.result())
            return results, "process"
        except (PermissionError, OSError) as exc:
            if worker_backend == "process":
                raise
            print(f"process backend unavailable ({exc}); falling back to thread execution.")

    if worker_backend in ("auto", "thread"):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(worker_fn, task): task["idx"] for task in tasks}
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(tasks), desc=desc):
                results.append(future.result())
        return results, "thread"

    return [worker_fn(task) for task in tqdm(tasks, total=len(tasks), desc=desc)], "sequential"


def process_records(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    cache_dir: Path,
    image_dir: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    res_ms_records = []
    res_smile_records = []
    res_ms_ids = []
    res_smile_ids = []
    for i, d in enumerate(records):
        if d['spectrumid'] not in res_ms_ids:
            res_ms_ids.append(d['spectrumid'])
            res_ms_records.append({
                "ms": d['mz_lists'],
                "parent_mz": d['pepmass'],
                "scan": d['scans'],
                "ms_level": d['mslevel'],
                "Ion_Mode": d['ionmode'],
                "Ion_Source": "LC-ESI",
                "Instrument": d['source_instrument'],
                "Charge": d['charge'],
                "smiles": None
            })

        for x in d['retrieval_top50']:
            if x['candidate_id'] not in res_smile_ids:
                res_smile_ids.append(x['candidate_id'])
                res_smile_records.append({
                    "ms": None,
                    "parent_mz": x['exact_mass'],
                    "scan": d['scans'],
                    "ms_level": d['mslevel'],
                    "Ion_Mode": d['ionmode'],
                    "Ion_Source": "LC-ESI",
                    "Instrument": d['source_instrument'],
                    "Charge": d['charge'],
                    "smiles": x['smiles']
                })

    t_records = res_ms_records + res_smile_records

    tasks = [
        {
            "idx": idx,
            "raw_record": raw_record,
            "top_k_peaks": args.top_k_peaks,
            "timeout": args.timeout,
            "force": args.force,
            "dry_run": args.dry_run,
            "api_key": resolve_api_key(args),
            "api_url": args.api_url,
            "ms_model": args.ms_model,
            "structure_model": args.structure_model,
            "temperature": args.temperature,
            "max_retries": args.max_retries,
            "sleep_seconds": args.sleep_seconds,
            "image_dir": str(image_dir),
            "cache_dir": str(cache_dir),
        }
        for idx, raw_record in enumerate(t_records)
    ]

    results, effective_backend = run_tasks_with_executor(
        tasks=tasks,
        worker_fn=process_record_task,
        num_workers=args.num_workers,
        desc="process ...",
        worker_backend=args.worker_backend,
    )

    results.sort(key=lambda item: item["idx"])

    assert len(results) == len(t_records), 'error'

    res0001 = results[:len(res_ms_records)]
    res0002 = results[len(res_ms_records):]

    t_res0001 = {x1: x2 for x1, x2 in zip(res_ms_ids, res0001)}
    t_res0002 = {x1: x2 for x1, x2 in zip(res_smile_ids, res0002)}

    res000 = []
    for i, d in enumerate(records):
        d["res_ms_ana"] = t_res0001[d['spectrumid']]

        ttt000 = []
        for x in d['retrieval_top50']:
            x["res_smile_ana"] = t_res0002[x['candidate_id']]
            ttt000.append(x)
        d['retrieval_top50_res_smile_ana'] = ttt000
        res000.append(d)

    image_rows = [item["image_row"] for item in results]
    prompt_rows = [item["prompt_row"] for item in results]
    analysis_rows = [item["analysis_row"] for item in results]
    return res000, image_rows, prompt_rows, analysis_rows, effective_backend


def run_pipeline(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)
    cache_dir = output_dir / "cache"
    analyses_path = output_dir / "analyses.jsonl"
    res_research_analyses_path = output_dir / "res_research_analyses.jsonl"

    if can_reuse_analysis_cache(args, input_path, output_dir):
        analysis_rows = read_jsonl(analyses_path)
        res000 = read_jsonl(res_research_analyses_path)
    else:
        records = read_json_array(input_path)
        if args.limit is not None:
            records = records[: args.limit]
        res000, image_rows, prompt_rows, analysis_rows, effective_backend = process_records(
            records,
            args,
            cache_dir=cache_dir,
            image_dir=image_dir,
        )

        write_jsonl(output_dir / "image_manifest.jsonl", image_rows)
        write_jsonl(output_dir / "prompts.jsonl", prompt_rows)

        if args.dry_run:
            write_jsonl(output_dir / "dry_run_inputs.jsonl", analysis_rows)
            write_json(
                output_dir / "run_summary.json",
                {
                    "input_path": str(input_path),
                    "record_count": len(analysis_rows),
                    "limit": args.limit,
                    "prompt_version": PROMPT_VERSION,
                    "mode": "dry_run",
                    "num_workers": args.num_workers,
                    "worker_backend": args.worker_backend,
                    "effective_worker_backend": effective_backend,
                },
            )
            return

        write_jsonl(analyses_path, analysis_rows)
        write_jsonl(res_research_analyses_path, res000)

        write_json(
            output_dir / "run_summary.json",
            {
                "input_path": str(input_path),
                "record_count": len(res000),
                "image_dir": str(image_dir),
                "ms_model": args.ms_model,
                "structure_model": args.structure_model,
                "embedding_model": args.embedding_model,
                "limit": args.limit,
                "prompt_version": PROMPT_VERSION,
                "num_workers": args.num_workers,
                "worker_backend": args.worker_backend,
                "effective_worker_backend": effective_backend,
            },
        )

    if args.skip_embedding:
        return

    # ms_texts = [row["ms_summary_text"] for row in analysis_rows]
    # structure_texts = [row["structure_summary_text"] for row in analysis_rows]
    # combined_texts = [row["combined_summary_text"] for row in analysis_rows]

    ms_texts = []
    structure_texts = []
    structure_texts_ids = []
    combined_texts = []
    for line in res000:
        ms_texts.append(line['res_ms_ana']['analysis_row']['ms_summary_text'])

        for x in line['retrieval_top50_res_smile_ana']:
            x11 = x['res_smile_ana']['analysis_row']['structure_summary_text']
            x12 = x['candidate_id']

            if x12 not in structure_texts_ids:
                structure_texts.append(x11)
                structure_texts_ids.append(x12)

    paths = embedding_cache_paths(output_dir, args.embedding_model)

    if args.force or not embeddings_match_record_count(paths, len(analysis_rows)):
        embedding_results = embed_texts(
            embedding_model=args.embedding_model,
            ms_texts=ms_texts,
            structure_texts=structure_texts,
            combined_texts=combined_texts,
            batch_size=args.batch_size,
        )
        save_embeddings(paths["ms"], embedding_results["ms_embeddings"])
        save_embeddings(paths["structure"], embedding_results["structure_embeddings"])
        save_embeddings(paths["combined"], embedding_results["combined_embeddings"])
    else:
        embedding_results = {
            "ms_embeddings": load_embeddings(paths["ms"]),
            "structure_embeddings": load_embeddings(paths["structure"]),
            "combined_embeddings": load_embeddings(paths["combined"]),
        }

    res_index_data = json.load(open(output_dir / "res_index_20.json", 'r', encoding='utf-8'))

    retrieval_results = evaluate_retrieval(
        ms_embeddings=embedding_results["ms_embeddings"],
        structure_embeddings=embedding_results["structure_embeddings"],
        analysis_rows=res000,
        mass_tolerance=args.mass_filter_tolerance,
        structure_texts_ids=structure_texts_ids,
        res_index_data=res_index_data,
    )

    if args.llm_rerank_k > 0 and not args.dry_run:
        session = requests.Session()
        api_key = resolve_api_key(args)
        rerank_model = args.rerank_model or args.ms_model
        llm_rerank_results = evaluate_llm_rerank(
            session=session,
            # cache_dir=output_dir / "cache" / "rerank",
            cache_dir=output_dir / "cache" / f"rerank_new_bert_{args.llm_rerank_k}_new",
            api_url=args.api_url,
            api_key=api_key,
            model=rerank_model,
            analysis_rows=res000,
            retrieval_results=retrieval_results,
            rerank_k=args.llm_rerank_k,
            rerank_source=args.llm_rerank_source,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            force=args.force,
        )
        retrieval_results["llm_rerank"] = llm_rerank_results

    # write_json(output_dir / "retrieval_results.json", retrieval_results)
    write_json(output_dir / f"retrieval_results_new_bert_{args.llm_rerank_k}_new.json", retrieval_results)
    print("retrieval_results ...", "ok ! ")


def main() -> None:
    args = parse_args()

    args.ms_model = ["openai/gpt-5.4-mini-2026-03-17", "qwen/qwen3.5-397b-a17b",
                     "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro",
                     "openai/gpt-5.4-nano-2026-03-17", "openai/gpt-5.4-2026-03-05",
                     "qwen/qwen3.5-27b", "qwen/qwen3.5-9b",
                     'deepseek-v4-flash', 'deepseek-v4-pro',
                     "qwen3.5-397b-a17b", "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b", "qwen3.5-4b",
                     "qwen3.5-4b-sft"][-2]
    args.structure_model = args.ms_model

    args.output_dir = args.output_dir + f"_{args.ms_model.replace('/', '_')}_v6"

    # args.skip_embedding = True
    args.llm_rerank_source = "embedding_only"

    args.embedding_model = ["/data3/xj/pre_model/Qwen3-Embedding-8B",
                            "/data3/xj/pre_model/Qwen3-Embedding-0.6B",
                            '/Users/xiaojie/Documents/预训练模型/qwen/Qwen3-Embedding-0.6B',
                            "/root/autodl-tmp/model/Qwen3-Embedding-0.6B"][-1]

    args.llm_rerank_k = 20
    args.num_workers = 1

    run_pipeline(args)
