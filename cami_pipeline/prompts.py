import json
from typing import Any, Dict, List, Sequence


MS_SYSTEM_PROMPT = """你是一位质谱检索系统的特征标注专家。
你的任务不是写分析报告，而是把仪器结果转换成稳定、短、可比较的检索签名。

重要约束：
1. 你只能使用输入中的仪器字段和 peak_summary。
2. 你不能引用 SMILES、结构图片、数据库条目或具体化合物名称。
3. 如果证据不足，请输出 unknown，不要编造。
4. 不要输出解释性长文本，不要输出 Markdown。
5. 只返回 JSON。
6. 优先保留能帮助结构检索的证据：质量锚点、代表性碎片、中性丢失、谱图稀疏度。

请严格返回如下 JSON：
{
  "analysis_type": "ms_retrieval_signature",
  "mass_anchor": {
    "likely_ion_forms": ["string"],
    "neutral_mass_hypotheses": [0.0]
  },
  "spectrum_profile": {
    "precursor_dominance": "very_high|high|medium|low|unknown",
    "fragmentation_richness": "very_low|low|medium|high|unknown",
    "top_fragments_mz": [0.0],
    "neutral_losses": [0.0]
  },
  "chemical_hints": ["string"],
  "retrieval_terms": ["string"]
}
"""


MS_USER_PROMPT_TEMPLATE = """请基于以下输入生成质谱检索签名。

输入：
{input_json}

要求：
1. mass_anchor 里的 likely_ion_forms 和 neutral_mass_hypotheses 只能基于输入已有信息做保守表达。
2. spectrum_profile 重点保留对结构检索最有用的峰模式信息，不要复述常量字段。
3. chemical_hints 只保留短标签，例如 acidic_molecule、sulfur_containing_loss、halogenated_pattern、polyfluoro_pattern。
4. retrieval_terms 只保留 3-6 个短语，尽量与结构侧的 scaffold、functional_groups、likely_losses 对齐。
5. 输出必须是合法 JSON。
"""


STRUCTURE_SYSTEM_PROMPT = """你是一位化学结构检索系统的特征标注专家。
你的任务不是写化学综述，而是把 SMILES 和 2D 结构图像转换成稳定、短、可比较的检索签名。

重要约束：
1. 你只能使用输入中的 SMILES、结构图像和由 SMILES 导出的确定性结构特征。
2. 你不能引用质谱实验字段或外部数据库中的具体实验结论。
3. 先做一致性校验，再输出适合检索的短标签。
4. 不要输出解释性长文本，不要输出 Markdown。
5. 只返回 JSON。
6. 优先保留能帮助质谱匹配的结构信息：质量锚点、官能团、酸碱性、易碎键、典型中性丢失。

请严格返回如下 JSON：
{
  "analysis_type": "structure_retrieval_signature",
  "consistency_check": "consistent|partially_consistent|uncertain|inconsistent",
  "structure_profile": {
    "scaffold": "string",
    "ring_system": "string",
    "heteroatom_signature": "string",
    "functional_groups": ["string"],
    "acid_base_class": "strong_acid|weak_acid|neutral|basic|zwitterionic|unknown",
    "shape_class": "fused_aromatic|planar_aromatic|flexible_aliphatic|mixed|unknown"
  },
  "fragmentation_priors": {
    "likely_losses": ["string"],
    "fragile_bonds": ["string"]
  },
  "retrieval_terms": ["string"]
}
"""


STRUCTURE_USER_PROMPT_TEMPLATE = """请基于以下输入生成结构检索签名。

输入：
{input_json}

要求：
1. exact_mass、molecular_formula、hbd、hba、xlogp_class、adduct_compatibility 都以输入给定的确定性特征为准，不要改写成其他数值。
2. scaffold、ring_system、functional_groups、fragile_bonds 都只保留短标签或短短语。
3. acid_base_class 要优先服务于检索，不要写成长解释。
4. likely_losses 只保留常见中性丢失或离去基标签，例如 H2O、CO2、SO3、HF。
5. retrieval_terms 只保留 3-6 个短语，尽量与 MS 侧的 neutral_losses、chemical_hints、retrieval_terms 对齐。
6. 输出必须是合法 JSON。
"""


RERANK_CHOICE_SYSTEM_PROMPT = """你是一位质谱到结构检索系统的重排专家。
你的任务是在给定的候选结构中，只选择一个最可能与查询质谱匹配的候选。

判断优先级：
1. 精确质量与可能 adduct 的一致性
2. 中性丢失与代表性碎片的一致性
3. 酸碱性、异原子组成、官能团与离子模式的一致性
4. 易碎键与 likely_losses 的一致性
5. 广义 scaffold 或化学家族相似性

重要约束：
1. 只能从给定候选中选择，不要输出候选列表之外的 candidate_id。
2. 不要编造数据库知识。
3. 如果证据不完整，也必须选择证据最强的一个候选。
4. 只返回 JSON。

请严格返回如下 JSON：
{
  "selected_candidate_id": "string",
  "selected_smiles": "string",
  "confidence": "low|medium|high",
  "decision_tags": ["string"]
}
"""


RERANK_CHOICE_USER_PROMPT_TEMPLATE = """请根据查询质谱签名与候选结构列表，选择最可能的一个匹配结构。

输入：
{input_json}

要求：
1. 优先看质量与 adduct 是否匹配，其次看 neutral_losses 与 likely_losses 是否匹配。
2. 不要只依据 scaffold 宽泛相似性做选择。
3. selected_candidate_id 必须来自 candidate_ids。
4. decision_tags 只保留 2-5 个短标签，例如 mass_match、co2_loss_match、acidic_compatible、halogen_pattern。
5. 输出必须是合法 JSON。
"""


RERANK_RANKING_SYSTEM_PROMPT = """你是一位质谱到结构检索系统的重排专家。
你的任务是对给定候选结构进行完整排序，从最可能到最不可能。

判断优先级：
1. 精确质量与 adduct 一致性
2. 中性丢失与碎片模式一致性
3. 官能团、酸碱性、异原子与离子模式一致性
4. 易碎键与 likely_losses 一致性
5. 宽义化学家族相似性

重要约束：
1. ranked_candidate_ids 必须是给定 candidate_id 的一个完整排列，不能重复，不能缺失。
2. 不要输出候选之外的 candidate_id。
3. 只返回 JSON。

请严格返回如下 JSON：
{
  "ranked_candidate_ids": ["string"],
  "top_choice_candidate_id": "string",
  "confidence": "low|medium|high",
  "ranking_tags": ["string"]
}
"""


RERANK_RANKING_USER_PROMPT_TEMPLATE = """请根据查询质谱签名与候选结构列表，对候选进行完整排序。

输入：
{input_json}

要求：
1. ranked_candidate_ids 必须包含全部 candidate_ids，且顺序为从最可能到最不可能。
2. 优先利用质量与中性丢失信息，再参考结构标签。
3. ranking_tags 只保留 2-6 个短标签，例如 mass_consistent、so3_loss_match、polyfluoro_match、acidic_negative_mode。
4. 输出必须是合法 JSON。
"""


def build_ms_prompt_input(
    record: Dict[str, Any],
    peak_summary: Dict[str, Any],
    ms_mass_anchor: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "parent_mz": record["parent_mz"],
        "charge": record["charge"],
        "observed_adduct": record.get("adduct") or "unknown",
        "ms_level": record["ms_level"],
        "ion_mode": record["ion_mode"] or "unknown",
        "ion_source": record["ion_source"] or "unknown",
        "instrument": record["instrument"] or "unknown",
        "scan": record["scan"],
        "peak_summary": {
            "peak_count": peak_summary["peak_count"],
            "fragment_peak_count": peak_summary["fragment_peak_count"],
            "precursor_dominance": peak_summary["precursor_dominance"],
            "fragmentation_richness": peak_summary["fragmentation_richness"],
            "base_peak": peak_summary["base_peak"],
            "precursor_peak": peak_summary["precursor_peak"],
            "top_fragment_peaks_by_intensity": peak_summary["top_fragment_peaks"],
            "neutral_losses": peak_summary["neutral_losses"],
        },
        "mass_anchor": ms_mass_anchor,
    }


def build_structure_prompt_input(
    record: Dict[str, Any],
    image_meta: Dict[str, Any],
    structure_features: Dict[str, Any],
) -> Dict[str, Any]:
    derived_features = {
        "molecular_formula": structure_features["molecular_formula"],
        "exact_mass": structure_features["exact_mass"],
        "heteroatom_signature": structure_features["heteroatom_signature"],
        "hbd": structure_features["hbd"],
        "hba": structure_features["hba"],
        "xlogp_class": structure_features["xlogp_class"],
        "adduct_compatibility": structure_features["adduct_compatibility"],
    }
    feature_source = structure_features.get("feature_source")
    if feature_source and feature_source != "rdkit":
        derived_features["feature_source"] = feature_source

    result = {
        "smiles": record["smiles"] or "unknown",
        "image_status": "available" if image_meta["image_available"] else "unavailable",
        "image_source": image_meta["image_source"],
        "derived_structure_features": derived_features,
    }
    image_warning = image_meta.get("structure_image_warning")
    if image_warning:
        result["image_warning"] = image_warning
    return result


def build_prompts(
    record: Dict[str, Any],
    peak_summary: Dict[str, Any],
    ms_mass_anchor: Dict[str, Any],
    image_meta: Dict[str, Any],
    structure_features: Dict[str, Any],
) -> Dict[str, str]:
    return {
        **build_ms_prompts(record, peak_summary, ms_mass_anchor),
        **build_structure_prompts(record, image_meta, structure_features),
    }


def build_ms_prompts(
    record: Dict[str, Any],
    peak_summary: Dict[str, Any],
    ms_mass_anchor: Dict[str, Any],
) -> Dict[str, str]:
    ms_prompt_input = build_ms_prompt_input(record, peak_summary, ms_mass_anchor)
    return {
        "ms_system_prompt": MS_SYSTEM_PROMPT,
        "ms_user_prompt": MS_USER_PROMPT_TEMPLATE.format(
            input_json=json.dumps(ms_prompt_input, ensure_ascii=False, indent=2)
        ),
    }


def build_structure_prompts(
    record: Dict[str, Any],
    image_meta: Dict[str, Any],
    structure_features: Dict[str, Any],
) -> Dict[str, str]:
    structure_prompt_input = build_structure_prompt_input(record, image_meta, structure_features)
    return {
        "structure_system_prompt": STRUCTURE_SYSTEM_PROMPT,
        "structure_user_prompt": STRUCTURE_USER_PROMPT_TEMPLATE.format(
            input_json=json.dumps(structure_prompt_input, ensure_ascii=False, indent=2)
        ),
    }


def build_rerank_query_input(query_row: Dict[str, Any]) -> Dict[str, Any]:
    ms_analysis = query_row.get("ms_analysis", {})
    peak_summary = query_row.get("peak_summary", {})
    normalized_record = query_row.get("normalized_record", {})
    return {
        "precursor_mz": normalized_record.get("parent_mz"),
        "charge": normalized_record.get("charge"),
        "ion_mode": normalized_record.get("ion_mode"),
        "observed_adduct": normalized_record.get("adduct") or "unknown",
        "likely_ion_forms": ms_analysis.get("likely_ion_forms") or query_row.get("ms_mass_anchor", {}).get("likely_ion_forms"),
        "neutral_mass_hypotheses": ms_analysis.get("neutral_mass_hypotheses") or query_row.get("ms_mass_anchor", {}).get("neutral_mass_hypotheses"),
        "peak_count": peak_summary.get("peak_count"),
        "fragment_peak_count": peak_summary.get("fragment_peak_count"),
        "precursor_dominance": ms_analysis.get("precursor_dominance") or peak_summary.get("precursor_dominance"),
        "fragmentation_richness": ms_analysis.get("fragmentation_richness") or peak_summary.get("fragmentation_richness"),
        "top_fragments_mz": ms_analysis.get("top_fragments_mz") or [peak.get("mz") for peak in peak_summary.get("top_fragment_peaks", [])],
        "neutral_losses": ms_analysis.get("neutral_losses") or peak_summary.get("neutral_losses"),
        "chemical_hints": ms_analysis.get("chemical_hints"),
        "retrieval_terms": ms_analysis.get("retrieval_terms"),
    }


def build_rerank_candidate_input(
    candidate_row: Dict[str, Any],
    source_rank: int,
    candidate_id: str,
) -> Dict[str, Any]:
    structure_analysis = candidate_row.get("structure_analysis", {})
    structure_features = candidate_row.get("structure_features", {})
    normalized_record = candidate_row.get("normalized_record", {})
    return {
        "candidate_id": candidate_id,
        "smiles": normalized_record.get("smiles"),
        "exact_mass": structure_features.get("exact_mass"),
        "molecular_formula": structure_features.get("molecular_formula"),
        "heteroatom_signature": structure_features.get("heteroatom_signature"),
        "adduct_compatibility": structure_features.get("adduct_compatibility"),
        "xlogp_class": structure_features.get("xlogp_class"),
        "scaffold": structure_analysis.get("scaffold"),
        "ring_system": structure_analysis.get("ring_system"),
        "functional_groups": structure_analysis.get("functional_groups"),
        "acid_base_class": structure_analysis.get("acid_base_class"),
        "shape_class": structure_analysis.get("shape_class"),
        "likely_losses": structure_analysis.get("likely_losses"),
        "fragile_bonds": structure_analysis.get("fragile_bonds"),
        "retrieval_terms": structure_analysis.get("retrieval_terms"),
    }


def build_rerank_prompts(
    query_row: Dict[str, Any],
    candidate_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_ids = [f"candidate_{idx + 1:03d}" for idx in range(len(candidate_rows))]
    candidate_inputs: List[Dict[str, Any]] = [
        build_rerank_candidate_input(
            candidate_row,
            source_rank=idx + 1,
            candidate_id=candidate_ids[idx],
        )
        for idx, candidate_row in enumerate(candidate_rows)
    ]
    payload = {
        "query_signature": build_rerank_query_input(query_row),
        "candidate_ids": candidate_ids,
        "candidates": candidate_inputs,
    }
    input_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "choice_system_prompt": RERANK_CHOICE_SYSTEM_PROMPT,
        "choice_user_prompt": RERANK_CHOICE_USER_PROMPT_TEMPLATE.format(input_json=input_json),
        "ranking_system_prompt": RERANK_RANKING_SYSTEM_PROMPT,
        "ranking_user_prompt": RERANK_RANKING_USER_PROMPT_TEMPLATE.format(input_json=input_json),
        "candidate_ids": candidate_ids,
    }
