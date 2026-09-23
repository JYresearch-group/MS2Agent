# MS2Agent

MS²Agent is an end-to-end LLM multi-agent framework for MS² annotation via textualization and collaborative candidate reranking.

The link to the initial version of the model is provided below.

| Model | Link |
|-------|------|
| **MS2Rep-0.8B-GRPO** | [Qwen3.5-0.8B-grpo-ms2agent](https://huggingface.co/monaaaaaa/Qwen3.5-0.8B-grpo-ms2agent/tree/main) |
| **BERT-Sim** | [bert-ms2agent](https://huggingface.co/monaaaaaa/bert-ms2agent) |

> Stay tuned for the model and other training code...

---

## Architecture

MS2Agent separates MS/MS annotation into four modules:

1. **MSpreprocessor** — Converts experimental MS/MS spectra into standardized retrieval signatures
2. **Structureextractor** — Converts candidate SMILES and structural features into standardized retrieval signatures
3. **Matcher** — Cross-modal matching via Qwen3-Embedding cosine similarity + BERT-Sim binary classification
4. **Reranker** — LLM-based reranking with evidence-priority criteria

---

## Installation

```bash
pip install -r requirements.txt
```

Environment: Python 3.10, PyTorch 2.10.0, CUDA 12.8, Transformers 5.3.0, vLLM 0.18.0, RDKit 2023.09.5

---

## Usage

```bash
# 1. Start vLLM service
bash src/casmi_pipeline/start_vllm.sh

# 2. Run pipeline
python -m src.casmi_pipeline.cli --input data/ --output results/
```

配置参数见 `configs/pipeline_config.yaml`，详细设置见 `src/casmi_pipeline/SETUP.md`。

---

## Repository Structure

```
MS2Agent/
├── src/
│   ├── casmi_pipeline/    # Main framework (4 modules)
│   ├── bert_sim/          # BERT-Sim training code
│   └── evaluation/        # Evaluation pipeline
├── prompts/               # Agent prompts
├── configs/               # Pipeline & training configs
├── models/                # Model configs
├── data/                  # Test datasets
├── results/               # Evaluation results
├── docs/                  # IMSS specification & SI
└── requirements.txt
```

---

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Hai Jiang (jianghai_777@126.com)
