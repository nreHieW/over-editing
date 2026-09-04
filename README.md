# 🩹 [EMNLP 2026] When Models Edit Too Much: On the Fidelity of Minimal Code Edits

Tongyao Zhu\*, Wei Hern Lim\*, and Min-Yen Kan<sup>†</sup>

<sub>National University of Singapore &middot; \*Equal contribution &middot; <sup>†</sup>Corresponding author</sub>

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-green.svg)](pyproject.toml)
[![Venue](https://img.shields.io/badge/EMNLP%202026-main%20conference-b31b1b.svg)](#citation-)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2609.04061-purple.svg)](https://arxiv.org/abs/2609.04061)

<p align="center">
  <img src="assets/figures/over_editing_example.png" alt="One off-by-one bug repaired by a minimal patch and by six frontier models, every patch passing the same tests" width="62%">
  <br>
  <sub>One off-by-one bug in a BigCodeBench function. The minimal fix changes a single line; GPT-5.4 adds 60 lines of validation, dtype coercion, NaN masking, and resampling that no test requires. Every patch shown passes all five tests — edit fidelity is what separates them.</sub>
</p>

This repository accompanies our EMNLP 2026 main conference paper, [When Models Edit Too Much: On the Fidelity of Minimal Code Edits](https://arxiv.org/abs/2609.04061). 🔧

LLMs are increasingly asked to edit code that already exists, and correctness alone is not enough there: a good repair should also be **minimal, reviewable, and faithful** to the original implementation. We study **over-editing** — a functionally correct repair that changes more code than the minimal fix requires. 🐛

Measuring it requires knowing what the minimal repair *is*, so we build a benchmark where it is known by construction: starting from 400 BigCodeBench problems, we inject localized AST-level corruptions into each reference solution and keep the example only if the corrupted program fails the original tests. The minimal repair is then exactly the reversal of the injected corruption. Every repair is scored on **Pass@1**, **excess Levenshtein distance** (tokens changed beyond the minimal patch), and **added cognitive complexity** (control-flow overhead the fix does not need). 🎯

## Main Results 🔎

| Question | What we find |
| --- | --- |
| 🤖 Do frontier models over-edit? | Yes, and correctness does not predict it. GPT-5.5, DeepSeek, and Gemini variants reach competitive Pass@1 while changing far more code than the minimal reversal. |
| 💬 Does one sentence help? | Substantially. A single preservation clause moves excess Levenshtein distance **0.195 → 0.131**, cuts added cognitive complexity by **26.6%**, and *raises* Pass@1 by **2.3 points** — shifting all 50 model-prompt settings toward smaller edits. |
| 🧠 Are reasoning and scale enough? | No. Reasoning effects are model-specific rather than monotonic, and across Qwen2.5-Coder 0.5B→32B Pass@1 rises with size while edit fidelity does not. |
| 🎓 Can minimal editing be learned? | Yes, best with RL: **0.782** out-of-domain Pass@1 at **0.050** excess Levenshtein, holding LiveCodeBench v6 at 33.2% (+0.6 over base). SFT collapses out-of-domain to 0.458 and loses 14.9 LCB points. |

<p align="center">
  <img src="assets/figures/frontier_prompt_effect.png" alt="Pass@1 versus excess Levenshtein distance for frontier models under generic and explicit prompts" width="100%">
  <br>
  <sub>Each connected pair is one model under the generic (blue) and explicit preservation (red) prompt. Up and to the left is better. The heaviest over-editors move furthest — GPT-5.5 High nearly halves its excess distance (0.299 → 0.159) — while the already-faithful Opus 4.7 barely moves.</sub>
</p>

## What You Will Find Here 📦

```text
partial_edits/          Benchmark construction, generation, and evaluation pipeline
  utils/                Corruption, extraction, similarity, and prompt helpers
  ui/                   Standalone HTML viewer for inspecting evaluation results
models/                 Provider interfaces (OpenRouter, OpenAI-compatible, Anthropic, Google, Qwen)
evaluator/              Dockerized sandbox that executes BigCodeBench tests
data/questions/         The 400-task corrupted evaluation set
train/                  Post-training: SFT/DPO configs, PRIME-RL environment, LiveCodeBench fork
assets/figures/         Paper figures shown above
```

`data/questions/corrupted_solutions_manual_easy_400.jsonl` is the evaluation set used in the paper: 232 tasks carry one injected corruption and 168 carry two, for 568 applications across 12 corruption families. Each record has `task_id`, `prompt`, `canonical_solution`, `corrupted_solution`, and `test_code` — but note the corruption labels live under **`mutation_type`** in 172 records and **`mutation_types`** in the other 228, so read both keys when parsing. Generated artifacts (`data/code_edits/`, logs, checkpoints) are gitignored.

## Reproduce The Main Run 🚀

Install [uv](https://docs.astral.sh/uv/), sync the environment, and set the API key for your provider (generation routes through [OpenRouter](https://openrouter.ai/) by default; a `.env` file at the repo root is picked up automatically):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
export OPENROUTER_API_KEY=...
```

**Step 1 — generate repairs.** This is the paper's *generic* condition; results are written to `data/code_edits/zero_shot_generic/`:

```bash
uv run partial_edits/generate_solutions.py \
  --questions_path data/questions/corrupted_solutions_manual_easy_400.jsonl \
  --model <MODEL_NAME> \
  --generic \
  --store_token_info
```

**Step 2 — start the execution sandbox.** Tests run inside Docker, never on the host. Build from `evaluator/`, since the image copies its build context:

```bash
cd evaluator
docker build -t code-evaluator .
docker run -d --name code-evaluator -p 8000:8000 code-evaluator
cd ..
```

**Step 3 — evaluate.** This executes the tests and computes the edit-fidelity metrics:

```bash
uv run partial_edits/evaluate.py \
  --sample_path data/code_edits/zero_shot_generic/results_<MODEL_NAME>_non_reasoning_400.jsonl \
  --eval_similarity
```

`evaluate.py` parallelises test execution but computes the similarity metrics single-threaded, so raising `--max_workers` past a handful gives diminishing returns.

### Prompt Conditions 🗣️

The two prompt settings compared throughout the paper map onto flags as follows. The naming is a trap worth reading twice:

| Paper condition | How to run it | Request the model sees |
| --- | --- | --- |
| **Generic** | `--generic` | *What is wrong? Fix and complete my function.* |
| **Explicit** (preservation) | `--generic --is_explicit` | *What is wrong? Fix and complete my function **but keep as much of the original code as possible**.* |

Both conditions use the **same** system prompt — the one that states the repair task and the signature/docstring constraint without mentioning preservation. That prompt is the one `--generic` selects, so **pass `--generic` in both conditions**; `--is_explicit` is what switches generic → explicit.

Omitting `--generic` is the trap: it swaps in a different system prompt that *already* asks the model to preserve the original code, which is not the configuration the reported runs used.

Other useful flags: `--is_reasoning` selects the reasoning variant of a model, `--include_test_cases` shows the tests to the model, and `--num_shots` / `--shots_file_path` enable few-shot prompting.

## Where To Go Next 🧭

| If you want to... | Go here |
| --- | --- |
| Inspect evaluation results in a browser | [`partial_edits/ui/code_viewer.html`](partial_edits/ui/code_viewer.html) |
| See how corruptions are injected | [`partial_edits/utils/code_corruptor.py`](partial_edits/utils/code_corruptor.py) |
| Understand the edit-fidelity metrics | [`partial_edits/utils/similarity_utils.py`](partial_edits/utils/similarity_utils.py) |
| Add or change a model provider | [`models/__init__.py`](models/__init__.py) |
| Run SFT, DPO, or RL | [`train/README.md`](train/README.md) |

## Notes On This Release 📝

- **Trained checkpoints and raw result files are not part of this release yet.** The code here reproduces the benchmark, the prompting experiments, and the training pipeline; the model artifacts will follow.
- **Generation routes through OpenRouter.** In [`models/__init__.py`](models/__init__.py), the provider-specific and `local/` (vLLM) branches of `get_model` are commented out, so serving an open-weight model from a local endpoint needs that routing re-enabled.
- **`train/LiveCodeBench/` is a vendored fork** of [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), which fixes several upstream bugs and adds model support. It keeps its own MIT `LICENSE` and is not covered by this repository's Apache-2.0 license.

## Acknowledgements 🙏

We thank the members of [WING@NUS](https://wing.comp.nus.edu.sg/) for their feedback throughout this project, and Prime Intellect for sponsoring the compute and API costs.

## Citation ✍️

If this code is useful for your research, please cite our EMNLP 2026 paper:

```bibtex
@misc{zhu2026modelseditmuchfidelity,
      title={When Models Edit Too Much: On the Fidelity of Minimal Code Edits}, 
      author={Tongyao Zhu and Wei Hern Lim and Min-Yen Kan},
      year={2026},
      eprint={2609.04061},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2609.04061}, 
}
```

## License 📄

This repository is released under the [Apache-2.0](LICENSE) license, except for `train/LiveCodeBench/`, which retains its upstream MIT license.
