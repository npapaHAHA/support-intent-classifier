# Evaluation evidence

Run: [GitHub Actions, actual CPU inference and Docker API check](https://github.com/npapaHAHA/support-intent-classifier/actions/runs/44603953595).
Code commit: `1f12ff84bc3003a79b161feeb55bdce4f1ddc7d8`.
Both public model revisions, package versions, input hashes, rankings and raw generated responses are recorded in [evaluation.json](evaluation.json).
The HTTP container response is saved in [api-smoke.json](api-smoke.json).

## Retrieval

Development: 12 authored questions, used only to select similarity thresholds.
Evaluation: 24 authored questions about 12 fictional instructions; 18 supported and 6 unsupported.
Recall and MRR denominators include only the 18 supported questions. These small samples do not estimate production quality.

| Retriever | Recall@1 | Recall@3 | MRR@3 | Unsupported rejection |
|---|---:|---:|---:|---:|
| tfidf | 0.778 | 1.000 | 0.870 | 0.833 |
| dense | 1.000 | 1.000 | 1.000 | 1.000 |

## Generation and service behavior

All values below cover 18 supported questions. Keyword coverage measures the **raw model output before fallback**, not factual correctness.

| Prompt | Raw action-term coverage | Verbatim model answers used | Source-quote fallbacks |
|---|---:|---:|---:|
| plain | 0.667 | 0 | 18 |
| grounded | 0.833 | 0 | 18 |

The paired difference in raw keyword coverage (grounded minus plain) is 0.167,
with percentile-bootstrap 95% interval [-0.111, 0.444]
(5,000 paired question resamples, seed 42).
The interval includes zero, so this small comparison does not establish an advantage for either prompt.

SmolLM2 and then Qwen were inspected during development. The historical runs are preserved in
[smollm2-baseline.json](smollm2-baseline.json) and
[qwen-before-quote-check.json](qwen-before-quote-check.json).
Qwen and the quote policy were chosen after reading model outputs on this same set.
The generation readout is exploratory; there is no fresh blinded model-selection test or online A/B experiment.

The final service accepts only complete source sentences or the complete source text from the LLM.
Other generated text is replaced with the retrieved instruction. This is intentionally conservative:
it prevents novel text from reaching the API, but it can discard useful paraphrases and does not fix a wrong retrieval result.
Raw model errors remain visible in the evaluation report. Six unsupported questions are evaluated separately from generation.

The full workflow passed 22 tests, built both application dependencies and the RAG container,
and verified that the HTTP answer is a source quote and omits raw generation.
