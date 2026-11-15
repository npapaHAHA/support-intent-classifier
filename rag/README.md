# Banking support RAG demo

This module adds retrieval-augmented generation alongside the BANKING77 intent classifier.
It answers English support questions using 12 short, fictional banking instructions.
These are authored demo materials, not Sber policies or official advice from a bank.

## Run

Use Python 3.13. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt -r requirements-rag.txt
python -m rag.evaluate --generation
python -m uvicorn rag.api:app --port 8001
```

The activation command above is for Linux/macOS. In Windows PowerShell use
`.\.venv\Scripts\Activate.ps1`. Use the Python executable from the activated virtual environment. On Windows it is
`.venv\Scripts\python.exe`; on Linux/macOS it is `.venv/bin/python`.
The first evaluation downloads public model weights; subsequent runs can use the cache.
No API key, paid endpoint or GPU is required. Both model revisions are pinned in source.

```bash
curl --fail-with-body -H "Content-Type: application/json" \
  --data-binary @rag/example-request.json http://127.0.0.1:8001/answer
```

In PowerShell use `curl.exe` with the same arguments on one line. The response contains
`answer`, `abstained`, and `sources` with document IDs, text and cosine scores.
The best matching document is passed to generation and returned as the source.
`mode` reports `verbatim_generation`, `retrieved_quote`, or `abstained`.
Sources are retrieved context, not proof that every generated claim is supported.

## What is implemented

- Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`, normalized embeddings
  and cosine similarity; word/bigram TF-IDF is the comparison baseline.
- Local CPU generation with `Qwen/Qwen2.5-0.5B-Instruct`. One prompt simply
  asks for an answer from the notes; the other asks for a short next action, gives
  explicit context boundaries, and asks the model to avoid inventing facts.
- A similarity threshold selected using **only 12 development questions**. Below it,
  the service returns a fixed refusal and does not call the language model.
- An exact-quote check on the generated response. It must equal one complete source
  sentence or the complete source document. Otherwise the service returns the source
  document itself. Raw generation is available in offline evaluation, not the HTTP reply.
- Evaluation on **24 separate authored questions**: 18 supported and 6 unsupported,
  including questions about loans and investments that are absent from the notes.

The 12 documents are already short, self-contained chunks. There is no need for a vector
database at this scale: the embeddings fit in memory. The language model has about 0.5B
parameters; read the generated answers in the report before drawing conclusions.

## Evaluation

The checked-in [evaluation summary and raw runs](reports/README.md) include measured
retrieval scores, raw model answers, quote-fallback counts and the Docker API response.

```bash
python -m rag.evaluate                 # retrieval and development-only calibration
python -m rag.evaluate --generation    # also compare the two prompts
python -m unittest discover -s tests -v
```

`reports/evaluation.json` records input checksums, model revisions, package versions,
retrieval Recall@1/3, MRR@3, supported acceptance and unsupported rejection rates.
The complete ranked documents and generated answers are included for inspection.
The calibration output is saved to `config.json` and used by the API.

For generation, the automatic proxy checks whether all manually annotated action terms
appear in the **raw model response, before quote fallback**. This is **keyword coverage**, not factual accuracy or a guarantee
against hallucinations. The paired bootstrap resamples supported questions, preserving
both prompt results for each question. Its 95% interval applies only to this small,
authored benchmark. This is an offline prompt comparison, not a live A/B test.

The initial SmolLM2-135M run with three retrieved documents is preserved in
`reports/smollm2-baseline.json`. Its answers sometimes copied unrelated context or
answered the wrong question. Qwen with a single context document was chosen after
reading those outputs. Consequently the current generation comparison on these
same questions is exploratory, not a fresh, blinded test of model selection.

Qwen also invented details on this set, including a fee not present in the instructions.
The quote fallback was added after inspecting those outputs. Counts of fallback and
verbatim generation are reported separately. This trades free-form answers for traceable
source text. A correct quote can still be irrelevant if retrieval chooses the wrong document.

The evaluation set does not establish quality on real customer traffic. A cosine
threshold can miss unsupported questions; prompt instructions do not guarantee protection
against prompt injection. Account data, transaction execution, agents, fine-tuning,
LoRA and RLHF are outside this implementation.

## Docker and CI

```bash
docker build -f Dockerfile.rag -t support-rag .
docker run --rm -p 127.0.0.1:8001:8001 -v support-rag-cache:/tmp/huggingface support-rag
```

On first startup the container downloads model weights. `/health` becomes available after
the models have loaded. The existing classification service still uses its own Dockerfile.
The regular CI runs unit/API tests without downloading LLM weights. The separate
**RAG evaluation** workflow runs both real models, builds the additional image, makes
an HTTP request to the running container, and uploads results as an artifact.

Model references: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[initial SmolLM2 baseline](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
