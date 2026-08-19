# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Offline, privacy-first RAG chatbot for PT-BR high-school tutoring content. Documents are indexed locally into FAISS; a local GGUF model (via `llama-cpp-python`) answers questions grounded only in retrieved context. Designed to run on constrained hardware (Raspberry Pi 5) — no internet dependency at runtime, no cloud calls.

Most in-code comments, prompt templates, and log messages are written in Portuguese (PT-BR) since the product targets Brazilian students; keep new code in the same style unless told otherwise.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env   # then set MODEL_PATH at minimum

# Index documents (reads DOCS_DIR from .env)
python scripts/index_documents.py
python scripts/index_documents.py --docs-dir /path/to/other/docs
python scripts/index_documents.py --chunk-size 400 --overlap 2

# Run the server
uvicorn api.main:app --port 8000

# Tests
pip install pytest pytest-cov pytest-asyncio httpx
pytest tests/ -v --cov=. --cov-report=term-missing
pytest tests/test_retriever.py -v                 # single file
pytest tests/test_retriever.py::test_name -v       # single test

# Lint (must pass exactly as CI runs it)
pip install flake8 black isort
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv,models,data,__pycache__  # dev CI (critical only)
flake8 . --count --max-line-length=120 --extend-ignore=E203,E501,E741,W503 --statistics --exclude=.venv,models,data,__pycache__  # homolog/main CI (strict)
black --check --diff .
isort --check-only --diff .

# Model benchmark (all .gguf files in models/ against Q&A pairs)
python scripts/speed_test.py
python scripts/speed_test.py --threads 4 --n-ctx 2048

# RAG evaluation / experimentation
python scripts/generate_eval_dataset.py          # builds data/eval/dataset.json from the real indexed chunks
python scripts/eval_rag.py --mode file --dataset data/eval/dataset.json
python scripts/run_rag_experiment.py run --name baseline --dataset data/eval/dataset.json
python scripts/analyze_rag_failures.py --dataset data/eval/dataset.json

# Generation quality evaluation (RAGAS — judged by models/ragas/*.gguf, not in requirements.txt)
pip install ragas
python scripts/eval_generation.py --dataset data/eval/dataset.json                              # every .gguf in models/
python scripts/eval_generation.py --dataset data/eval/dataset.json --model models/some-model.gguf  # one candidate model
```

CI installs dependencies with `llama-cpp-python` stripped out of `requirements.txt` (heavy binary, mocked in tests) — all tests must pass with that package absent/mocked. `tests/conftest.py` adds the repo root to `sys.path`; there is no package install step.

## Architecture

### Request flow

```
POST /chat → RAGPipeline.chat()
  1. Retriever.retrieve(query)         — hybrid search + filter cascade → chunks or []
  2. topical relevance check           — keyword overlap between query and chunks;
                                          below threshold → canned "not in material" response, LLM never called
  3. build system prompt with chunks   — context capped at max_context_chars, confidence prefix prepended
  4. LocalModel.chat(messages)         — llama-cpp-python, temperature 0.1
  5. clean_model_response()            — strips <think> blocks, prompt echoes
  6. persist turn to ConversationManager (JSON file)
```

Every setting that tunes this pipeline (score thresholds, weights, filter toggles, chunk sizes...) lives in `core/config.py::Settings`, sourced from environment variables / `.env`. `core/config.py` is the single source of truth — do not hardcode tuning constants elsewhere; when a module needs a fallback default (e.g. module-level constants in `retriever.py`), keep it in sync with the `Settings` default it mirrors.

### Structure-aware chunking (`rag/loader.py`, `rag/structure.py`)

Didactic PDFs in this corpus mark headings inconsistently: Unidade/Capítulo use a literal text prefix ("Unidade 1:", "Capitulo 1:"), but subtítulo (topic) headings often carry **no textual marker at all** — they're distinguished purely by bold formatting at body font size. `rag/loader.py::_load_pdf_pages()` extracts real font info per line via PyMuPDF's `get_text("dict")` (span-level `bold` flag + `size`), attaching `bold_lines`/`body_size` to each page dict — plain `get_text("text")` extraction discards this and cannot detect these headings. `rag/structure.py::_classify_line()` combines the existing text-pattern regex (unit/chapter/numbered-topic) with this font signal: bold + much-larger-than-body → unit-level heading; bold + body-sized + passes a heading-shape sanity filter (`_looks_like_heading_text`) → subtítulo. `merge_tiny_sections()` folds sections that end up too small (a lone short paragraph under a subtítulo) into the next one before chunking, so `chunk_document()` — called once per detected section in `scripts/index_documents.py` — doesn't emit a flood of tiny, low-context chunks. Getting this right is the highest-leverage lever on retrieval recall/precision, since it directly controls chunk boundaries.

### Retrieval pipeline (`rag/retriever.py`)

Two-stage design:
- **Stage 1 (candidates):** FAISS HNSW semantic search returns `top_k * candidate_multiplier` chunks; a dependency-free BM25 implementation scores the same chunks lexically.
- **Stage 2 (filter + rerank):** absolute min-score filter → adaptive filter (`best_score - sigma * std`) → optional gap filter → optional keyword-overlap filter → hybrid rerank (weighted fusion of normalized semantic + BM25 scores) → optional cross-encoder rerank (`rag/cross_encoder.py`, score-fused with the hybrid score when `CROSS_ENCODER_ENABLED=true`).

Each filter can be toggled independently via `Settings` (`gap_filter_enabled`, `keyword_filter_enabled`, `adaptive_sigma`, etc.). `effective_k` is always the explicit `k` passed to `retrieve()` or `top_k` from `Settings` — there is intentionally no query-phrase-based dynamic adjustment (a prior PT-BR biographical/period heuristic and a single-topic score boost were removed for being corpus-specific overfits that provided no benefit outside one narrow slice of one particular book; don't reintroduce query-type special-casing here — if a query pattern needs different retrieval depth, that's a `Settings` change, not a heuristic). `retrieve_with_trace()` exposes a `RetrievalTrace` with per-stage candidate snapshots for debugging retrieval quality — use it (via `scripts/analyze_rag_failures.py`) instead of ad hoc logging when investigating a bad retrieval.

### Hallucination guardrails (`rag/pipeline.py`)

Layered defenses, in order: retriever score filters → topical relevance check (keyword overlap, with proper-noun special-casing) → repeated "use only the context" instruction (in both system prompt and appended to the user message — needed because small models like Gemma tend to ignore system-only instructions) → confidence-scaled response prefix. When changing prompt templates or thresholds here, all three layers need to stay consistent or hallucination regressions can slip through one gap while another layer is being tuned.

### Other modules

- `rag/chunking.py` — token-based semantic chunking with overlap, paragraph/sentence-aware splitting, boilerplate stripping.
- `rag/embeddings.py` — SentenceTransformer wrapper with an optional disk cache for embeddings across restarts.
- `rag/vectorstore.py` — FAISS HNSW index with deduplication.
- `rag/evaluation.py` — generic retrieval-quality metrics (Precision/Recall/F1/Hit Rate/MRR/NDCG/MAP@K) and dataset I/O, decoupled from any particular corpus; `relevant_chunks` in a dataset entry is a list, so a query can cite 2+ chunks as relevant (already handled by `evaluate_query`, no special-casing needed). `scripts/generate_eval_dataset.py` builds `data/eval/dataset.json` from the real indexed chunks — stratified sampling across Unidade/Capítulo via `sample_chunk_groups()`, then one LLM-generated question per group via the same `LocalModel` used in chat: `n_single` groups of 1 chunk (category `factual`), plus `n_pairs`/`n_triples` groups of 2-3 chunks from the *same chapter* (category `multi_hop`, prompted to require combining all chunks in the group, not answerable from just one), plus a small fixed generic negative-query set. This is the primary, whole-corpus eval dataset; `scripts/eval_rag.py` / `run_rag_experiment.py` / `analyze_rag_failures.py` consume it (or any dataset in the same schema) via the shared `scripts/_eval_common.py::load_retriever_from_settings()`, which mirrors `api/main.py`'s retriever construction exactly (including the cross-encoder when enabled) so evaluation numbers reflect the real deployed pipeline. `data/eval/benchmark_v1*.json` is a legacy hand-curated dataset covering only one narrow slice of one corpus — kept for reference, no longer the default.
- `rag/generation_evaluation.py` — generation-quality metrics via RAGAS (faithfulness/answer_relevancy/context_precision/context_recall/answer_correctness), scored by a local GGUF judge model (`Settings.ragas_model_path`, default `models/ragas/*.gguf`) instead of an external API. `ragas`/`langchain-core` are only imported lazily inside `_score_with_ragas()` so the module (and its pure aggregation logic in `GenerationEvaluationReport`) stays importable/testable without those packages installed — same reasoning as `llama-cpp-python` being stripped from CI. `scripts/eval_generation.py` is the CLI: evaluates every `.gguf` candidate in `models/` (or one via `--model`) against the same dataset schema as `rag/evaluation.py` (needs `reference_answer`), writing versioned results to `data/metrics/generation/vN/` via the same `next_metrics_version()` helper `eval_rag.py` uses for `data/metrics/`.
- `core/conversation.py` — session CRUD with JSON file persistence (`CONVERSATIONS_FILE`).
- `api/main.py` — FastAPI lifespan wires up `LocalModel → EmbeddingModel → VectorStore → Retriever → RAGPipeline → ConversationManager` once at startup and stores them on `app.state`; `api/routes.py` holds the actual endpoints.

**The FAISS index, `data/documents/`, and `data/conversations.json` are local to each device and are never versioned or touched by the deploy pipeline.** Each Raspberry Pi indexes its own documents.

## Branching and CI/CD

```
main  ←  homolog  ←  dev  ←  feature/*, fix/*
```

- `dev`: fast CI (flake8 critical-only, tests on 3.10/3.11/3.12 with no coverage minimum, bandit HIGH-blocking, structure validation).
- `homolog`: strict CI gate (full lint, tests with **70% minimum coverage**, fail-fast, API smoke tests, bandit HIGH+MEDIUM blocking, pip-audit, secret scanning) — CD then deploys automatically to the staging Raspberry Pi via a self-hosted runner.
- `main`: production, deployed via semantic git tag (`v1.2.3`) after a `homolog → main` PR with 2 required reviewers; CI requires **80% minimum coverage** and a zero-tolerance bandit scan.

Both CD pipelines run directly on Raspberry Pi self-hosted runners (no SSH, no open ports) and never touch `data/`. See `README.md` for the full release flow and self-hosted runner registration steps.

## Agent skills

### Issue tracker

Issues live on GitHub (`ArthurFariasds/offline-chatbot1`), via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout — `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
