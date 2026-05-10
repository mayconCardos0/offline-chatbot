# Offline Chatbot

A fully offline, privacy-first RAG chatbot. Drop documents into a folder, index them once, and chat with a local GGUF model — no internet, no cloud.

Built to run on constrained hardware (Raspberry Pi 5), with a hybrid semantic + BM25 retrieval pipeline, hallucination guardrails, and a web frontend included.

## How it works

```
User message
      │
      ▼
Retriever (FAISS HNSW + BM25)
┌─────────────────────────────┐
│ 1. Semantic search (top-k×4)│
│ 2. Minimum score filter     │
│ 3. Adaptive filter (σ)      │
│ 4. Gap detection            │
│ 5. Keyword overlap filter   │
│ 6. Hybrid reranking         │
└─────────────────────────────┘
      │ top-k chunks
      ▼
RAGPipeline
┌─────────────────────────────┐
│ Topical relevance check     │
│ (keyword overlap)           │
│ Context compression         │
│ Duplicated guardrail        │
└─────────────────────────────┘
      │
      ▼
LocalModel (llama-cpp-python)
      │
      ▼
  Response
```

1. Documents in `data/documents/` are loaded, split into semantic chunks, and indexed into FAISS.
2. On each turn, the most relevant chunks are retrieved and injected into the LLM prompt.
3. A local GGUF model generates the response — low temperature (0.1) for factual answers.
4. Conversation history is persisted to JSON between sessions.

## Project structure

```
offline-chatbot/
├── api/
│   ├── main.py          # FastAPI app, lifespan, middleware
│   └── routes.py        # REST endpoints
├── core/
│   ├── config.py        # Settings read from environment variables
│   └── conversation.py  # Session CRUD, JSON persistence
├── llm/
│   └── local_model.py   # llama-cpp-python wrapper (GGUF)
├── rag/
│   ├── loader.py        # Loads .txt, .md, .pdf, .json
│   ├── chunking.py      # Token-based semantic chunking with overlap
│   ├── embeddings.py    # SentenceTransformer + disk cache
│   ├── vectorstore.py   # FAISS HNSW with deduplication
│   ├── retriever.py     # Hybrid semantic + BM25 search
│   └── pipeline.py      # Orchestrates retrieval + LLM + guardrails
├── scripts/
│   ├── index_documents.py  # CLI to index documents
│   └── speed_test.py       # GGUF model benchmark
├── tests/               # pytest test suite
├── frontend/
│   ├── index.html       # Web UI (vanilla JS)
│   ├── main.js          # Chat logic, session management
│   └── styles.css       # Responsive layout
├── data/
│   ├── documents/       # Put your source documents here
│   └── index/           # FAISS index (written after indexing)
├── .env.example
├── requirements.txt
└── .github/
    └── workflows/       # CI/CD (see Development section)
```

## Requirements

- Python 3.10+
- A GGUF model file (e.g. `gemma-2-2b-it-Q4_K_M.gguf`) placed in `models/`
- For PDFs: PyMuPDF is installed via `requirements.txt`

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the environment
cp .env.example .env
# Edit .env — at minimum set MODEL_PATH to your GGUF file

# 4. Add your documents
cp your-docs/*.pdf data/documents/

# 5. Index the documents
python scripts/index_documents.py

# 6. Start the server
uvicorn api.main:app --port 8000
```

Open the frontend by serving `frontend/` with any static file server, or open `frontend/index.html` directly in your browser.

## Indexing documents

Supported formats: `.txt`, `.md`, `.pdf`, `.json`

```bash
# Uses DOCS_DIR from .env
python scripts/index_documents.py

# Override the directory for a one-off run
python scripts/index_documents.py --docs-dir /path/to/other/docs

# Override chunk size and overlap
python scripts/index_documents.py --chunk-size 400 --overlap 2
```

Output on completion:

```
Indexing documents from: data/documents
Config: chunk_size=512 chars  overlap=1 sentences

Documents loaded (2 file(s)):
  • textbook.pdf — 48 pages
  • notes.txt

  Chunk statistics:
    Total chunks      : 312
    Total tokens      : 47,840
    Tokens per chunk  : min=42  avg=153  max=398

───────────────────────────────────────────────────
  ✓  Indexing complete
───────────────────────────────────────────────────
  Files indexed        : 2
  Chunks in index      : 312
  Index saved to       : /home/pi/offline-chatbot/data/index
  Embedding model      : paraphrase-multilingual-MiniLM-L12-v2
  Index type           : HNSW (approximate search, low latency)
───────────────────────────────────────────────────
```

> **Note:** the FAISS index and documents are **local to each device** and are never versioned or synced by the deploy pipeline. Each Raspberry Pi has its own.

## API

Base URL: `http://localhost:8000`

### `POST /chat`

Send a message. The session is created automatically if it doesn't exist.

```json
// Request
{ "session_id": "abc123", "message": "What is photosynthesis?" }

// Response
{ "session_id": "abc123", "response": "Photosynthesis is the process by which..." }
```

### `GET /conversations`

List all conversations, sorted by most recently updated.

```json
[
  { "id": "abc123", "title": "New Chat", "updated_at": 1715000000.0 },
  { "id": "xyz789", "title": "New Chat", "updated_at": 1714990000.0 }
]
```

### `GET /conversations/{session_id}`

Return a conversation with its full message history. Returns `404` if not found.

### `DELETE /chat/{session_id}`

Delete a session. Returns `404` if it doesn't exist.

### `GET /health`

Liveness check. Returns `{ "status": "ok" }`.

## Configuration

All settings are read from environment variables (or a `.env` file). See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server TCP port |
| `MODEL_PATH` | `models/gemma-2-2b-it-Q4_K_M.gguf` | Path to the GGUF model file |
| `N_CTX` | `4096` | LLM context window (tokens) |
| `N_THREADS` | `4` | CPU threads for inference |
| `N_GPU_LAYERS` | `0` | Layers offloaded to GPU (0 = CPU only) |
| `EMBED_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers model |
| `EMBED_CACHE_DIR` | `models` | Local embedding model cache directory |
| `EMBED_BATCH_SIZE` | `32` | Encoding batch size |
| `EMBED_DISK_CACHE` | `true` | Cache embeddings to disk between restarts |
| `DOCS_DIR` | `data/documents` | Source documents directory |
| `INDEX_DIR` | `data/index` | FAISS index output directory |
| `TOP_K` | `4` | Chunks retrieved per query |
| `CANDIDATE_MULTIPLIER` | `4` | Semantic candidates = TOP_K × this value |
| `MIN_SCORE` | `0.40` | Minimum score to include a chunk |
| `LEXICAL_WEIGHT` | `0.30` | BM25 weight in reranking (0 = semantic only) |
| `CHUNK_SIZE` | `512` | Maximum chunk size in characters |
| `CHUNK_OVERLAP` | `1` | Overlap between chunks (in sentences) |
| `MAX_CONTEXT_CHARS` | `3500` | Maximum context characters injected into the prompt |
| `MAX_HISTORY_TURNS` | `4` | Conversation history turns kept per session |
| `CONVERSATIONS_FILE` | `data/conversations.json` | Session persistence file |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

## GPU acceleration

Set `N_GPU_LAYERS` to the number of model layers to offload to your GPU. A value of `-1` offloads all layers. Requires a GPU-enabled build of `llama-cpp-python`:

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
```

## Model benchmark

The `speed_test.py` script runs all `.gguf` files in `models/` against a set of Q&A pairs and prints a comparative speed and accuracy table:

```bash
python scripts/speed_test.py
python scripts/speed_test.py --threads 4 --n-ctx 2048
```

---

## Development

### Branches

```
main  ←  homolog  ←  dev  ←  feature/*
                              fix/*
```

| Branch | Purpose | Deploy |
|---|---|---|
| `main` | Production | Via semantic tag (`v1.2.3`) |
| `homolog` | Staging | Automatic after CI passes |
| `dev` | Continuous integration | — |
| `feature/*` | New features | PR → dev |

### Running tests

```bash
pip install pytest pytest-cov pytest-asyncio httpx
pytest tests/ -v --cov=. --cov-report=term-missing
```

The suite covers: chunking, config, conversation manager, loader, local model, pipeline, retriever, and API routes. All tests mock the heavy dependencies (`llama-cpp-python`, FAISS, SentenceTransformer).

### CI/CD

The project uses GitHub Actions with separate pipelines per branch and environment.

#### CI — Dev (`.github/workflows/ci-dev.yml`)

Runs on push and PR to `dev`. Optimised for speed:

- Lint: flake8 (critical errors only), black, isort
- Tests: Python 3.10 / 3.11 / 3.12 in parallel, no minimum coverage
- Security: bandit (HIGH blocks, MEDIUM does not), safety
- Structure validation: required files, directories, `.env.example`, no committed secrets

#### CI — Homolog (`.github/workflows/ci-homolog.yml`)

Runs on push and PR to `homolog`. Stricter — gates the deploy:

- Full lint (flake8 with no exceptions, black, isort)
- Tests with **minimum 70% coverage**, `fail-fast: true`
- API endpoint smoke tests
- Security: bandit HIGH+MEDIUM block, safety, pip-audit
- Hardcoded secret detection in source code
- Dependency conflict and version-pinning checks

#### CD — Homolog (`.github/workflows/cd-homolog.yml`)

Triggers automatically when the Homolog CI passes. Runs **directly on the Raspberry Pi** via a self-hosted runner — no external SSH, no open ports:

1. Code checkout on the RPi
2. `pip install` into the local venv
3. Validates that `.env` exists and contains the required variables
4. Restarts the service (systemd or nohup fallback)
5. Health check with 60-second retry
6. Smoke tests on `/health` and `/conversations`
7. Stops the service on failure to avoid running an inconsistent version

#### CD — Production (`.github/workflows/cd-production.yml`)

Triggers via semantic tag. Same self-hosted runner approach, with additional steps:

- Full test suite with **minimum 80% coverage** and zero-tolerance bandit scan
- Automatic GitHub Release creation with git log notes
- Health check with 90-second retry
- Each deploy logged to `logs/deploys.log`

**The deploy never touches `data/`** — documents, the FAISS index, and conversations are local to each device and the operator's responsibility.

#### Registering the self-hosted runner on a Raspberry Pi

The runner needs to be registered once on each RPi:

```
GitHub → Settings → Actions → Runners → New self-hosted runner
  → Linux → ARM64 → follow the commands shown on screen
```

Install as a systemd service so it starts automatically with the RPi:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

Add the label `raspberry-homolog` to the staging RPi and `raspberry-prod` to the production one.

#### Setting up `.env` on the Raspberry Pi (once only)

`.env` is not versioned and is never overwritten by the deploy:

```bash
cp .env.example .env
nano .env   # set MODEL_PATH, DOCS_DIR, INDEX_DIR, etc.
```

#### Release flow

```bash
# Develop in a feature branch, merge into dev, then into homolog
# (CI runs at each step)

# Once homolog is validated, open a PR: homolog → main
# After approval (2 reviewers required) and merge:
git checkout main && git pull origin main
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
# → Production CD triggers automatically
```
