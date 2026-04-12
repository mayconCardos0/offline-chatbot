# Offline Chatbot

A fully offline, privacy-first RAG chatbot. Drop documents into a folder, index them once, then chat with a local GGUF model — no internet required.

## How it works

1. Documents in `data/documents/` are loaded, chunked, and embedded into a FAISS vector store.
2. On each chat turn the top-k most relevant chunks are retrieved and injected into the LLM prompt.
3. A local GGUF model (via `llama-cpp-python`) generates the response.
4. Conversation history is persisted to a JSON file between sessions.

```
User message
    │
    ▼
Retriever (FAISS + sentence-transformers)
    │  top-k chunks
    ▼
RAGPipeline  ──► LocalModel (llama-cpp-python)
    │
    ▼
Response
```

## Project structure

```
offline-chatbot/
├── api/            # FastAPI app, routes
├── core/           # Config, conversation manager
├── llm/            # LocalModel wrapper (llama-cpp-python)
├── rag/            # Loader, chunker, embeddings, vector store, retriever, pipeline
├── scripts/        # index_documents.py CLI
├── data/
│   ├── documents/  # Put your source documents here
│   └── index/      # FAISS index written here after indexing
├── .env.example
└── requirements.txt
```

## Requirements

- Python 3.10+
- A GGUF model file (e.g. `Qwen3-0.6B-Q8_0.gguf`) placed in `../models/` relative to this directory

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit the environment file
cp .env.example .env
# Edit .env — at minimum set MODEL_PATH to your GGUF file

# 3. Add documents to index
cp your-docs/*.pdf data/documents/

# 4. Build the vector index
python scripts/index_documents.py

# 5. Start the API server
uvicorn api.main:app --port 8000
```

## Indexing documents

Supported formats: `.txt`, `.md`, `.pdf`, `.json`

```bash
# Use the configured DOCS_DIR from .env
python scripts/index_documents.py

# Or override the directory for a one-off run
python scripts/index_documents.py --docs-dir /path/to/other/docs
```

The script prints a summary on completion:

```
Indexed 3 document(s), 42 chunk(s) -> /absolute/path/to/data/index
```

## API

Base URL: `http://localhost:8000`

### `POST /chat`

Send a message. The session is created automatically if it doesn't exist.

```json
// Request
{ "session_id": "abc123", "message": "What is the refund policy?" }

// Response
{ "session_id": "abc123", "response": "According to the policy document..." }
```

### `DELETE /chat/{session_id}`

Delete a conversation session. Returns `404` if the session doesn't exist.

### `GET /health`

Liveness check. Returns `{ "status": "ok" }`.

## Configuration

All settings are read from environment variables (or a `.env` file). See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port |
| `MODEL_PATH` | `../models/Qwen3-0.6B-Q8_0.gguf` | Path to GGUF model |
| `N_CTX` | `4096` | LLM context window (tokens) |
| `N_THREADS` | `4` | CPU threads for inference |
| `N_GPU_LAYERS` | `0` | GPU layers (0 = CPU only) |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBED_CACHE_DIR` | `../models` | Local model cache directory |
| `DOCS_DIR` | `data/documents` | Source documents directory |
| `INDEX_DIR` | `data/index` | FAISS index output directory |
| `TOP_K` | `4` | Chunks retrieved per query |
| `CHUNK_SIZE` | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `CONVERSATIONS_FILE` | `data/conversations.json` | Session persistence file |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

## GPU acceleration

Set `N_GPU_LAYERS` to the number of model layers to offload to your GPU. A value of `-1` offloads all layers. Requires a GPU-enabled build of `llama-cpp-python`:

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
```
