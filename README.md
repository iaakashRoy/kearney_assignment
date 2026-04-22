# Enterprise Document Intelligence Platform

A self-contained FastAPI service that ingests PDFs, scanned documents, and
component photos, then answers two flavours of questions:

- **Structured** — natural-language → SQL over the relational catalog.
- **Analytical** — hybrid (vector + BM25) RAG with cited sources + page-level
  PDF previews.

The bundled React SPA is served by the API itself, so the whole platform
runs as a single image.

> Architecture details live in [architecture.md](architecture.md).

---

## 1. Prerequisites

| Tool          | Version     | Notes                                          |
| ------------- | ----------- | ---------------------------------------------- |
| Python        | 3.11+       | For running locally without Docker.            |
| Docker        | 24+         | Includes Docker Compose v2 (`docker compose`). |
| Groq API key  | —           | Free tier at <https://console.groq.com/keys>.  |

---

## 2. Configure environment variables

Copy the template and fill in your Groq key:

```bash
cp .env.example .env
$EDITOR .env          # set GROQ_API_KEY=...
```

Only `GROQ_API_KEY` is required; everything else has sensible defaults.

| Variable            | Default                | Purpose                                       |
| ------------------- | ---------------------- | --------------------------------------------- |
| `GROQ_API_KEY`      | —                      | **Required.** Auth for Groq text + vision.    |
| `DB_PATH`           | `platform.db`          | SQLite file location.                         |
| `LANCEDB_PATH`      | `platform_lancedb`     | LanceDB directory.                            |
| `SOURCES_DIR`       | `platform_sources`     | Where original uploads are kept for previews. |
| `INGEST_WORKERS`    | `min(4, cpu_count())`  | Async ingestion pool size.                    |
| `RAG_TOP_K`         | `5`                    | Chunks pulled per query.                      |
| `PREVIEW_DPI_SCALE` | `1.5`                  | PDF page render scale (pypdfium2).            |

See [`.env.example`](.env.example) for the full list.

---

## 3. Run with Docker (recommended)

From the repository root:

```bash
docker compose up --build
```

That builds one image and starts a single `api` service on port **8000**.
Local `./data/` is mounted at `/data/` inside the container so SQLite,
LanceDB, and uploaded source files survive container restarts.

Open the UI at:

> **<http://localhost:8000/>**

Stop with `Ctrl+C` (or `docker compose down`).

---

## 4. Run locally (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

First launch downloads the Docling layout models (~300 MB) and the
`all-MiniLM-L6-v2` embedding model (~90 MB) — subsequent runs use the cache.

Open the UI at **<http://localhost:8000/>**.

---

## 5. Using the UI

The SPA has four tabs:

1. **Ingest** — drag-and-drop PDFs / images. Uploads run on a background
   process pool; a progress bar + per-file status table updates live.
2. **Marketplace** — browse every ingested document and component.
   Inline thumbnail preview, row-level *Delete*, and a guarded *Delete ALL*
   wipe (clears SQLite, LanceDB, and `SOURCES_DIR`).
3. **SQL Query** — natural-language → SQL. Returns the generated SQL plus
   the result rows.
4. **Analytical Query** — hybrid RAG. Each cited source expands into a
   page-level preview image (PDF page render or original picture).

Re-ingest is required after upgrading from a build that didn't track
`page_number`: hit **Delete ALL** in Marketplace, then re-upload.

---

## 6. API quick reference

| Method | Path                                | Description                              |
| ------ | ----------------------------------- | ---------------------------------------- |
| GET    | `/`                                 | Serves the React SPA.                    |
| GET    | `/health`                           | Liveness probe.                          |
| POST   | `/upload`                           | Synchronous ingest of one or more files. |
| POST   | `/upload/async`                     | Submit files to the worker pool.         |
| GET    | `/jobs/{job_id}`                    | Poll async ingestion progress.           |
| POST   | `/query/structured`                 | NL → SQL.                                |
| POST   | `/query/analytical`                 | Hybrid RAG with cited sources.           |
| GET    | `/catalog`                          | List documents + components.             |
| DELETE | `/catalog/document/{id}`            | Delete a document (cascades + vectors).  |
| DELETE | `/catalog/component/{id}`           | Delete a component.                      |
| DELETE | `/catalog`                          | Wipe everything.                         |
| GET    | `/catalog/source?file=&page=`       | PDF page → PNG / image bytes preview.    |

OpenAPI docs are auto-generated at **<http://localhost:8000/docs>**.

---

## 7. Project layout

```
kearney_assignment/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── architecture.md
├── .env.example
│
├── app/                     ← installable package
│   ├── main.py              ← FastAPI factory + lifespan + UI route
│   ├── config.py            ← Pydantic Settings
│   ├── api/routes/          ← HTTP layer
│   ├── services/            ← business logic (ingestion, rag, sql_agent, …)
│   ├── db/                  ← sqlite.py, vector_store.py, schema.sql
│   ├── models/              ← Pydantic schemas
│   ├── core/                ← logging, exceptions
│   ├── utils/               ← image + text helpers
│   ├── web/                 ← bundled React SPA
│   └── data/                ← bundled benchmark CSV
│
└── tests/                   ← pytest smoke tests
```

---

## 8. Tests

```bash
source .venv/bin/activate
pip install pytest
pytest -q
```

The test suite uses an isolated temp directory for SQLite, LanceDB, and
`SOURCES_DIR` (see [`tests/conftest.py`](tests/conftest.py)) so it never
touches your local data.

---

## 9. Troubleshooting

| Symptom                                           | Fix                                                                                    |
| ------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `groq.AuthenticationError`                        | Set `GROQ_API_KEY` in `.env` (and restart).                                            |
| `database is locked`                              | Should not happen — SQLite is in WAL mode. If it does, lower `INGEST_WORKERS`.         |
| LanceDB schema error after upgrading              | Hit **Delete ALL** in the Marketplace tab — old chunks lacked `page_number`.           |
| PDF preview returns 500                           | Ensure `pypdfium2` is installed (it is a Docling dependency, so usually automatic).    |
| UI loads but cannot reach the API                 | Confirm the API is on `:8000` and that you opened `http://localhost:8000/`, not `file://…/index.html`. |

---

## 10. License

Internal project — see repository root for licensing.
