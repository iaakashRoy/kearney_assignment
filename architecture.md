# Architecture

Enterprise Document Intelligence Platform — a self-contained FastAPI service
that ingests mixed documents and component photos, then answers two flavours
of questions:

| Endpoint              | Purpose                                                           |
| --------------------- | ----------------------------------------------------------------- |
| `POST /query/structured`  | Natural-language → SQL over the relational catalog.           |
| `POST /query/analytical`  | Hybrid (vector + keyword) RAG with cited sources.             |

The bundled React SPA (`app/web/index.html`) is served at `GET /` so the API
ships as a single image with no separate frontend container.

---

## 1. Layered View

```mermaid
graph TB
    subgraph Client["Clients"]
        UI["index.html<br/>React SPA"]
        EXT["curl / external<br/>API consumers"]
    end

    subgraph API["API Layer · app/api/"]
        H["health.py"]
        ING["ingestion.py<br/>(upload, /jobs)"]
        QRY["query.py<br/>(structured, analytical)"]
        CAT["catalog.py<br/>(list, delete, preview)"]
    end

    subgraph Services["Service Layer · app/services/"]
        IS["ingestion.py<br/>extract → chunk → embed → persist"]
        JM["job_manager.py<br/>ProcessPoolExecutor"]
        RA["rag.py<br/>hybrid retrieval + RRF + LLM"]
        SA["sql_agent.py<br/>NL → SQL"]
        LLM["llm.py<br/>Groq client (text + vision)"]
        EMB["embedding.py<br/>SentenceTransformer"]
    end

    subgraph Storage["Persistence · app/db/ + app/data/"]
        SQ["sqlite.py<br/>SQLite + FTS5"]
        VS["vector_store.py<br/>LanceDB"]
        FS["sources_dir/<br/>original uploads"]
    end

    subgraph External["External / Local Models"]
        GROQ["Groq API<br/>(text + vision)"]
        ST["sentence-transformers<br/>local CPU"]
        DL["Docling<br/>local PDF parser"]
        PD["pypdfium2<br/>PDF → PNG"]
    end

    UI -->|HTTP| API
    EXT -->|HTTP| API
    ING --> IS
    ING --> JM
    JM --> IS
    QRY --> RA
    QRY --> SA
    CAT --> SQ
    CAT --> VS
    CAT --> FS
    CAT --> PD
    IS --> LLM
    IS --> EMB
    IS --> SQ
    IS --> VS
    IS --> FS
    IS --> DL
    RA --> EMB
    RA --> VS
    RA --> SQ
    RA --> LLM
    SA --> SQ
    SA --> LLM
    LLM --> GROQ
    EMB --> ST
```

---

## 2. Folder Layout

```
kearney_assignment/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── architecture.md
├── .env.example
├── .gitignore
│
├── app/                         ← installable Python package
│   ├── __init__.py
│   ├── main.py                  ← FastAPI factory + lifespan + UI route
│   ├── config.py                ← Pydantic Settings (env-driven)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── health.py        ← GET /health
│   │       ├── ingestion.py     ← POST /upload, /upload/async, GET /jobs/{id}
│   │       ├── query.py         ← POST /query/structured, /query/analytical
│   │       └── catalog.py       ← GET/DELETE /catalog, GET /catalog/source
│   │
│   ├── services/
│   │   ├── ingestion.py         ← orchestrates extract → chunk → embed → persist
│   │   ├── job_manager.py       ← async ProcessPoolExecutor for /upload/async
│   │   ├── rag.py               ← hybrid retrieval + RRF + LLM reasoning
│   │   ├── sql_agent.py         ← NL → SQL with read-only guardrails
│   │   ├── llm.py               ← Groq text + vision client (tenacity retry)
│   │   └── embedding.py         ← SentenceTransformer wrapper
│   │
│   ├── db/
│   │   ├── sqlite.py            ← connections, migrations, CRUD, FTS5
│   │   ├── vector_store.py      ← LanceDB schema + add/search/delete
│   │   └── schema.sql           ← canonical DDL (loaded by init_db)
│   │
│   ├── models/
│   │   └── schemas.py           ← Pydantic request/response models
│   │
│   ├── core/
│   │   ├── exceptions.py        ← typed exception hierarchy
│   │   └── logging.py           ← structured logger config
│   │
│   ├── utils/
│   │   ├── image_preprocess.py  ← OpenCV pre-OCR pipeline
│   │   └── text.py              ← chunking helpers
│   │
│   ├── web/
│   │   └── index.html           ← React SPA (served by FastAPI at /)
│   │
│   └── data/
│       └── benchmark_component_costs.csv
│
└── tests/
    ├── conftest.py              ← isolates DB / LanceDB to a temp dir
    └── test_smoke.py            ← package + factory smoke tests
```

Each layer depends only on layers below it:

```
api  →  services  →  {db, llm, embedding}  →  config / core
```

---

## 3. Ingestion Flow (`POST /upload` and `/upload/async`)

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as routes/ingestion.py
    participant JM as services/job_manager.py
    participant IS as services/ingestion.py
    participant DL as Docling
    participant LLM as services/llm.py
    participant EMB as services/embedding.py
    participant SQ as db/sqlite.py
    participant VS as db/vector_store.py
    participant FS as sources_dir/

    Client->>API: multipart upload
    API->>JM: submit_job(files)
    JM-->>API: job_id (pending)
    API-->>Client: 202 + job_id

    loop per file (in worker process)
        JM->>IS: ingest_file(name, bytes)
        IS->>FS: save original (for previews)
        alt PDF
            IS->>DL: convert → DoclingDocument
            IS->>IS: per-page export_to_markdown()
        else Image
            IS->>LLM: vision OCR + classify
        end
        IS->>IS: structure_aware_chunks(page)
        IS->>EMB: encode(chunks)
        IS->>SQ: insert document + entities + FTS rows (with page_number)
        IS->>VS: add_chunks(vectors + page_number)
        IS-->>JM: per-file status
    end

    Client->>API: GET /jobs/{job_id}
    API-->>Client: progress + per-file status
```

---

## 4. Analytical Query Flow (`POST /query/analytical`)

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as routes/query.py
    participant RAG as services/rag.py
    participant EMB as services/embedding.py
    participant VS as db/vector_store.py
    participant SQ as db/sqlite.py
    participant LLM as services/llm.py

    Client->>API: { question }
    API->>RAG: reason(question)
    RAG->>EMB: encode(question)
    par vector search
        RAG->>VS: search(embedding, top_k)
    and keyword search
        RAG->>SQ: keyword_search_chunks(question)
    end
    RAG->>RAG: RRF fuse (k=60)
    RAG->>LLM: prompt(chunks_with_pages)
    LLM-->>RAG: answer + cited sources
    RAG->>RAG: attach page_number per source from retrieval
    RAG-->>API: AnalyticalQueryResponse
    API-->>Client: { answer, sources[{ file, page, excerpt, score }] }
```

The UI then calls `GET /catalog/source?file=…&page=…` per source to render an
inline preview (PDF page → PNG via pypdfium2, or the original image bytes).

---

## 5. Persistence Schema (high-level)

| Store    | Holds                                                                | Where                       |
| -------- | -------------------------------------------------------------------- | --------------------------- |
| SQLite   | `documents`, `entities`, `components`, `relationships`, `chunks_fts` | `settings.db_path` (WAL)    |
| LanceDB  | `chunks` table (vector + source_file + page_number + chunk metadata) | `settings.lancedb_path/`    |
| Filesystem | Original upload bytes (for preview rendering)                      | `settings.sources_dir/`     |

`chunks_fts` is a SQLite FTS5 virtual table tokenised with `porter unicode61`.
LanceDB uses cosine distance over a 384-dim space (`all-MiniLM-L6-v2`).

---

## 6. Concurrency Model

* **API process** — single uvicorn worker; FastAPI handles requests cooperatively.
* **Ingestion** — `ProcessPoolExecutor` (size = `INGEST_WORKERS`, default
  `min(4, cpu_count())`). Each file is processed in a child process so heavy
  CPU work (Docling, sentence-transformers, OpenCV) doesn't block the API.
* **SQLite** — WAL journal mode + 30 s `busy_timeout` lets workers commit
  concurrently without "database is locked".

---

## 7. Configuration

All knobs live in [`app/config.py`](app/config.py) and are env-driven via
Pydantic Settings. See [`.env.example`](.env.example) for every variable.
# Architecture — Enterprise Document Intelligence Platform

## 1. High-Level System Overview

```mermaid
graph TB
    subgraph Client["Client Layer"]
        UI["index.html<br/>React SPA<br/>(served by API at /)"]
        EXT["External Clients<br/>(curl / API)"]
    end

    subgraph API["API Layer  ·  app/main.py"]
        H["/health<br/>routes/health.py"]
        ING["/upload<br/>routes/ingestion.py"]
        QS["/query/structured<br/>routes/query.py"]
        QA["/query/analytical<br/>routes/query.py"]
    end

    subgraph Services["Service Layer"]
        IS["ingestion.py<br/>File Orchestrator"]
        RA["rag.py<br/>RAG Pipeline"]
        SA["sql_agent.py<br/>NL→SQL Agent"]
        LLM["llm.py<br/>Groq LLM Client"]
        EMB["embedding.py<br/>SentenceTransformer"]
    end

    subgraph Storage["Storage Layer"]
        SQ["db/sqlite.py<br/>SQLite  platform.db"]
        VS["db/vector_store.py<br/>LanceDB  platform_lancedb/"]
    end

    subgraph External["External APIs"]
        GV["Groq Vision API<br/>(image OCR + classification)"]
        GT["Groq Text API<br/>(entity extraction, reasoning, SQL)"]
        ST["sentence-transformers<br/>(local CPU model)"]
        DC["Docling<br/>(local PDF parser)"]
    end

    UI -->|HTTP REST| API
    EXT -->|HTTP REST| API

    H -->|200 ok| UI
    ING --> IS
    QS --> SA
    QA --> RA

    IS --> LLM
    IS --> EMB
    IS --> SQ
    IS --> VS

    RA --> EMB
    RA --> VS
    RA --> SQ
    RA --> LLM

    SA --> SQ
    SA --> LLM

    LLM -->|vision calls| GV
    LLM -->|text calls| GT
    EMB -->|local inference| ST
    IS -->|PDF extraction| DC
```

---

## 2. File Map & Responsibilities

```mermaid
graph LR
    subgraph Entry["Entry Points"]
        MP["main.py<br/>─────────────────<br/>IN: uvicorn start<br/>OUT: FastAPI app"]
        UI["index.html<br/>─────────────────<br/>IN: browser GET /<br/>OUT: React SPA bundle"]
    end

    subgraph AppCore["app/"]
        CFG["config.py<br/>─────────────────<br/>IN: .env / env vars<br/>OUT: Settings singleton"]
        MAIN["main.py<br/>─────────────────<br/>IN: lifespan hook<br/>OUT: FastAPI + routers"]
    end

    subgraph CorePkg["app/core/"]
        LOG["logging.py<br/>─────────────────<br/>IN: module name<br/>OUT: Logger"]
        EXC["exceptions.py<br/>─────────────────<br/>IN: –<br/>OUT: typed exception classes"]
    end

    subgraph Models["app/models/"]
        SCH["schemas.py<br/>─────────────────<br/>IN: raw dicts / JSON<br/>OUT: validated Pydantic models<br/>(QueryRequest, IngestResult,<br/>StructuredQueryResponse,<br/>AnalyticalQueryResponse, …)"]
    end

    subgraph Utils["app/utils/"]
        TXT["text.py<br/>─────────────────<br/>IN: raw strings / bytes<br/>OUT: chunks[], JSON dict,<br/>base64 data-URL"]
    end

    subgraph Routes["app/api/routes/"]
        RH["health.py<br/>─────────────────<br/>IN: GET /health<br/>OUT: {status: ok}"]
        RI["ingestion.py<br/>─────────────────<br/>IN: multipart files<br/>OUT: UploadResponse"]
        RQ["query.py<br/>─────────────────<br/>IN: {question}<br/>OUT: StructuredQueryResponse<br/>     AnalyticalQueryResponse"]
    end

    subgraph DB["app/db/"]
        SQdb["sqlite.py<br/>─────────────────<br/>IN: typed args<br/>OUT: rows, doc_id, schema DDL"]
        VSdb["vector_store.py<br/>─────────────────<br/>IN: chunk dicts + float[] vector<br/>OUT: top-k scored chunks"]
    end

    subgraph Svc["app/services/"]
        LLMs["llm.py<br/>─────────────────<br/>IN: prompt str / image bytes<br/>OUT: str (text), dict (vision)<br/>Retries: tenacity exp back-off"]
        EMBs["embedding.py<br/>─────────────────<br/>IN: list[str]<br/>OUT: list[list[float]] (384-d)"]
        INGs["ingestion.py<br/>─────────────────<br/>IN: filename, bytes<br/>OUT: IngestResult<br/>(status, detail, input_type)"]
        RAGs["rag.py<br/>─────────────────<br/>IN: question str<br/>OUT: AnalyticalQueryResponse<br/>(answer, sources, confidence)"]
        SQLs["sql_agent.py<br/>─────────────────<br/>IN: question str<br/>OUT: StructuredQueryResponse<br/>(sql, result rows, assumption)"]
    end

    MP --> MAIN
    UI --> RH & RI & RQ
    MAIN --> RH & RI & RQ
    RH --> LOG
    RI --> INGs
    RQ --> RAGs & SQLs

    CFG --> LLMs & EMBs & INGs & RAGs & SQLs & SQdb & VSdb
    LOG --> LLMs & EMBs & INGs & RAGs & SQLs & SQdb & VSdb
    EXC --> LLMs & EMBs & INGs & RAGs & SQLs & SQdb & VSdb
    SCH --> RI & RQ & INGs & RAGs & SQLs

    INGs --> LLMs & EMBs & SQdb & VSdb & TXT
    RAGs --> LLMs & EMBs & SQdb & VSdb & TXT
    SQLs --> LLMs & SQdb & TXT
```

---

## 3. Ingestion Pipeline (POST /upload)

```mermaid
flowchart TD
    A(["Client uploads file(s)<br/>PDF | PNG | JPG"])
    B["routes/ingestion.py<br/>reads UploadFile bytes"]
    C{"Extension?"}

    D["services/ingestion.py<br/>_extract_pdf()<br/>─ Docling converts PDF → markdown"]
    E["services/llm.py<br/>ocr_image()<br/>─ Groq Vision reads image text"]

    F{"Enough text?<br/>> ocr_text_threshold chars"}

    G["Stream A — Document"]
    H["Stream B — Component Photo"]

    G1["services/ingestion.py<br/>_extract_entities()<br/>─ Groq text LLM → JSON<br/>{doc_type, confidence,<br/> entities[], summary}"]
    G2["app/db/sqlite.py<br/>insert_document()<br/>insert_entities()"]
    G3["services/embedding.py<br/>embed(chunks)<br/>─ SentenceTransformer → float[][]"]
    G4["app/db/vector_store.py<br/>add_chunks()<br/>─ LanceDB stores vectors"]

    H1["services/llm.py<br/>classify_component_image()<br/>─ Groq Vision → JSON<br/>{component_type, material,<br/> size_category, confidence}"]
    H2["services/ingestion.py<br/>_lookup_cost()<br/>─ benchmark_component_costs.csv"]
    H3["app/db/sqlite.py<br/>insert_component()"]

    Z(["IngestResult<br/>status: ok | error | skipped"])

    A --> B --> C
    C -->|.pdf| D --> G
    C -->|.png/.jpg| E --> F
    F -->|yes → scanned_doc| G
    F -->|no → component_photo| H

    G --> G1 --> G2
    G2 --> G3 --> G4 --> Z

    H --> H1 --> H2 --> H3 --> Z
```

---

## 4. Analytical RAG Query (POST /query/analytical)

```mermaid
flowchart TD
    A(["Client: {question}"])
    B["routes/query.py"]
    C["services/rag.py<br/>analytical_query()"]

    subgraph Agent1["Agent 1 — Retrieval"]
        D["services/embedding.py<br/>embed([question])<br/>→ float[384]"]
        E["app/db/vector_store.py<br/>search(vector, top_k=5)<br/>→ chunks with _distance"]
        F["app/db/sqlite.py<br/>fetch_all_components()<br/>→ component rows"]
    end

    subgraph Agent2["Agent 2 — Reasoning"]
        G["Build grounded prompt<br/>system + chunks_text + components_text"]
        H["services/llm.py<br/>generate_text()<br/>─ Groq text model<br/>─ tenacity retry"]
        I["utils/text.py<br/>extract_json(raw)<br/>→ {answer, sources, confidence}"]
        J["_determine_confidence()<br/>based on best _distance:<br/>< 0.30 → high<br/>0.30-0.50 → medium<br/>0.50-0.65 → low<br/>> 0.65 → none"]
    end

    Z(["AnalyticalQueryResponse<br/>{answer, sources[], confidence,<br/>grounding_warning}"])

    A --> B --> C
    C --> Agent1
    D --> E
    D --> F
    Agent1 --> Agent2
    G --> H --> I --> J --> Z
```

---

## 5. Structured SQL Query (POST /query/structured)

```mermaid
flowchart TD
    A(["Client: {question}"])
    B["routes/query.py"]
    C["services/sql_agent.py<br/>structured_query()"]

    D["app/db/sqlite.py<br/>get_schema_ddl() + get_sample_rows(3)<br/>→ schema DDL, sample JSON"]
    E["Build system prompt<br/>schema + samples + question<br/>(+ prior_sql + prior_error on retry)"]
    F["services/llm.py<br/>generate_text()<br/>─ Groq text model"]
    G["utils/text.py<br/>extract_json(raw)<br/>→ {sql, assumption}"]
    H{"_validate_sql()<br/>blocks DML/DDL<br/>checks parseable"}
    I["app/db/sqlite.py<br/>execute SELECT<br/>→ list[dict]"]

    J{{"Attempt 1 OK?"}}
    K["Retry attempt 2<br/>injects prior_sql + error<br/>into prompt (self-correction)"]

    Z(["StructuredQueryResponse<br/>{question, sql, result[],<br/>assumption, error}"])

    A --> B --> C --> D
    D --> E --> F --> G --> H
    H -->|valid| I --> Z
    H -->|invalid| J
    I -->|SQLite error| J
    J -->|no| K --> F
    J -->|yes / 2nd fail| Z
```

---

## 6. Storage Schema

```mermaid
erDiagram
    documents {
        INTEGER id PK
        TEXT source_file UK
        TEXT input_type
        TEXT doc_type
        REAL doc_type_confidence
        TEXT summary
        TEXT raw_text
        DATETIME created_at
    }
    entities {
        INTEGER id PK
        INTEGER document_id FK
        TEXT entity_type
        TEXT value
        TEXT normalized
    }
    components {
        INTEGER id PK
        TEXT source_file UK
        TEXT component_type
        TEXT material
        TEXT size_category
        REAL min_cost_usd
        REAL max_cost_usd
        REAL avg_cost_usd
        TEXT manufacturing_process
        REAL confidence
        INTEGER low_confidence
        TEXT reasoning
        DATETIME created_at
    }
    lancedb_chunks {
        INT64 document_id FK
        INT64 chunk_index
        STRING source_file
        STRING chunk_text
        FLOAT32_384 vector
    }

    documents ||--o{ entities : "has"
    documents ||--o{ lancedb_chunks : "chunked into"
    components }o--|| source_file : "keyed by"
```

---

## 7. Project File Tree

```
kearney_assignment/
├── requirements.txt                 ← Python dependencies
├── Dockerfile                       ← Single-image build: api + bundled UI
├── docker-compose.yml               ← api service
├── .env.example                     ← environment variable template
│
├── app/
│   ├── main.py                      ← FastAPI factory + async lifespan (uvicorn entry: app.main:app)
│   ├── config.py                    ← Pydantic Settings (env-driven)
│   │
│   ├── core/
│   │   ├── logging.py               ← Structured logging setup
│   │   └── exceptions.py            ← Exception hierarchy
│   │       (AppError → IngestionError, LLMError,
│   │        EmbeddingError, DatabaseError,
│   │        VectorStoreError, SQLValidationError)
│   │
│   ├── models/
│   │   └── schemas.py               ← Pydantic DTOs
│   │       (QueryRequest, IngestResult, UploadResponse,
│   │        ExtractionResult, ComponentClassification,
│   │        AnalyticalQueryResponse, StructuredQueryResponse,
│   │        SourceReference, ChunkRecord, EntityRecord)
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── health.py            ← GET  /health
│   │       ├── ingestion.py         ← POST /upload
│   │       └── query.py             ← POST /query/structured
│   │                                   POST /query/analytical
│   │
│   ├── db/
│   │   ├── sqlite.py                ← All SQLite ops (context-manager conn)
│   │   └── vector_store.py          ← LanceDB add / search / reset
│   │
│   ├── services/
│   │   ├── llm.py                   ← Groq client (text + vision)
│   │   │                               tenacity retry, LLMError wrapping
│   │   ├── embedding.py             ← SentenceTransformer (lru_cache singleton)
│   │   ├── ingestion.py             ← File orchestrator (Stream A + B)
│   │   ├── rag.py                   ← Retrieve → Reason pipeline
│   │   └── sql_agent.py             ← NL→SQL with self-correction retry
│   │
│   └── utils/
│       └── text.py                  ← chunk_text, extract_json, image_to_data_url
│
└── platform_lancedb/                ← LanceDB data directory (runtime)
    └── chunks.lance/
```

---

## 8. Cross-Cutting Concerns

| Concern | Implementation |
|---|---|
| **Configuration** | `app/config.py` — Pydantic `BaseSettings`, all values overridable via env or `.env` |
| **Logging** | `app/core/logging.py` — structured `%(asctime)s \| %(levelname)s \| %(name)s \| %(message)s` on stdout |
| **Error handling** | Typed exception hierarchy in `app/core/exceptions.py`; services raise typed errors, routes return HTTP errors |
| **Retries** | `tenacity` exponential back-off on all Groq API calls (`LLM_MAX_RETRIES`, `LLM_RETRY_MIN/MAX_WAIT`) |
| **Validation** | Pydantic models at every API boundary; SQL safety via `_validate_sql()` (regex + sqlparse) |
| **Singletons** | `lru_cache(maxsize=1)` for Groq client, embedding model, Docling converter, LanceDB connection |
| **Secrets** | Never hardcoded — read from env / `.env` file via Pydantic Settings |
| **SQL injection** | Only `SELECT` allowed; DML/DDL blocked by regex; parameterised queries throughout |
