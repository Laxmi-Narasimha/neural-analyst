# AI Data Adequacy Agent — Implementation Blueprint

**Version:** 1.0 — research/currentization as of **2025-08-22**

**Purpose:** A single-repo, Cursor-IDE friendly blueprint that lets developers build a local, web-based AI system that **strictly evaluates** whether user-provided data is adequate for an intended AI assistant. The system uses OpenAI (o3 family), Pinecone, LangChain-style orchestration, and a modular multi-agent architecture (8 agents defined — core agents implemented now, others scaffolded for future scale).

---

## Table of Contents

1. Goals & high-level constraints
2. Final agent list (core now + later)
3. End-to-end architecture (components & responsibilities)
4. Data flows and lifecycle
5. Implementation notes (Cursor-friendly repo layout)
6. Libraries, infra & versions (recommended)
7. Detailed agent responsibilities & pseudo-implementations
8. Data-quality checks — exhaustive list + scoring
9. Retrieval & RAG-aware evaluation procedures
10. Prompt templates & OpenAI usage patterns (o3 family)
11. Example API flows (pseudocode + sample curl/py)
12. Report format & generation logic
13. Local dev, Docker, and Cursor instructions
14. Security, privacy & compliance considerations (optional)
15. Scale-up roadmap (agent expansion & external APIs)
16. Appendix: scoring formulas, sample prompts, heuristics

---

# 1. Goals & high-level constraints

* Must validate, strictly and comprehensively, whether a provided knowledge base (files <50MB or embeddings via Pinecone) is **sufficient, consistent, relevant and safe** for a user-specified AI assistant task.
* The flow is interactive: system asks clarifying questions when the user's agent goal is unclear.
* Run **locally** (developer/dev machine) with a web UI accessible on `localhost` (Cursor-friendly). Pinecone remains cloud; OpenAI API used for LLM/embedding actions (o3 models).
* Build full modular architecture (8 agents) now but **implement only core agents** (Orchestrator, Question Generator, Data Ingestion, Quality Analyzer, Validation Results/Report Gen). Scaffold others for later activation.
* Provide deterministic, transparent scoring and remediation guidance — no ambiguous or vague outputs.

# 2. Agent List

## Core agents (implement now)

1. **Interactive Orchestrator Agent (IOA)** — central brain, ReAct loop, routes tasks to agents and compiles results.
2. **Question Generation Agent (QGn)** — generates clarifying questions based on domain and initial data.
3. **Data Ingestion Agent (DI)** — file extraction, chunking, metadata, embeddings, Pinecone indexing.
4. **Quality Analysis Agent (QA)** — runs the exhaustive data-quality checks and simulated query coverage tests.
5. **Validation Results & Report Agent (VR / RG)** — compiles scores, generates remediation plan and the final comprehensive report.

## Additional agents (scaffold / implement later)

6. **Schema Validator Agent (SV)** — validates structured data schemas & relationships (CSV/DB/JSON). Useful for manufacturing sensor data / DMS/CRM systems.
7. **Context Analyzer Agent (CA)** — deep semantic coherence, chunking evaluation, retrieval behavior diagnostics, hallucination risk estimation.
8. **Remediation & Automation Agent (RG)** — auto-suggests transformations, produces scripts or SQL / pandas code to fix issues.

> Note: the later agents are included in the architecture (endpoints, interface contracts), but only stubs are required now. This allows Cursor to scaffold tests and later activate deeper modules.

# 3. End-to-end architecture & responsibilities

```
[User Browser UI] <--> [Frontend (React/Streamlit)] <--> [FastAPI backend]
                                             |
                                  [Orchestrator Agent]
                                   /     |        \
                        [DI] ----/   [QGn]     [QA]
                          |                \    |
                [Document Parsers]         [Pinecone]  <-> [OpenAI o3 models]
                          |                /
                        [Embeddings] <---

Validation Results (VR/RG) -> Report -> Frontend
```

### Components

* **Frontend (React/Streamlit)**: Upload forms, chat interface, results dashboard, file manager. Use Streamlit for faster build or React+Next for production-like UI. Cursor supports either; recommend **React (Vite/Next)** if you will scale UI logic; use **Streamlit** for super-fast prototype.

* **Backend (FastAPI)**: Serves REST + WebSocket endpoints, orchestrates agents, manages uploads, calls OpenAI & Pinecone, houses a small local DB (SQLite/Postgres) for session state.

* **Orchestrator**: Coordinates agents, maintains session memory, enforces constraints & workflows. Implements ReAct loop and rate-limits LLM calls to control cost.

* **Data Ingestion**: PDF/Docx/CSV extraction (PyMuPDF/pdfplumber/docx2txt/pandoc), OCR for scanned docs (Tesseract optional), chunking module (semantic + rule-based), metadata extractor.

* **Vector Store**: Pinecone (namespaces per user/session). Store: chunk text, doc-id, source, chunk-range, timestamp, chunk-hash, extracted metadata.

* **LLM / embeddings**: OpenAI o3 family (o3-research for deep research and question generation; o3-small for cheap embedding judgment tasks — pick exact model names from your key). Adjust temperature per task (0.0-0.3 for scoring/verification; 0.4-0.8 for question generation creativity).

* **Quality Analyzer**: Runs metrics (coverage recall tests, duplicates, staleness, formatting errors, semantic conflicts), synthesizes remediation steps.

* **Report Generator**: Creates multi-section Markdown + JSON report summarizing issues, evidence, remediation steps, and readiness level.

# 4. Data flows & lifecycle

1. **User**: supplies `agent_goal` (free text + structured form) and uploads files (<50MB each) or registers a Pinecone index (embedding ingest path).
2. **Pre-Check**: Quick parse: file types, sizes, detect encryption/protected. Reject malicious/incompatible.
3. **Initial Clarify**: QGn generates initial questions; user answers necessary ones. Orchestrator updates goal schema.
4. **Ingest**: DI extracts text → normalizes → semantic chunking (size=800±200 tokens recommended; overlap=50-150 tokens) → metadata tagging → embed → index into Pinecone namespace.
5. **Sanity-run**: QA runs quick checks (embedding success, language detection, empty/garbled text rate). Return early-fail if >X% chunks bad.
6. **Coverage Testing**: QA generates domain-specific test queries (seeded from goal + domain-template) → executes retrieve (Pinecone top-K) → LLM answers using retrieved context → judge answer vs expectation (LLM-as-judge or heuristics).
7. **Conflict & Consistency Checks**: QA identifies contradictions, duplicates, inconsistent values.
8. **Scoring**: produce per-dimension scores and composite readiness score.
9. **Report**: VR compiles report, includes suggested remediations and prioritized action items.
10. **Iterate**: user provides more data or accepts risks & deploys agent (outside scope).

# 5. Cursor-friendly repo layout (single repo)

```
ai-data-validator/
├─ frontend/               # React or Streamlit app
├─ backend/
│  ├─ app/
│  │  ├─ main.py          # FastAPI server, routes
│  │  ├─ orchestrator.py  # Orchestrator agent logic
│  │  ├─ agents/
│  │  │  ├─ ingestion.py  # DI
│  │  │  ├─ qgen.py       # QGn
│  │  │  ├─ quality.py    # QA
│  │  │  ├─ validation.py # VR
│  │  │  └─ scaffolds/    # stubs for SV, CA, RG
│  │  ├─ utils/
│  │  │  ├─ docparse.py
│  │  │  ├─ chunker.py
│  │  │  └─ embeddings.py
│  │  └─ config.py
│  └─ requirements.txt
├─ docker-compose.yml
├─ README.md
└─ docs/                  # architecture diagrams, prompt templates, runbook
```

Notes for Cursor: make each module small, testable, and instrumented. Cursor engineers prefer single-file components and iterative commits — keep modules <400 lines for easy navigation.

# 6. Libraries, infra & recommended versions

* Python 3.11+
* FastAPI, Uvicorn
* LangChain (latest stable 2025 release) or LlamaIndex (pick one; LangChain recommended for tooling)
* Pinecone SDK (python) — use v5+ (2025 stable) and enable namespaces
* PyMuPDF, pdfplumber, python-docx
* tesseract-ocr for scanned PDFs (optional)
* OpenAI python client (oai) supporting o3 family
* Pandas / Polars for tabular analysis
* Pydantic for schemas & validation
* SQLite (dev) or Postgres (optional) for session state
* Docker, docker-compose
* Frontend: React (Vite/Next) OR Streamlit (if speed prioritized)

# 7. Detailed agent responsibilities & pseudo-implementations

> Below are high-level pseudo-implementations and the *exact responsibilities* of core agents so Cursor can execute without guesswork.

## 7.1 Orchestrator Agent (orchestrator.py)

**Responsibilities:**

* Initialize session, hold `goal_schema` and state
* Sequence agent calls (QGn → DI → QA → VR)
* Implement ReAct loop: Reason (use LLM to decide), Act (call tool), Observe (collect output), reflect & limit steps
* Retry logic & cost control (max LLM calls per session)

**Pseudo**:

```py
class Orchestrator:
    def __init__(self, openai_client, pinecone_client):
        self.openai = openai_client
        self.pinecone = pinecone_client
        self.session_state = {}

    async def run_validation(self, user_goal, files=None, pinecone_namespace=None):
        # Step 0: quick question generation
        clarifying_qs = QGn.generate_questions(user_goal, sample_files=files)
        user_answers = await self.ask_user(clarifying_qs)
        goal = self.fill_goal_schema(user_goal, user_answers)

        # Step 1: ingest
        ingest_result = DI.ingest(files, namespace=pinecone_namespace, goal=goal)
        if ingest_result.failed: return ingest_result.report

        # Step 2: quality checks
        quality_report = QA.run_all_checks(ingest_result, goal)

        # Step 3: validation & remediation plan
        final_report = VR.compile(quality_report, goal)
        return final_report
```

Key points: Orchestrator must enforce a **strict ‘no-deploy’ decision**: if any *critical* failure (e.g., missing legal docs for compliance tasks, or >30% of chunks are GD/garbage), mark readiness as *failed* until user remediates.

## 7.2 Question Generation Agent (qgen.py)

**Responsibilities:**

* Accept free-text user goal + domain selection (car/mfg/real-estate)
* Run a small deep-research prompt using OpenAI o3 to produce 8–15 prioritized clarifying questions, each mapped to failure modes
* Return JSON with questions, priority, rationale, expected evidence

**Prompt strategy:** (see section 10) use an instruction-heavy system prompt and `temperature=0.2` for targeted Qs or `0.6` when more exploration required. Use function-calling schema so the LLM returns structured JSON.

**Pseudo**:

```py
def generate_questions(user_goal, industry=None, sample_text=None):
    prompt = build_qgen_prompt(user_goal, industry, sample_text)
    resp = openai.chat.completions.create(model="o3-research", messages=[...], temperature=0.2)
    return parse_questions(resp)
```

## 7.3 Data Ingestion Agent (ingestion.py)

**Responsibilities:**

* Validate files (size/type)
* Extract text & metadata
* Language detect & OCR as needed
* Chunk text with semantic + rule-based heuristics
* Create chunk IDs, compute chunk-hash, generate embeddings (OpenAI or local), and push to Pinecone with metadata

**Chunking rules:**

* Prefer semantic chunking aligned to headings/paragraphs.
* Default chunk token target: **800 tokens ±200** (for o3-class models). Overlap: **80–150 tokens**.
* Special-case: tables and code blocks — keep intact; add `is_table:true` metadata.

**Pseudo**:

```py
def ingest(files, namespace):
    parsed = []
    for f in files:
        text, meta = parse_file(f)
        chunks = semantic_chunker(text, target_tokens=800, overlap=120)
        for c in chunks:
            emb = embed(c.text)
            pinecone.upsert(namespace, id=c.id, vector=emb, metadata=c.meta)
    return summary_report
```

## 7.4 Quality Analysis Agent (quality.py)

**Responsibilities:**

* Execute the exhaustive data quality checks (see section 8)
* Generate domain-specific simulated queries and perform retrieval/answer evaluation
* Run conflict detection & duplication checks
* Compute numeric scores & confidence
* Rank remediation steps by impact & effort

**High-level checks (run in order)**:

1. **Sanity checks**: embedding success rate, language, document readability score
2. **Duplicates**: near-duplicate detection via cosine similarity (>0.92 => duplicate candidate)
3. **Coverage tests**: auto-generate N seed questions using QGn + user-provided sample Qs; run retrieval (top-k), then LLM answer & judge. Keep K=5.
4. **Consistency/conflict checks**: compare entities (dates, prices, model names) across top retrieved docs; flag discrepancies
5. **Timeliness**: parse dates and compute age; flag domain thresholds (e.g., >12 months stale for car inventory/pricing)
6. **Format & Schema**: detect bad table parses, broken lists, OCR artifacts
7. **Hallucination risk estimation**: use CA stub or heuristics: % of answers requiring external facts vs KB facts

**Pseudo**:

```py
def run_all_checks(ingest_summary, goal):
    checks = {}
    checks['sanity'] = sanity_check(ingest_summary)
    checks['dupes'] = find_duplicates(ingest_summary)
    checks['coverage'] = coverage_test(ingest_summary, goal)
    checks['consistency'] = consistency_checks(ingest_summary)
    checks['timeliness'] = staleness_checks(ingest_summary)
    scores = compute_scores(checks)
    remediation = generate_remediation(checks, scores)
    return {"checks":checks, "scores":scores, "remediation":remediation}
```

**Important:** For coverage testing use LLM-as-judge with conservative thresholds. Ask the model: “Is the answer *fully supported* by the provided context? If not, say which facts are missing.” Use `temperature=0` and a scoring rubric.

## 7.5 Validation Results / Report Agent (validation.py)

**Responsibilities:**

* Aggregate QA outputs into a human-readable Markdown & machine-readable JSON
* Provide prioritized remediation (critical -> high -> medium -> low)
* Provide a final **Readiness Level**: {Blocked, Unsafe, Partially Ready, Ready}
* Provide evidence: KB snippets, chunk-IDs, reasons

**Report skeleton:**

* Executive summary (one-paragraph readiness + top 3 issues)
* Scoring dashboard (table of dimension scores)
* Detailed issues (per-issue evidence + remediation steps)
* Suggested minimal additions (explicit list of docs/tables/fields)
* Optional auto-scripts (small code snippets to fix common issues)

**Pseudo render:**

```py
report_md = render_report_md(quality_report, goal)
report_json = {...}
save_report(session_id, report_md, report_json)
return report_md
```

# 8. Data-quality checks — exhaustive checklist + scoring

Each check must return `(status, score [0-1], evidence, remediation_hint)`.

**A. Sanity & Parsing**

* Language detection success (score 1.0 if matches user language)
* OCR artifact rate (score drops if >5% gibberish tokens)
* Empty chunk rate (score drops if >2% chunks are empty)

**B. Uniqueness / Duplication**

* Duplicate chunk ratio (target <5%). Use cosine similarity threshold >0.92.

**C. Completeness / Coverage**

* Percentage of auto-generated seed questions answered from KB (target >85%)
* Coverage by topic cluster (use embeddings to cluster; ensure top N clusters map to expected topics)

**D. Accuracy & Verification**

* When possible, cross-validate numerical facts against a trusted source or user-supplied master file. Score by mismatch rate.

**E. Consistency**

* Conflicting facts count per topic. Ideally zero critical conflicts.

**F. Timeliness**

* Percent of docs older than domain threshold. E.g., car inventory: >3 months flagged; pricing: >1 month flagged.

**G. Formatting & Chunking**

* Chunk drift score: percent of chunks that lack context at boundaries.

**H. Retrieval Performance**

* Average top-1 relevance (LLM judge) across simulated queries. Target top-1 >= 0.8.

**I. Hallucination Risk**

* Fraction of answers that required external knowledge or where the LLM says "not enough info". Lower is better.

**J. Security & PII**

* Detect PII (names, phone numbers, SSN) in KB — flag and advise redaction if needed.

**Composite Score**

```
composite = 0.2*coverage + 0.15*accuracy + 0.15*consistency + 0.1*timeliness + 0.1*uniqueness + 0.15*retrieval + 0.15*formatting
```

(Weights are configurable per domain.)

Readiness thresholds (example):

* `Ready` if composite >= 0.8 and no critical issues
* `Partially Ready` if 0.6 <= composite < 0.8
* `Unsafe` if 0.4 <= composite < 0.6
* `Blocked` if composite < 0.4 or any critical missing compliance docs

# 9. RAG-aware evaluation procedures (details)

## 9.1 Seed query generation

* QGn generates **N = 30** seed queries by combining:

  * Common domain FAQs (from internal templates)
  * User examples (if supplied)
  * Edge cases (contradictions, long-tail questions)

## 9.2 Retrieval testing

* For each seed query:

  * Retrieve top-K chunks from Pinecone (K=5), with metadata filtering by namespace.
  * Compose an LLM prompt: show retrieved chunks as `CONTEXT` and ask `ANSWER`.
  * Use `system` role instructing model to answer only from context and to respond `I DON'T KNOW` when insufficient.

## 9.3 LLM-as-Judge

* After generating the answer, call the LLM again (or the same call with a `judge` subprompt) to decide if the answer is:

  * **Fully supported** (2 points)
  * **Partially supported** (1 point)
  * **Unsupported / hallucinated** (0 points)
* Use temperature=0 for deterministic scoring.

## 9.4 Conflict detection algorithm

* Extract named entities and facts (dates, numeric specs) from top-K chunks using an entity-extractor prompt.
* For each fact: compare distributions; if variance > domain-specific threshold, flag conflict.

## 9.5 Hallucination audit

* For each simulated query where the answer is unsupported, log whether the LLM invented facts.
* Provide examples in the report and indicate potential business impact.

# 10. Prompt templates & OpenAI usage patterns

> Use these exact templates (Cursor can paste them into code). Use function-calling where possible.

## 10.1 System-level guidelines

* **Research / QGen tasks:** `model=o3-research`, temperature=0.2–0.4
* **Deterministic verification / scoring:** `model=o3-small` (or o3-discuss with temperature=0), temperature=0
* **Answer generation (user-facing):** `model=o3-chat` with temperature=0.0–0.2 (for factual responses) or 0.3–0.6 for more conversational tone.
* Always include `max_tokens` bounds and `top_p` defaults. Use `presence_penalty` low.

## 10.2 Question Generation Prompt (QGn)

```
SYSTEM: You are a question generation assistant for data adequacy checks. You will receive:
 - goal: short text describing the user's target agent
 - industry: [car / manufacturing / real-estate / general]
 - sample_context(optional): a short text sample or list of headings from provided KB

INSTRUCTIONS:
 - Produce 8-15 clarifying questions that will help determine whether the available data is sufficient for the stated goal.
 - For each question, include: id, text, priority (critical/high/medium/low), failure_mode_tag (e.g. MISSING_COVERAGE, TIMELINESS, INCONSISTENT), expected_evidence (what doc/chunk would satisfy the question).
 - Return ONLY valid JSON with key `questions`.

USER: {goal}

RESPONSE FORMAT (JSON):
{ "questions": [ {"id": "q1","text":"...","priority":"critical","failure_mode":"MISSING_COVERAGE","expected_evidence":"vehicle_spec sheet or warranty doc"}, ... ] }
```

**Usage:** parse JSON and show to user in the UI for answers.

## 10.3 Coverage Test / Answering Prompt (QA)

```
SYSTEM: You are an objective fact-extraction assistant. You will be given CONTEXT pieces (numbered). Answer the QUESTION strictly using only the context. If the context does not answer the question, reply "I DON'T KNOW" and list missing facts.

CONTEXT:
[1] {chunk1}
[2] {chunk2}
...
QUESTION: {seed_question}

RESPONSE:
- Answer: <short answer>
- EvidenceIDs: [1,2]
- Confidence: <0-1>
- MissingFacts: []
```

**Judge:** After the above, send a `judge` prompt:

```
SYSTEM: You are a judge. Given the Answer and the provided context, label: FULLY_SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED. Explain briefly (1-2 sentences) and give a numeric support_score (0-1).
```

## 10.4 Hallucination Audit Prompt (for CA / QA)

```
SYSTEM: Given the answered Q&A with evidence IDs, determine if any factual statements in the answer are not present in the evidence. For each unsupported statement, quote it, and say why it is unsupported. Classify risk impact: LOW / MEDIUM / HIGH.
```

## 10.5 Final report generation instruction (VR)

```
SYSTEM: You are a meticulous technical reporter. Using structured inputs (quality checks, scores, remediation hints), create:
 - Executive Summary (2-3 sentences)
 - Scoring table (CSV/markdown)
 - Top 10 issues with evidence and remediation steps (prioritized)
 - Minimal dataset additions required to fix critical issues
 - Suggested code snippets for automated remediation (if trivial)

Return a Markdown document and a JSON with the same content.
```

# 11. Example API flows & pseudocode

## 11.1 FastAPI endpoints (suggested)

```
POST /api/validate
  body: { goal: str, industry: str, files: multipart[] } -> starts session
GET /api/session/{id}/questions -> returns QGn output
POST /api/session/{id}/answers -> user answers to questions
POST /api/session/{id}/ingest -> starts DI
GET /api/session/{id}/report -> returns latest VR report (md + json)
```

## 11.2 Minimal Python call (seed)

```py
# create embeddings and upsert
from openai import OpenAI
import pinecone

openai = OpenAI(api_key=OPENAI_KEY)
pinecone.init(api_key=PINECONE_KEY)
idx = pinecone.Index('ai-kb')

text = "...chunk text..."
emb = openai.embeddings.create(model='o3-embed', input=text)['data'][0]['embedding']
idx.upsert([(chunk_id, emb, {'source': 'file.pdf','chunk_index':0})])
```

## 11.3 Coverage test flow pseudocode

```py
questions = QGn.generate_questions(goal)
for q in questions:
    topk = pinecone.query(q_embedding, top_k=5)
    context = fetch_chunks(topk)
    answer = openai.chat.create(model='o3-chat', messages=[sys_prompt, {user: compose_prompt(context,q)}])
    judge = openai.chat.create(model='o3-eval', messages=[sys_judge_prompt, ...])
    record result
```

# 12. Report format & generation logic (full example)

**File:** `report_{session_id}.md`

**Header:** Project name, date, author (Agent), readiness summary

**Sections:**

* Executive summary (1 para)
* Readiness label & composite score
* Scoring table: metric, score, threshold, pass/fail
* Top critical issues (each: title, description, evidence snippet with chunk-id(s), priority, remediation steps — with estimated effort)
* Medium issues (similar)
* Minor/optional suggestions
* Minimal dataset additions (explicit filenames/fields/examples to add)
* Example automated fixes (code blocks)
* Appendices: all seed questions & results, raw metrics, list of chunk hashes

Provide both `report.md` and `report.json`. UI will show `report.md` and allow download.

# 13. Local development, Docker & Cursor instructions

* Provide `.env` example with `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENV`, `PINECONE_INDEX`.
* Use `docker-compose` to start `backend` and `frontend` containers. Pinecone is cloud-based; no local Pinecone container.

**Sample docker-compose** (simplified):

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports: ['8000:8000']
    env_file: .env
  frontend:
    build: ./frontend
    ports: ['3000:3000']
```

**Cursor tips:**

* Create small commits per agent. Cursor's vibe coding flow prefers micro-commits and immediate live previews. Use `pyproject.toml` or `requirements.txt` for reproducibility.
* Provide a single `run_dev.sh` which runs uvicorn and starts Streamlit / React dev server.

# 14. Security, privacy & compliance (optional)

* PII detection: run regex + LLM PII detector on each chunk; flag & require user acknowledgment before indexing.
* Data retention policy: by default, *do not persist embeddings beyond the session* unless user opts in. If needed, store with short TTL in Pinecone (or maintain separate user namespace and clear).
* For HIPAA/GDPR-sensitive deployments, add encryption & on-premise storage; disable cloud vector stores.

# 15. Scale-up roadmap

**Stage 1 (current):** Core agents implemented; robust report generation; run locally.
**Stage 2:** Implement Schema Validator (SV) and Context Analyzer (CA) for deeper semantics & tabular validation.
**Stage 3:** Add Remediation Agent (RG) that can produce runnable code to fix common issues (cleaning scripts, entity resolution) and integrate open-source specialized tools for embeddings if needed.
**Stage 4:** Add connectors for cloud data sources (S3, GCS, DBs), advanced analytics dashboards, and MLOps integration.

**External APIs to add on scale-up:**

* Free/low-cost web search APIs for external fact-checking (e.g., Bing Search API or open-source scrapers)
* Domain-specific data sources (NHTSA VIN DB for automotive, local MLS APIs for real estate)
* Advanced on-prem embedding solutions if cost is a problem (e.g., E5 embeddings, sentence-transformers) — switch to hybrid embedding strategy.

# 16. Appendix: scoring formulas, heuristics & sample prompts

**Composite formula (configurable weights):**

```
composite = sum(w[i]*metric[i]) / sum(w)
```

Where `metric[i]` are normalized to \[0,1].

**Duplicate detection heuristic:** cosine > 0.92 → mark duplicate; 0.85-0.92 → near-duplicate

**Chunk size guidance:** 800 token target for o3, overlap 80-150 tokens

**Seed question counts:** 30 general + 10 user-supplied (if provided)

**LLM call quotas:** default session max LLM calls 50. Adjust keyboard to control cost.

---

## Final notes for Cursor AI

* This document provides everything Cursor needs to bootstrap the project: modules, endpoints, prompts, constraints, scoring, and a complete report format.
* **Actionable next steps for Cursor engineers:**

  1. Clone repo skeleton and implement DI & basic ingestion pipeline (file parse + chunking + embeddings to Pinecone).
  2. Implement QGn and simple ReAct Orchestrator with mocked QA to close the loop.
  3. Implement QA checks (sanity, duplication, coverage tests) and VR report generation.
  4. Iterate on UI (Streamlit quick or React) and test with real documents in car/manufacturing/real estate.

---

*End of blueprint.*

---

## 17. Pinecone configuration modes and BYO embeddings

This project must support two deployment modes for vector storage and also allow users to provide embeddings directly.

### A) Project-managed Pinecone (default)

* __Who provides keys__: The app owner configures `.env` with `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME`.
* __Namespace strategy__: create a per-session/user namespace, e.g. `session_{uuid}`.
* __Index__: create (if missing) with `dimension=3072`, `metric=cosine` to match OpenAI `text-embedding-3-large`.
* __Pros__: zero friction for end users; consistent infra and metrics.
* __Cons__: all data indexed under the app owner’s Pinecone account.

### B) User-provided Pinecone (Bring Your Own Pinecone)

UI/API accepts optional overrides per session:

```json
{
  "pinecone": {
    "api_key": "<user_key>",
    "environment": "<region>",
    "index": "<name>",
    "namespace": "<optional namespace>"
  }
}
```

Validation rules:

* If overrides are present, use them instead of `.env`.
* Verify index exists and `dimension` matches the embedding model (default 3072). If mismatched, return a clear 400 error with remediation tips: either choose an index with matching dimension or switch embedding model.
* Never persist user keys server-side beyond the session; keep in memory only.

### C) BYO embeddings (direct embedding upload)

Allow advanced users to supply precomputed embeddings instead of raw files:

API contract (multipart or JSON):

```json
{
  "vectors": [
    {
      "id": "doc1_chunk_0001",
      "values": [0.0123, -0.0345, ...],
      "metadata": {"text": "...", "source": "doc1.pdf", "chunk_index": 1}
    }
  ],
  "dimension": 3072,
  "namespace": "optional"
}
```

Server behavior:

* Validate `dimension` and ensure `len(values)==dimension` for all vectors.
* If `namespace` missing, generate one per session.
* Upsert to Pinecone using the same index selection rules as above (owner Pinecone or user-provided Pinecone).
* Skip OpenAI embedding step for these vectors but still run all quality checks and coverage tests against the supplied vectors.

### D) Model–index compatibility matrix

* `text-embedding-3-large` → 3072 dims (default here)
* `text-embedding-3-small` → 1536 dims

If the user selects a different embedding model, update `config.MODELS["embedding"]` and either:

* Use an index with matching dimension; or
* Let the backend create a new index automatically with the correct dimension.

### E) Minimal UI adjustments (Streamlit)

* Toggle: "Use my Pinecone account" → shows fields (API key, environment, index, optional namespace).
* Advanced accordion: "Upload precomputed embeddings (JSON)" with file picker and schema preview; validate on client before POST.
* Display active mode and target index/dimension on the ingest screen for transparency.

