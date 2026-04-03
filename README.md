# Assignment 3 — Autonomous Multi-Doc Financial Analyst

Python implementation for Tasks **A–E**: LangChain ReAct prompt, LangGraph router, relevance grader, query rewriter, and cited generator. **Default LLM is Ollama** (`LLM_PROVIDER=ollama`).

## Prerequisites

- Python 3.11+ recommended (3.12/3.13 usually fine).
- [Ollama](https://ollama.com) running locally with a capable model, e.g. `ollama pull llama3.2`
- PDFs in `data/` (filenames must match `config.FILES` or add your own keys)

### Data files

Place:

- Apple: `data/FY24_Q4_Consolidated_Financial_Statements.pdf` (or equivalent FY24 consolidated statements from Apple IR / earnings materials).
- Tesla: `data/tsla-20241231-gen.pdf` — [Tesla 2024 10-K PDF](https://ir.tesla.com/_flysystem/s3/sec/000162828025003063/tsla-20241231-gen.pdf)

## Setup

```bash
cd assignment3
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Fully automated setup (recommended)

From `assignment3/`, after `pip install -r requirements.txt`:

```bash
python bootstrap.py
```

This will: create `.env` from `.env.example` if missing, download both course PDFs into `data/`, run `ollama pull` for `OLLAMA_MODEL` when the `ollama` CLI exists, and run `build_rag.py`.

Ollama’s daemon must be reachable at `OLLAMA_BASE_URL` (default `http://localhost:11434`) for `ollama pull` and for agents to run.

## Build vector stores (manual)

Default **chunk_size=2000** and **chunk_overlap=200** (override with `CHUNK_SIZE` / `CHUNK_OVERLAP`).

```bash
python build_rag.py
```

Indices are stored under `chroma_db/<EMBEDDING_TAG>/<apple|tesla>/`. Change `EMBEDDING_MODEL` and `EMBEDDING_TAG` to compare embedding models without overwriting indexes.

## Run agents

- **LangGraph pipeline** (router → retrieve → grade → rewrite loop → generate):

  ```bash
  python -c "from langgraph_agent import run_graph_agent; print(run_graph_agent('What was Apple total net sales in 2024?'))"
  ```

- **Legacy LangChain ReAct** (`Task A` template in `langgraph_agent.REACT_TEMPLATE`):

  ```bash
  EVAL_MODE=LEGACY python evaluator.py
  ```

## Evaluation harness

```bash
EVAL_MODE=GRAPH python evaluator.py
```

Logs are written to `evaluation_log_YYYYMMDD_HHMM.txt`.

### Automated / CI-style run

- Human-readable log + JSON summary line (prefix `__EVAL_JSON__` for easy `grep`/`jq`):

  ```bash
  EVAL_MODE=GRAPH python evaluator.py --ci
  ```

- Quiet run (human-readable progress goes to a buffer; **stderr** shows per-case ticks), JSON only, **exit 1** if any case fails:

  ```bash
  EVAL_MODE=GRAPH python evaluator.py --ci --json-only --strict; echo $?
  ```

  The full run can take **many minutes** (each case hits Ollama several times). It is not stuck if you see no stdout until the final `__EVAL_JSON__` line.

- Custom log file while keeping console output:

  ```bash
  python evaluator.py --log-file my_run.txt
  ```

## Task mapping (code)

| Task | Location |
|------|----------|
| A — ReAct prompt | `REACT_TEMPLATE` + `run_legacy_agent` |
| B — Router | `ROUTER_SYSTEM`, `retrieve_node` |
| C — Grader | `GRADER_SYSTEM`, `grade_documents_node` |
| D — Rewriter | `REWRITER_SYSTEM`, `rewrite_node` |
| E — Generator | `GENERATOR_SYSTEM`, `generate_node` |

## Report reminders (not auto-generated)

- Compare **at least two** embedding models (rebuild with different `EMBEDDING_MODEL` + `EMBEDDING_TAG`).
- Contrast **LangGraph vs LangChain** (routing, grading, rewrite loop vs ReAct tool loop).
- Discuss **chunk size** vs large tables: precision vs completeness.

`report.pdf` is your write-up for submission.
