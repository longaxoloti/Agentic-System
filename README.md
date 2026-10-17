# 🤖 Tesla — Self-Hosted Agentic AI System

<p align="center">
  <strong>A local-first, multi-model AI agent system — orchestrated by LangGraph, delivered through Telegram.</strong>
</p>

<p align="center">
  <a href="https://github.com/longaxoloti/Multi_Agent/actions"><img src="https://img.shields.io/github/actions/workflow/status/longaxoloti/Multi_Agent/ci.yml?branch=main&style=for-the-badge" alt="CI Status"></a>
  <a href="https://github.com/longaxoloti/Multi_Agent/releases"><img src="https://img.shields.io/github/v/release/longaxoloti/Multi_Agent?include_prereleases&style=for-the-badge" alt="GitHub Release"></a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="MIT License"></a>
</p>

**Multi Agent** is a _self-hosted, multi-model AI agent system_ built with [LangGraph](https://github.com/langchain-ai/langgraph). It runs entirely on local hardware via [Ollama](https://ollama.com/) with optional cloud LLM fallback (Gemini, OpenAI, Anthropic).

You interact with your agent through **Telegram**. The Orchestrator analyzes your intent, plans multi-step work across specialized worker models (Researcher, Reasoner), and synthesizes coherent responses — all while managing RAM on a single machine by loading one model at a time.

[Getting Started](#getting-started) · [Architecture](#architecture) · [Features](#features) · [Model Routing](#model-routing) · [Research Mode](#research-mode) · [Health Check](#health-check) · [Docker](#run-with-docker) · [Presentation (VI)](PRESENTATION_MULTI_AGENT_VI.md)

---

## Highlights

- **[LangGraph Workflow](#architecture)** — Stateful graph pipeline with conditional routing and iterative worker loop.
- **[Multi-Model Routing](#model-routing)** — Route each task type to the optimal model: Ollama, Gemini, OpenAI, or Anthropic.
- **[RAM-Aware Model Swapping](#features)** — Automatically unloads the previous model before loading the next; context is serialized to JSON for seamless handoff.
- **[Smart Web Research](#research-mode)** — Chrome CDP (primary) with Camoufox MCP fallback; human-like mouse movements, smooth scrolling, and Bezier-curve paths to bypass bot detection.
- **[Persistent Knowledge & RAG](#knowledge-commands)** — PostgreSQL + pgvector for semantic search across your personal knowledge base.
- **[Telegram Interface](#getting-started)** — Single-instance locking, retry with backoff, scheduled daily reports.
- **[Airflow Reporting Pipeline](#airflow-daily-reporting)** — Automated daily briefing DAG via Apache Airflow.

---

## Architecture

The system is organised into five layers:

```
┌─────────────────────────────────────────────┐
│           Presentation Layer                │
│    Telegram Bot  ·  CLI (tesla / Typer)     │
├─────────────────────────────────────────────┤
│           Orchestration Layer               │
│  request_router → orchestrator → workers    │
│       → synthesizer → orchestrator          │
├────────────────────────────────────────────-┤
│             Service Layer                   │
│  KnowledgeService · UserProfileService      │
│  UnifiedMemoryService · SecurityService     │
├─────────────────────────────────────────────┤
│             Storage Layer                   │
│  PostgreSQL (4 schemas) + pgvector HNSW     │
├─────────────────────────────────────────────┤
│              Tools Layer                    │
│  Chrome CDP · Camoufox MCP · Crawl4AI       │
└─────────────────────────────────────────────┘
```

### Workflow Pipeline

```
Request Router → Orchestrate (PLAN)
               → [Research | Coding | Reasoning | Briefing]
               → Orchestrate (PROGRESS) → ... (loop)
               → Synthesize → END
```

The **Orchestrator** runs in two phases:
- **PLAN**     — Identifies intent, topic, and search query; generates `plan_steps` and `routing_decision`.
- **PROGRESS** — Monitors worker results; decides whether to route to the next worker or synthesize.

### Key Source Files

| File | Role |
|---|---|
| `graph/workflow.py` | StateGraph definition and route logic |
| `graph/state.py` | `AgentState` schema |
| `graph/nodes/*` | Worker node implementations |
| `graph/llm_router.py` | Provider routing per task type |
| `main/config.py` | Central env configuration |
| `telegram_bot/bot.py` | Message handling + daily schedule |
| `storage/models.py` | ORM models for 4 DB schemas |
| `storage/knowledge_service.py` | Knowledge save/search/list/delete |
| `memory/memory_manager.py` | Hybrid memory (cache + DB) |
| `airflow/dags/` | Daily reporting DAG |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/longaxoloti/Multi_Agent.git
cd Multi_Agent
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate multi-agent
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `TELEGRAM_BOT_TOKEN` — Your bot token from [@BotFather](https://t.me/BotFather)
- `TELEGRAM_USER_ID` — Your Telegram user ID
- API keys for cloud LLM providers *(optional)*
- Database URL *(optional — defaults to local Postgres)*

### 4. Run the Agent

```bash
# Start Telegram bot only (default)
./main/run.sh

# Start full stack: bot + Airflow + crawler
./main/run.sh stack

# Start Airflow scheduler only
./main/run.sh airflow
```

### 5. Talk to Your Agent

Open Telegram, find your bot, and send `/start`.

---

## Run with Docker

```bash
cp .env.example .env   # or create .env manually

# Build and start bot + Postgres
docker compose up -d --build

# Follow logs
docker compose logs -f multi-agent
```

Default container command runs the Telegram bot via `python -m main.main`.

---

## Health Check

Run a prestart validation to verify your environment, provider routes, and Ollama models:

```bash
python scripts/health_check.py --prestart
```

---

## Model Routing

Override which provider handles each task type via environment variables:

```env
MODEL_ORCHESTRATOR   # Orchestration and planning
MODEL_RESEARCH       # Research tasks
MODEL_CODE           # Coding tasks
MODEL_CHAT           # Direct chat / reasoning
```

Supported providers: `ollama` · `gemini` · `openai` · `anthropic`

The router includes a **fallback mechanism** — if the primary provider is unavailable, it automatically switches to the next configured provider.

---

## Features

### Knowledge Commands

Interact with your personal knowledge base directly in Telegram chat:

| Command | Action |
|---|---|
| `/save <category> <content>` | Save a knowledge entry |
| `/get <id>` | Retrieve a record by ID |
| `/search <query>` | Semantic search via embedding |
| `/list` | List recent entries |
| `/delete <id>` | Delete a record |
| `/profile` | View, search, or ingest your profile |

### Memory & Context Management

- **In-memory cache** for fast conversation lookup during a session.
- **Persistent flush** to `system.conversation_sessions` table.
- Configurable conversation history window (`MAX_CONVERSATION_HISTORY`).
- **Session serialization to JSON** for seamless context handoff between model swaps.

### RAG & Knowledge Persistence

- `KnowledgeService` generates embeddings and stores them in **PostgreSQL/pgvector**.
- Deduplication for `web_news` entries (by source + content hash).
- Top-k semantic retrieval by vector distance.
- Metadata support for filtering and source attribution.

### Storage Architecture (4 Schemas)

| Schema | Purpose |
|---|---|
| `system` | Audit logs, conversation sessions |
| `profile` | User facts and profile embeddings |
| `knowledge` | User knowledge, entities, memory embeddings |
| `security` | Policies and secret references (deny-by-default) |

### Airflow Daily Reporting

- DAG `daily_user_knowledge_report` runs on a cron schedule.
- Triggers a research workflow for the configured data interval.
- Sends a summarized report to your Telegram.
- Deduplicates and persists `web_news` sources.

---

## Research Mode

### Parallel Search MCP (Optional)

Set `RESEARCH_SEARCH_PROVIDER=parallel` in `.env` to use
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) for
free web search and page fetching without an account or API key. Anonymous search
uses Fast mode. Research queries and selected URLs are sent to
`https://search.parallel.ai/mcp`; no local browser server is needed for this path.
The research worker passes search and fetched excerpts, with source URLs, to its
existing synthesis step.

Keep `CAMOUFOX_ENABLED=true`, which remains the interactive research enable switch
for both backends. `CAMOUFOX_ENABLED=false` or `RESEARCH_SEARCH_PROVIDER=none`
disables interactive web research. `RESEARCH_SEARCH_PROVIDER=browser` (the default)
keeps the Chrome/Camoufox path and its existing settings. Daily fixed-source reports
continue to use their configured URLs and crawler.

`PARALLEL_MCP_TIMEOUT_SECONDS` defaults to 60 for the complete search/fetch operation.
Parallel errors, timeouts and service rate limits produce an unavailable-context
message for synthesis; they never trigger a switch to another search provider.
The anonymous service has usage limits. Update dependencies when enabling this
backend (`mcp>=1.30,<2` and `httpx>=0.27,<1`); existing `main/config.py` files need no
new settings.

### Chrome CDP (Primary)

The agent defaults to Chrome CDP — controlling your real, daily-use browser. This approach bypasses bot detection by using your real IP, established cookies/account history, and injecting human-like browser interactions:

- Smooth Bezier-curve mouse paths
- Smooth scrolling
- Injected custom visual cursor

**Setup:** Completely close Google Chrome before starting the agent. It will open and control a new tab autonomously.

### Camoufox MCP (Fallback)

If Chrome CDP is unavailable, the agent falls back to `camofox-mcp`.

**Required:**
1. `camofox-browser` reachable at `/health` on port `9377` (default).
2. MCP transport configured: `stdio` (default, spawned per session) or `http` (background process via `scripts/start_camoufox.sh`).

**MCP Environment Variables:**

```env
# Browser server URL
CAMOFOX_URL=http://127.0.0.1:9377
CAMOUFOX_API_URL=http://127.0.0.1:9377

# MCP transport: stdio | http
CAMOFOX_MCP_TRANSPORT=stdio

# For stdio spawn
CAMOFOX_MCP_COMMAND=npx
CAMOFOX_MCP_ARGS=-y camofox-mcp@latest

# For HTTP mode
CAMOFOX_MCP_URL=http://127.0.0.1:3000/mcp

# Optional auth
CAMOFOX_API_KEY=

# MCP client runtime
CAMOFOX_MCP_TIMEOUT_MS=30000
CAMOFOX_MCP_MAX_RETRIES=2
CAMOFOX_MCP_RETRY_BACKOFF_SECONDS=1.0
```

**Verify MCP Setup:**

```bash
# Check browser health
curl -fsS http://127.0.0.1:9377/health

# Run e2e MCP smoke test (server_status → create_tab → navigate/snapshot → close_tab)
python scripts/e2e_mcp_workflow_smoke.py
```

**Startup Behavior:**
- `stdio` mode: `start_camoufox.sh` starts only `camofox-browser`; MCP server is spawned per Python session.
- `http` mode: `start_camoufox.sh` starts both `camofox-browser` and `camofox-mcp`, with PID/log files in `data/logs`.

---

## CLI (tesla)

The `tesla` CLI manages the agent lifecycle:

```bash
tesla init    # Initialize configuration
tesla start   # Start the agent
tesla stop    # Stop the agent
```

Install with: `pip install -e .`

---

## Tech Stack

| Technology | Role |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful agent orchestration |
| [LangChain](https://www.langchain.com/) | Unified LLM provider interface |
| [Ollama](https://ollama.com/) | Local model serving (self-hosted) |
| [PostgreSQL](https://www.postgresql.org/) + [pgvector](https://github.com/pgvector/pgvector) | Relational + vector storage (HNSW) |
| [Apache Airflow](https://airflow.apache.org/) | Scheduled reporting pipeline |
| [python-telegram-bot](https://python-telegram-bot.org/) | Telegram interface |
| [Crawl4AI](https://github.com/unclecode/crawl4ai) | Web content extraction |
| [Alembic](https://alembic.sqlalchemy.org/) | Database schema migrations |
| [Docker](https://www.docker.com/) | Infrastructure (DB + services) |

---

## Development

### Running Tests

```bash
pytest tests/
```

### E2E Smoke Tests

```bash
# Multi-agent workflow: CHAT, RESEARCH, CODING cases
python scripts/e2e_workflow_smoke.py

# MCP browser pipeline
python scripts/e2e_mcp_workflow_smoke.py
```

### Project Structure

```
Multi_Agent/
├── graph/          # LangGraph workflow, state, nodes, router
├── main/           # Entry point, config, CLI, run script
├── telegram_bot/   # Bot handlers and scheduler
├── storage/        # ORM models and service layer
├── memory/         # Hybrid memory manager
├── tools/          # Browser automation, crawlers
├── rag/            # Retrieval-augmented generation utilities
├── pipelines/      # Data ingestion pipelines
├── airflow/        # Airflow DAGs
├── scripts/        # Health checks and smoke tests
├── tests/          # Test suite
├── infra/          # Docker Compose for DB services
├── workspace/      # Role-specific system prompt files
└── data/           # Runtime data, logs, session files
```

---

## License

[MIT](LICENSE)
