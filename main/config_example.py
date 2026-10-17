# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
MEMORY_DIR = DATA_DIR / "memory"
LOGS_DIR = DATA_DIR / "logs"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

for d in [DATA_DIR, SESSIONS_DIR, MEMORY_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _resolve_model_ref(value: str) -> str:
    if not value:
        return value
    direct = os.getenv(value)
    if direct is not None and direct.strip():
        return direct.strip()
    normalized = value.strip().upper().replace(".", "_")
    normalized_ref = os.getenv(normalized)
    if normalized_ref is not None and normalized_ref.strip():
        return normalized_ref.strip()
    return value.strip()

# API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Local Host
OLLAMA_ENABLED = _env_bool("OLLAMA_ENABLED", True)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "{your_ollama_base_url}")

# Set up roles here
YOUR_MODEL = _resolve_model_ref(os.getenv("YOUR_MODEL_ALIAS"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))

# Scheduler
BRIEFING_HOUR = int(os.getenv("BRIEFING_HOUR", "7"))
BRIEFING_MINUTE = int(os.getenv("BRIEFING_MINUTE", "0"))
AIRFLOW_DAILY_REPORT_CRON = os.getenv("AIRFLOW_DAILY_REPORT_CRON", "0 7,19 * * *")
AIRFLOW_TIMEZONE = os.getenv("AIRFLOW_TIMEZONE", "Asia/Saigon")

# Database
TRUSTED_DB_URL = os.getenv(
    "TRUSTED_DB_URL",
    "postgresql+psycopg://agent:agent@localhost:5432/agent_ai",
)
TRUSTED_DB_REQUIRED = _env_bool("TRUSTED_DB_REQUIRED", False)

# Knowledge persistence (PostgreSQL + pgvector only)
KNOWLEDGE_DB_ENABLED = _env_bool("KNOWLEDGE_DB_ENABLED", True)
KNOWLEDGE_DB_REQUIRED = _env_bool("KNOWLEDGE_DB_REQUIRED", False)
KNOWLEDGE_ALLOW_NATURAL_LANGUAGE_COMMANDS = _env_bool(
    "KNOWLEDGE_ALLOW_NATURAL_LANGUAGE_COMMANDS", True
)
KNOWLEDGE_MAX_SEARCH_RESULTS = int(os.getenv("KNOWLEDGE_MAX_SEARCH_RESULTS", "5"))
KNOWLEDGE_MAX_CONTENT_CHARS = int(os.getenv("KNOWLEDGE_MAX_CONTENT_CHARS", "12000"))
KNOWLEDGE_MAX_RECENT_ITEMS = int(os.getenv("KNOWLEDGE_MAX_RECENT_ITEMS", "10"))
KNOWLEDGE_PGVECTOR_REQUIRED = _env_bool("KNOWLEDGE_PGVECTOR_REQUIRED", True)

KNOWLEDGE_EMBEDDING_PROVIDER = os.getenv("KNOWLEDGE_EMBEDDING_PROVIDER", "ollama").strip().lower()
KNOWLEDGE_EMBEDDING_MODEL = os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "bge-m3").strip()
KNOWLEDGE_EMBEDDING_DIMS = int(os.getenv("KNOWLEDGE_EMBEDDING_DIMS", "1024"))

# Multi-schema database settings
DB_VECTOR_INDEX_TYPE = os.getenv("DB_VECTOR_INDEX_TYPE", "hnsw").strip().lower()  # hnsw
DB_SECURITY_BACKEND = os.getenv("DB_SECURITY_BACKEND", "env_var").strip().lower()  # env_var, vault, encrypted_file
DB_SKILL_CHUNK_SIZE = int(os.getenv("DB_SKILL_CHUNK_SIZE", "800"))
DB_SKILL_CHUNK_OVERLAP = int(os.getenv("DB_SKILL_CHUNK_OVERLAP", "100"))
DB_RETRIEVAL_PRIORITY = [
    s.strip().lower()
    for s in os.getenv(
        "DB_RETRIEVAL_PRIORITY", "knowledge,profile,system"
    ).split(",")
    if s.strip()
]
DB_UNIFIED_MEMORY_MODE = _env_bool("DB_UNIFIED_MEMORY_MODE", True)
DB_CONVERSATION_PERSIST = _env_bool("DB_CONVERSATION_PERSIST", True)

# Airflow runtime policy
AIRFLOW_REPORT_CATCHUP = _env_bool("AIRFLOW_REPORT_CATCHUP", True)
AIRFLOW_REPORT_RETRIES = int(os.getenv("AIRFLOW_REPORT_RETRIES", "3"))
AIRFLOW_REPORT_RETRY_DELAY_MINUTES = int(os.getenv("AIRFLOW_REPORT_RETRY_DELAY_MINUTES", "10"))
AIRFLOW_REPORT_DAGRUN_TIMEOUT_MINUTES = int(os.getenv("AIRFLOW_REPORT_DAGRUN_TIMEOUT_MINUTES", "60"))
AIRFLOW_REPORT_MAX_ACTIVE_RUNS = int(os.getenv("AIRFLOW_REPORT_MAX_ACTIVE_RUNS", "1"))
AIRFLOW_REPORT_CHAT_ID = os.getenv(
    "AIRFLOW_REPORT_CHAT_ID",
    str(TELEGRAM_USER_ID) if TELEGRAM_USER_ID else "",
).strip()
AIRFLOW_REPORT_CATEGORIES = [
    item.strip().lower()
    for item in os.getenv("AIRFLOW_REPORT_CATEGORIES", "").split(",")
    if item.strip()
]

# Crawler runtime
# Research reads these directly from the environment so existing config.py files
# continue to work: RESEARCH_SEARCH_PROVIDER=browser|parallel|none (default browser)
# and PARALLEL_MCP_TIMEOUT_SECONDS=60 (whole search/fetch operation).
CRAWLER_POLL_SECONDS = int(os.getenv("CRAWLER_POLL_SECONDS", "900"))

# Camoufox Server
CAMOUFOX_ENABLED = _env_bool("CAMOUFOX_ENABLED", True)
CAMOFOX_URL = os.getenv("CAMOFOX_URL", os.getenv("CAMOUFOX_API_URL", "http://127.0.0.1:9377"))
CAMOUFOX_API_URL = os.getenv("CAMOUFOX_API_URL", CAMOFOX_URL)
CAMOUFOX_STRICT_ONLY = _env_bool("CAMOUFOX_STRICT_ONLY", True)

# Chrome CDP
CHROME_CDP_ENABLED = _env_bool("CHROME_CDP_ENABLED", True)
CHROME_CDP_PORT = int(os.getenv("CHROME_CDP_PORT", "9222"))

CAMOFOX_MCP_TRANSPORT = os.getenv("CAMOFOX_MCP_TRANSPORT", "stdio").strip().lower()
if CAMOFOX_MCP_TRANSPORT not in {"stdio", "http"}:
    CAMOFOX_MCP_TRANSPORT = "stdio"

CAMOFOX_MCP_COMMAND = os.getenv("CAMOFOX_MCP_COMMAND", "npx").strip() or "npx"
_camofox_mcp_args_raw = os.getenv("CAMOFOX_MCP_ARGS", "-y camofox-mcp@latest")
CAMOFOX_MCP_ARGS = shlex.split(_camofox_mcp_args_raw)
CAMOFOX_MCP_URL = os.getenv("CAMOFOX_MCP_URL", "http://127.0.0.1:3000/mcp")
CAMOFOX_MCP_TIMEOUT_MS = int(os.getenv("CAMOFOX_MCP_TIMEOUT_MS", "30000"))
CAMOFOX_MCP_MAX_RETRIES = int(os.getenv("CAMOFOX_MCP_MAX_RETRIES", "2"))
CAMOFOX_MCP_RETRY_BACKOFF_SECONDS = float(os.getenv("CAMOFOX_MCP_RETRY_BACKOFF_SECONDS", "1.0"))
CAMOFOX_API_KEY = os.getenv("CAMOFOX_API_KEY", "")
CAMOFOX_AUTH_REQUIRED = _env_bool("CAMOFOX_AUTH_REQUIRED", True)
CAMOFOX_BLOCK_REMOTE = _env_bool("CAMOFOX_BLOCK_REMOTE", True)
CAMOFOX_REQUIRE_HTTPS_REMOTE = _env_bool("CAMOFOX_REQUIRE_HTTPS_REMOTE", True)
CAMOFOX_AUTH_PROBE_ENABLED = _env_bool("CAMOFOX_AUTH_PROBE_ENABLED", True)
CAMOFOX_REUSE_TAB = _env_bool("CAMOFOX_REUSE_TAB", True)
CAMOFOX_CHALLENGE_MAX_RETRIES = int(os.getenv("CAMOFOX_CHALLENGE_MAX_RETRIES", "2"))
CAMOFOX_BEHAVIOR_MIN_DELAY_SECONDS = float(os.getenv("CAMOFOX_BEHAVIOR_MIN_DELAY_SECONDS", "2.5"))
CAMOFOX_BEHAVIOR_MAX_DELAY_SECONDS = float(os.getenv("CAMOFOX_BEHAVIOR_MAX_DELAY_SECONDS", "7.5"))
if CAMOFOX_BEHAVIOR_MAX_DELAY_SECONDS < CAMOFOX_BEHAVIOR_MIN_DELAY_SECONDS:
    CAMOFOX_BEHAVIOR_MAX_DELAY_SECONDS = CAMOFOX_BEHAVIOR_MIN_DELAY_SECONDS

CAMOFOX_ALLOWED_HOSTS = [
    item.strip().lower()
    for item in os.getenv("CAMOFOX_ALLOWED_HOSTS", "127.0.0.1,localhost,::1").split(",")
    if item.strip()
]

# Crawl4AI
CRAWL4AI_ENABLED = _env_bool("CRAWL4AI_ENABLED", True)
RESEARCH_MAX_SEARCH_QUERIES = int(os.getenv("RESEARCH_MAX_SEARCH_QUERIES", "2"))
RESEARCH_MAX_DISCOVERED_SOURCES = int(os.getenv("RESEARCH_MAX_DISCOVERED_SOURCES", "5"))

# Google risk controls (these reduce self-inflicted bot flags; they don't guarantee access)
GOOGLE_MIN_INTERVAL_SECONDS = float(os.getenv("GOOGLE_MIN_INTERVAL_SECONDS", "15"))
GOOGLE_COOLDOWN_ON_CHALLENGE_SECONDS = float(os.getenv("GOOGLE_COOLDOWN_ON_CHALLENGE_SECONDS", "3600"))
GOOGLE_SEARCH_CACHE_TTL_SECONDS = float(os.getenv("GOOGLE_SEARCH_CACHE_TTL_SECONDS", str(48 * 3600)))
GOOGLE_SEARCH_LOCK_TIMEOUT_SECONDS = float(os.getenv("GOOGLE_SEARCH_LOCK_TIMEOUT_SECONDS", "120"))

RESEARCH_SOURCE_ALLOWLIST = [
    item.strip().lower()
    for item in os.getenv(
        "RESEARCH_SOURCE_ALLOWLIST",
        "reuters.com,apnews.com,bbc.com,bloomberg.com,ft.com,wsj.com,theguardian.com,cnn.com,nytimes.com,vnexpress.net,thanhnien.vn,tuoitre.vn",
    ).split(",")
    if item.strip()
]

DAILY_REPORT_FIXED_SOURCE_URLS = [
    item.strip()
    for item in os.getenv("DAILY_REPORT_FIXED_SOURCE_URLS", "https://vnexpress.net/").split(",")
    if item.strip()
]

# Memory in conversation
MAX_CONVERSATION_HISTORY = 20
KNOWLEDGE_MEMORY_TYPE = os.getenv("KNOWLEDGE_MEMORY_TYPE", "user_knowledge")

# Load system prompts from markdown files
WORKSPACE_PRIMING_ENABLED = _env_bool("WORKSPACE_PRIMING_ENABLED", True)
WORKSPACE_PRIMING_MAX_CHARS = int(os.getenv("WORKSPACE_PRIMING_MAX_CHARS", "24000"))
_workspace_priming_files_raw = os.getenv(
    "WORKSPACE_PRIMING_FILES",
    "AGENTS.md,BOOTSTRAP.md,IDENTITY.md,SOUL.md,USER.md,TOOLS.md,HEARTBEAT.md",
)
WORKSPACE_PRIMING_FILES = [
    item.strip() for item in _workspace_priming_files_raw.split(",") if item.strip()
]

WORKSPACE_PRIMING_FILE_SETS: dict[str, list[str]] = {
    "orchestrator": [
        "AGENTS.md",
        "IDENTITY.md",
        "SOUL.md",
        "USER.md",
    ],
    "researcher": [
        "skills/Researching/RESEARCHER.md",
    ],
    "coder": [
        "skills/Coding/CODER.md",
    ],
}

# Temp context dir (for model handoff serialisation)
TEMP_CONTEXT_DIR = SESSIONS_DIR
