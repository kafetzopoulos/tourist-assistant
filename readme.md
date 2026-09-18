# AI Tourist Assistant — Alexandroupolis

A conversational assistant that helps visitors explore Alexandroupolis, Greece. It answers factual questions about attractions, recommends places based on user interests, generates feasible time-boxed itineraries, and reports live weather — all grounded in a curated knowledge base and a deterministic feasibility engine.

The design deliberately separates probabilistic LLM behaviour (intent classification, preference extraction, natural-language phrasing) from deterministic business logic (retrieval, scheduling, opening-hours checks, travel-time computation, weather fetch).

## Table of contents
- [Features](#features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Setup](#setup)
- [Building the knowledge index](#building-the-knowledge-index)
- [Running the assistant](#running-the-assistant)

---

## Features

- **Intent routing** — classifies each message into one of eight intents: `weather_query`, `factual_qa`, `recommendation`, `plan_request`, `modify_plan`, `out_of_scope`, `chitchat`, `conversation_meta`.
- **Grounded factual Q&A** — answers questions about a specific place using retrieved context only, with source attribution.
- **Personalised recommendations** — extracts interests/dislikes/pace/transport from free text and retrieves matching attractions.
- **Feasible itinerary planning** — builds a schedule that respects travel time, opening hours, visit duration, pace, and (optionally) rain forecasts. No LLM in the scheduling loop.
- **Itinerary modification** — remove, replace, add activities, or change pace/transport; the plan is re-flowed deterministically.
- **Live weather** — current conditions and an hourly forecast, with indoor alternatives suggested when rain is likely.
- **Prompt-injection guard** — a pre-LLM gate (`is_malicious`) blocks unsafe or rule-overriding input.
- **Conversation memory** — bounded per-session history plus a structured `UserPreferences` object.

---

## Architecture

```text
User message
   │
   ▼
[Prompt guard] ─────────► deterministic (is_malicious)
   │
   ▼
[Router] ───────────────► LLM → {intent, time_window, start_date, destination, reason}
   │
   ├─ out_of_scope ──────────────► canned reply           (deterministic)
   ├─ chitchat ──────────────────► LLM reply              (probabilistic, no facts)
   ├─ conversation_meta ─────────► LLM reply over transcript (probabilistic)
   ├─ weather_query ─────────────► weather tool → LLM phrasing
   ├─ factual_qa ────────────────► RAG → LLM answer         (grounded)
   ├─ recommendation ────────────► prefs → RAG → LLM answer (grounded)
   ├─ plan_request ──────────────► prefs → RAG → planner → LLM phrasing
   └─ modify_plan ───────────────► prefs → RAG → planner → LLM phrasing
   │
   ▼
[Session store] ────────► append to history, trim, save
```

### Probabilistic vs Deterministic Components

| Component | Type | Why |
| :--- | :--- | :--- |
| `prompt_guard.is_malicious` | Deterministic | Pre-LLM gate; blocks before any cost |
| `RouterOutput` (intent, time window, start date, destination) | Probabilistic | Natural-language intent is fuzzy |
| `PreferenceOutput` extraction | Probabilistic | Free-text $\rightarrow$ structured interests |
| `ModifyOutput` extraction | Probabilistic | "swap the museum" $\rightarrow$ structured action |
| RAG retrieval (`Retriever`) | Deterministic | FAISS + cosine threshold |
| `plan_day` / `build_itinerary` | Deterministic | Pure function of candidates + window + prefs |
| `travel_time_minutes` | Deterministic | Haversine + effective speed |
| `hours.is_open_at` / `next_opening` | Deterministic | Parsed from KB |
| Weather fetch | Deterministic | Fetch, external data, cached per TTL |
| Final phrasing | Probabilistic, grounded | Must not alter facts |

> The LLM classifies, extracts, and phrases. It never invents the schedule, opening hours, or travel times.

---

## Project structure

```text
tourist_assistant/
├── config.py                  # Settings (env-driven)
├── models.py                  # Pydantic domain models
├── conversation/
│   ├── manager.py             # ConversationManager — orchestrator
│   └── state.py               # SessionStore (in-memory; swap for Redis)
├── llm/
│   ├── client.py              # Azure OpenAI client, chat + chat_structured
│   ├── prompts.py             # All system/user prompt templates
│   └── schemas.py             # RouterOutput, PreferenceOutput, ModifyOutput
├── planning/
│   ├── feasibility.py         # plan_day / build_itinerary / check_itinerary
│   ├── geo.py                 # Haversine travel time
│   ├── hours.py               # Opening-hours logic
│   └── itinerary.py           # ItineraryService (domain service)
├── rag/
│   ├── ingest.py              # FAISS index builder
│   ├── retriever.py           # Semantic retriever
│   └── attractions.json       # Knowledge base
├── security/
│   └── prompt_guard.py        # is_malicious
└── tools/
    └── weather.py             # Open-Meteo wrapper
```

---

## Requirements

- Python 3.11+
- An Azure OpenAI resource with a deployed chat model (e.g., `gpt-4o-mini`)
- Network access to Open-Meteo (no API key required)
- Python packages:
  ```text
  openai>=1.0
  pydantic>=2.0
  faiss-cpu
  sentence-transformers
  httpx
  ```

---

## Configuration

Settings are loaded from environment variables via `tourist_assistant.config.settings`.

| Variable | Purpose | Example |
| :--- | :--- | :--- |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key | — |
| `AZURE_OPENAI_API_VERSION` | API version | `2024-08-01-preview` |
| `AZURE_OPENAI_ENDPOINT` | Resource endpoint | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Sentence-transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `ATTRACTIONS_PATH` | Path to KB JSON | `tourist_assistant/rag/attractions.json` |
| `FAISS_INDEX_PATH` | FAISS index output | `.cache/faiss.index` |
| `FAISS_META_PATH` | Chunk metadata output | `.cache/faiss_meta.json` |
| `HISTORY_TURNS_TO_LLM` | Turns of history sent to LLM | `6` |
| `RAG_TOP_K_FACTUAL` | Chunks for factual Q&A | `5` |
| `RAG_TOP_K_RECOMMENDATION` | Attractions for recommendations | `5` |
| `RAG_TOP_K_PLAN` | Attractions for planning | `8` |
| `RAG_TOP_K_REPLACEMENT` | Candidates for replace/add | `5` |
| `DEFAULT_PLAN_HOURS` | Fallback plan length | `4` |
| `DEFAULT_PLAN_START_HOUR` | Start hour for dated plans | `9` |
| `DEFAULT_PLAN_START_OFFSET_HOURS` | Offset from "now" for undated plans | `1` |
| `WEATHER_FORECAST_HOURS` | Hours of forecast to summarize | `6` |
| `DEFAULT_LAT`, `DEFAULT_LON` | City center origin | `40.8475, 25.8744` |

> Never commit keys. Use a `.env` file locally and a secret manager in production.

---

## Setup

```bash
git clone <repo>
cd tourist-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Building the knowledge index

The retriever builds the FAISS index automatically on first use if it's missing. To force a rebuild (e.g., after editing `attractions.json`):

```bash
python -m tourist_assistant.rag.ingest
```

You can also sanity-check retrieval directly:

```bash
python -m tourist_assistant.rag.retriever
```

This prints top matches for a few sample queries.

---

## Running the assistant

Minimal REPL:

```python
from tourist_assistant.conversation.manager import get_manager

manager = get_manager()
session_id = None

while True:
    msg = input("You: ")
    if msg.strip().lower() in {"exit", "quit"}:
        break
    result = manager.handle(msg, session_id=session_id)
    session_id = result["session_id"]
    print("Assistant:", result["reply"])
```

`handle()` returns `{"session_id": str, "reply": str}`.