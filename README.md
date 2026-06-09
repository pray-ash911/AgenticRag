# Agentic RAG with LangGraph

A scalable Retrieval-Augmented Generation (RAG) system built with LangGraph, featuring intelligent document grading, web search fallback, and persistent session memory.

## Features

- **Intelligent Retrieval**: Vector database search with PostgreSQL + PGVector
- **Document Grading**: LLM-based relevance scoring (Llama 3.1 8B)
- **Web Search Fallback**: Tavily search integration for knowledge gaps
- **Session Persistence**: PostgreSQL-backed conversation history
- **Streaming API**: FastAPI with real-time token streaming
- **Interactive Frontend**: Streamlit UI with session management

## Architecture

```
langgraphRag/
├── app/
│   ├── main.py                 # FastAPI server
│   ├── core/
│   │   ├── agent.py           # LangGraph workflow
│   │   ├── chains.py          # LLM configuration
│   │   └── nodes.py           # Agent nodes (retrieve, grade, generate)
│   ├── db/
│   │   ├── postgres_db.py     # Database connections
│   │   ├── persistence.py     # Session storage
│   │   └── vector_store.py    # Vector retrieval setup
│   ├── frontend/
│   │   └── streamlit_app.py   # Web UI
│   └── utils/
│       └── logger.py          # Logging setup
├── data/
│   ├── ingest.py              # Document ingestion
│   └── pdfs/                  # PDF storage
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Groq API Key
- Tavily API Key (optional)

### Installation

1. **Clone and setup environment**

```bash
cd langgraphRag
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

2. **Configure environment**

Create `.env`:

```env
# PostgreSQL
POSTGRES_USER=langgraph
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=langgraph_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# APIs
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

# Langfuse (optional)
LANGFUSE_PUBLIC_KEY=your_langfuse_key
LANGFUSE_SECRET_KEY=your_langfuse_secret
```

3. **Start PostgreSQL**

```bash
docker-compose up -d postgres
```

4. **Ingest Documents**

Place PDFs in `data/pdfs/` then:

```bash
python -m data.ingest
```

### Running the Application

**Option 1: FastAPI + Streamlit**

```bash
# Terminal 1: Start API server
python -m app.main

# Terminal 2: Start Streamlit frontend
cd app && streamlit run frontend/streamlit_app.py
```

**Option 2: Docker Compose (All-in-one)**

```bash
docker-compose up
```

## API Endpoints

### POST `/api/v1/chat`

Stream a response for a user query.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is RAG?", "thread_id": "session-123"}'
```

### GET `/api/v1/sessions`

List all chat sessions.

```bash
curl http://localhost:8000/api/v1/sessions
```

### GET `/api/v1/history/{thread_id}`

Retrieve chat history for a session.

```bash
curl http://localhost:8000/api/v1/history/session-123
```

### DELETE `/api/v1/history/{thread_id}`

Delete a session.

```bash
curl -X DELETE http://localhost:8000/api/v1/history/session-123
```

## Models Used

- **Generation**: Llama 3.3 70B (Groq) - High-quality answers
- **Grading & Logic**: Llama 3.1 8B (Groq) - Fast relevance scoring
- **Embeddings**: all-MiniLM-L6-v2 - Semantic search

## Troubleshooting

**Empty Vector Store**
```bash
python -m data.ingest
```

**PostgreSQL Connection Issues**
```bash
# Check connection
psql "postgresql://langgraph:password@localhost:5432/langgraph_db"
```

**API Not Starting**
```bash
# Check logs
python -m app.main --reload
```

## Performance Tips

1. Increase document grading threshold for faster responses
2. Cache embeddings for common queries
3. Use connection pooling for PostgreSQL
4. Enable Langfuse for observability

## License

MIT
