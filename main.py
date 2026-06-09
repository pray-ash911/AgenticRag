from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from app.core.agent import agent_executor
from app.utils.logger import langfuse_handler, logger
from app.db.postgres_db import DB_URI
from dotenv import load_dotenv
import uuid
import psycopg
import json

load_dotenv()

app = FastAPI(
    title="Agentic RAG API",
    description="Scalable FastAPI for Llama-3.3 Agent with Session Memory",
    version="1.0.0"
)

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str = str(uuid.uuid4())

async def stream_tokens(message: str, thread_id: str):
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler]
    }
    try:
        streamed = False
        async for event in agent_executor.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config,
            version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # Filter out token IDs and encoding artifacts
                    if not _is_artifact(content):
                        streamed = True
                        yield content
            elif kind == "on_chain_end" and event.get("name") == "generate" and not streamed:
                # Only use fallback if streaming didn't fire
                output = event["data"].get("output", {})
                messages = output.get("messages", [])
                if messages:
                    yield messages[-1].content
    finally:
        langfuse_handler.flush()
        logger.info("--- LANGFUSE: Traces flushed ---")

def _is_artifact(text: str) -> bool:
    """Check if text is an encoding artifact rather than real content."""
    # Token IDs, encoding markers, etc.
    if text.strip() and all(c in '0123456789NO' for c in text.replace(' ', '')):
        return True
    # Very short sequences that look like tokens
    if len(text.strip()) <= 3 and all(c.isdigit() for c in text.strip()):
        return True
    return False

@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    logger.info(f"Session {request.thread_id}: Received message")
    return StreamingResponse(
        stream_tokens(request.message, request.thread_id),
        media_type="text/event-stream"
    )

@app.get("/api/v1/sessions")
async def get_sessions():
    try:
        conn = psycopg.connect(DB_URI)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
        thread_ids = [row[0] for row in cur.fetchall()]
        conn.close()

        sessions = []
        for tid in thread_ids:
            try:
                config = {"configurable": {"thread_id": tid}}
                state = await agent_executor.aget_state(config)
                messages = state.values.get("messages", [])
                # Get first user message for session name
                first_msg = next((m.content for m in messages if isinstance(m, HumanMessage)), tid)
                # Generate short context-based name
                name_response = await utility_llm.ainvoke(
                    f"Summarize this message in 4 words max as a chat title, no quotes: {first_msg[:200]}"
                )
                name = name_response.content.strip()[:40]
            except:
                name = tid[:20] + "..."
            sessions.append({"thread_id": tid, "name": name})

        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return {"sessions": [], "error": str(e)}

@app.get("/api/v1/history/{thread_id}")
async def get_chat_history(thread_id: str):
    """Get full chat history for a session"""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = await agent_executor.aget_state(config)
        messages = []
        for msg in state.values.get("messages", []):
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        return {"thread_id": thread_id, "messages": messages}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return {"thread_id": thread_id, "messages": []}

@app.delete("/api/v1/history/{thread_id}")
async def delete_session(thread_id: str):
    """Delete a session"""
    try:
        conn = psycopg.connect(DB_URI)
        cur = conn.cursor()
        cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
        cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
        conn.commit()
        conn.close()
        return {"status": "deleted", "thread_id": thread_id}
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)