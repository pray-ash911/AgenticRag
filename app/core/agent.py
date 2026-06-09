from typing import TypedDict, Annotated, List, Any
import asyncio
import psycopg_pool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.nodes import (
    retrieve,
    grade_documents,
    web_search,
    generate
)
from app.db.postgres_db import DB_URI
from app.utils.logger import logger


# State shared between all nodes
class AgentState(TypedDict):
    question: str
    documents: List[Any]
    search_needed: str

    # Combine old and new messages together
    messages: Annotated[List[Any], lambda old, new: old + new]


# Decide next step after grading documents
def route_after_grading(state: AgentState) -> str:
    search_needed = state.get("search_needed")

    # Use previous history directly
    if search_needed == "history":
        logger.info("Routing to generate from history")
        return "generate"

    # Perform web search if needed
    if search_needed == "yes":
        logger.info("Routing to web search")
        return "web_search"

    # Default route
    logger.info("Routing to generate")
    return "generate"


# Create workflow graph
def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Add graph nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    # Starting node
    workflow.set_entry_point("retrieve")

    # Connect nodes
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "web_search": "web_search"
        }
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow


# Create graph with PostgreSQL checkpointing
async def create_graph():

    # Database connection pool
    pool = psycopg_pool.AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0
        }
    )

    # Open database connection
    await pool.open()

    # Create checkpoint manager
    checkpointer = AsyncPostgresSaver(pool)

    # Create checkpoint tables if missing
    await checkpointer.setup()

    # Build workflow
    workflow = build_workflow()

    # Compile graph
    app = workflow.compile(
        checkpointer=checkpointer
    )

    logger.info("Graph compiled successfully")

    return app


# Run async graph creation synchronously
def get_agent():
    return asyncio.get_event_loop().run_until_complete(
        create_graph()
    )


# Main graph object
agent_executor = get_agent()
