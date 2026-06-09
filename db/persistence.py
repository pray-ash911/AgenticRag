from langgraph.checkpoint.postgres import PostgresSaver
from app.db.postgres_db import DB_URI


# Create PostgreSQL checkpointer
def get_checkpointer():

    # Stores chat history and graph state
    # for multiple users using thread_id
    return PostgresSaver.from_conn_string(
        DB_URI
    )


# Global memory object
memory = get_checkpointer()