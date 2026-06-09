import os
import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
from app.utils.logger import logger

load_dotenv()

# Build the connection string from your docker-compose environment
DB_URI = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

# Async connection pool for scalability
def get_connection_pool():
    try:
        pool = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True})
        logger.info("Successfully established PostgreSQL connection pool.")
        return pool
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL pool: {e}")
        raise e

db_pool = get_connection_pool()