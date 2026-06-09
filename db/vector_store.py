from sqlalchemy.ext.asyncio import create_async_engine
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.storage import SQLStore
from app.db.postgres_db import DB_URI
from app.utils.logger import logger


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Split small chunks for vector search
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)


# Split larger parent documents
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)


# Database URLs
SYNC_DB_URI = DB_URI.replace(
    "postgresql://",
    "postgresql+psycopg2://"
)

ASYNC_DB_URI = DB_URI.replace(
    "postgresql://",
    "postgresql+asyncpg://"
)


# Create synchronous retriever
def get_retriever():

    # PGVector vector database
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="rag_collection",
        connection=SYNC_DB_URI,
        async_mode=False,
        use_jsonb=True
    )

    # Store parent documents
    byte_store = SQLStore(
        db_url=SYNC_DB_URI,
        namespace="parent_docs"
    )

    # Create table if not exists
    byte_store.create_schema()

    # Convert documents to bytes automatically
    docstore = create_kv_docstore(
        byte_store
    )

    # Parent-child retriever
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )


# Create asynchronous retriever
def get_async_retriever():

    # Async PGVector store
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="rag_collection",
        connection=ASYNC_DB_URI,
        async_mode=True,
        use_jsonb=True,
        create_extension=False
    )

    # Async PostgreSQL engine
    async_engine = create_async_engine(
        ASYNC_DB_URI
    )

    # Async SQL document store
    byte_store = SQLStore(
        engine=async_engine,
        namespace="parent_docs",
        async_mode=True
    )

    # Handle document serialization
    docstore = create_kv_docstore(
        byte_store
    )

    # Parent-child retriever
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
