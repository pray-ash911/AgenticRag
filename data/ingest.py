import asyncio
import os
from langchain_community.document_loaders import PyPDFLoader
from app.db.vector_store import get_retriever
from app.utils.logger import logger


# Load all PDF files and store them in vector database
async def ingest_documents():

    retriever = get_retriever()

    pdf_dir = "data/pdfs"

    all_documents = []

    # Read all PDF files from folder
    for filename in sorted(os.listdir(pdf_dir)):

        # Skip non-pdf files
        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(pdf_dir, filename)

        logger.info(f"Loading PDF: {file_path}")

        # Load PDF pages
        loader = PyPDFLoader(file_path)

        documents = loader.load()

        logger.info(f"Loaded {len(documents)} pages")

        # Store pages in main list
        all_documents.extend(documents)

    logger.info(f"Total pages loaded: {len(all_documents)}")

    # Add documents to vector store
    retriever.add_documents(all_documents)

    logger.info("Document ingestion completed")


# Run ingestion script
if __name__ == "__main__":
    asyncio.run(
        ingest_documents()
    )