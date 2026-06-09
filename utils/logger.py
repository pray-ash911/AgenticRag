import logging
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(
    "AgenticRAG"
)


# Optional Langfuse integration
try:

    from langfuse.callback import CallbackHandler

    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )

except (ImportError, Exception):

    # Fallback handler if Langfuse is unavailable
    class DummyHandler:

        def flush(self):
            pass

    langfuse_handler = DummyHandler()

    logger.warning(
        "Langfuse not available - running without observability"
    )


# Return Langfuse callback handler
def get_langfuse_callback():
    return langfuse_handler