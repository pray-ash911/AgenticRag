import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()


# Fast model for logic-related tasks
# Used for:
# - Grading
# - Routing
# - Query transformation
utility_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# Main model for final response generation
primary_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    streaming=False,
    api_key=os.getenv("GROQ_API_KEY")
)