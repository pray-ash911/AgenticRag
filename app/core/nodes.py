# Node logic (retrieve, grade, generate)
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from app.core.chains import utility_llm, primary_llm
from app.utils.logger import logger
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
import asyncio
from app.db.vector_store import get_retriever, get_async_retriever  # sync and async retrievers

#  1. Structured Output for Grader 
# This ensures the 8B model gives for relevance, which is crucial for the "Sub-LLM" logic.
class DocumentGrader(BaseModel):
    """Assessment of the retrieved document."""
    relevance_score: int = Field(
        description="Score from 0-10 on how well the document matches the query.", 
        ge=0, le=10
    )
    contains_answer: bool = Field(
        description="True if the document contains a direct answer to the user's core question."
    )
    hallucination_risk: bool = Field(
        description="True if the document seems contradictory or contains noisy, irrelevant data."
    )
    rationale: str = Field(
        description="Brief explanation of why this score was given."
    )

# Bind the more complex schema to the 8B model
advanced_grader_llm = utility_llm.with_structured_output(DocumentGrader)

# 2. Retrieve Node 
async def retrieve(state: Dict[str, Any]):
    logger.info("--- NODE: RETRIEVING DOCUMENTS ---")
    
    logger.info(f">>> Total messages in state: {len(state['messages'])}")
    for i, m in enumerate(state["messages"]):
        content = m.content if hasattr(m, "content") else str(m)
        logger.info(f"  [{i}] {type(m).__name__}: {content[:60]}")
    
    last_message = state["messages"][-1]
    question = last_message[1] if isinstance(last_message, tuple) else last_message.content
    
    # Initialize search_needed to "no" by default
    search_needed = "no"

    # Check if question can be answered from chat history directly
    if len(state["messages"]) > 1:
        recent_messages = state["messages"][:-1][-6:]  # last 6 messages excluding current
        history_text = "\n".join([
            f"{'User' if i % 2 == 0 else 'Assistant'}: {m.content if hasattr(m, 'content') else m[1]}"
            for i, m in enumerate(recent_messages)
        ])

        try:
            # Ask LLM if history already contains the answer
            check_response = await utility_llm.ainvoke(
                f"Given this conversation history:\n{history_text}\n\n"
                f"Question: {question}\n\n"
                f"Can this question be answered DIRECTLY from the history above?\n"
                f"Reply with ONLY: YES or NO"
            )
            check_text = check_response.content.strip().upper()
            
            # Extract YES/NO, ignoring any other text
            if "YES" in check_text:
                logger.info("--- ANSWERING FROM CHAT HISTORY ---")
                # Return empty docs — generate node will use history via state
                return {"documents": [], "question": question, "search_needed": "history"}
        except Exception as e:
            logger.warning(f"History check failed: {e}, proceeding with vector search")

    # Build context-aware search query
    search_query = question
    if len(state["messages"]) > 1:
        recent = state["messages"][-4:-1]
        history_text = " ".join([m.content if hasattr(m, "content") else m[1] for m in recent])
        search_query = f"{history_text} {question}"

    try:
        retriever = get_async_retriever()
        documents = await retriever.ainvoke(search_query)
        logger.info(f"Retrieved {len(documents) if documents else 0} documents from vector store")
        if not documents:
            logger.warning("No documents retrieved! Vector store may be empty. Run: python data/ingest.py")
            search_needed = "yes"  # Trigger web search if no docs found
        return {"documents": documents or [], "question": question, "search_needed": search_needed}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        logger.info("Proceeding with web search as fallback")
        return {"documents": [], "question": question, "search_needed": "yes"}

#  3. Grade Node (The "Sub-LLM" logic) 
async def grade_documents(state: Dict[str, Any]):
    logger.info("--- NODE: ADVANCED DOCUMENT GRADING ---")
    question = state["question"]
    documents = state["documents"]
    search_needed = state.get("search_needed", "no")
    
    # If already answering from history, skip grading and don't override the flag
    if search_needed == "history":
        logger.info("Already answering from chat history - skipping document grading.")
        return {"documents": [], "search_needed": "history"}

    # If no documents to grade, need web search (unless already determined)
    if not documents:
        logger.warning("No documents to grade. Triggering web search.")
        return {"documents": [], "search_needed": "yes"}

    filtered_docs = []
    total_relevance = 0

    for doc in documents:
        try:
            content = doc.page_content[:1500] if hasattr(doc, "page_content") else str(doc)[:1500]

            # Use plain LLM call instead of structured output — more reliable with Groq
            response = await utility_llm.ainvoke(
                f"Rate the relevance of this document to the question on a scale of 0-10.\n"
                f"Question: {question}\n"
                f"Document: {content}\n\n"
                f"Reply with ONLY a single integer from 0 to 10. Nothing else."
            )

            score_text = response.content.strip()
            # Extract first number found
            import re
            numbers = re.findall(r'\d+', score_text)
            score = int(numbers[0]) if numbers else 0
            score = min(10, max(0, score))  # clamp to 0-10

            logger.info(f"Doc Assessment: Score {score}/10")

            if score >= 7:
                filtered_docs.append(doc)
                total_relevance += score

        except Exception as e:
            logger.warning(f"Grading failed, skipping doc: {e}")
            continue

    if not filtered_docs or (len(documents) > 0 and total_relevance / len(documents) < 5):
        logger.warning("KNOWLEDGE GAP DETECTED: Proceeding to Web Search.")
        search_needed = "yes"
    else:
        logger.info("SUFFICIENT CONTEXT FOUND: Proceeding to Generation.")
        search_needed = "no"

    logger.info(f"Grade result - filtered_docs: {len(filtered_docs)}, search_needed: {search_needed}")
    return {"documents": filtered_docs, "search_needed": search_needed}

# Initialize Tavily
tavily_tool = TavilySearchResults(k=3)

async def web_search(state: Dict[str, Any]):
    logger.info("--- NODE: WEB SEARCHING (TAVILY) ---")
    question = state["question"]
    documents = state.get("documents", [])

    search_query_msg = await utility_llm.ainvoke(
        f"Transform this into a concise web search query, reply with ONLY the query, nothing else: {question}"
    )
    search_query = search_query_msg.content.strip()

    logger.info(f"Searching web for: {search_query}")
    docs = await tavily_tool.ainvoke({"query": search_query})

    web_docs = [Document(page_content=d["content"], metadata={"source": d.get("url", "web")}) for d in docs]
    documents.extend(web_docs)
    return {"documents": documents}

#  4. Generate Node (The "Big LLM" logic) 
async def generate(state: Dict[str, Any]):
    logger.info("--- NODE: GENERATING ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    
    # Build context from documents
    context_parts = []
    for d in documents:
        if isinstance(d, Document):
            context_parts.append(d.page_content)
        elif isinstance(d, BaseMessage):
            context_parts.append(d.content)
        else:
            context_parts.append(str(d))

    context = "\n\n".join(context_parts)

    # Include recent chat history in prompt
    history_text = ""
    if len(state["messages"]) > 1:
        history_text = "\n".join([
            f"{'User' if i % 2 == 0 else 'Assistant'}: {m.content if hasattr(m, 'content') else m[1]}"
            for i, m in enumerate(state["messages"][:-1][-6:])
        ])

    # Build prompt with clear separation markers
    prompt_parts = [
        "You are a helpful AI assistant.",
        "",
        "IMPORTANT RULES:",
        "1. NEVER quote or repeat the user's question in your response",
        "2. NEVER include the question text anywhere in your answer",
        "3. Answer the question directly and completely",
        #"4. Do not prefix with 'Based on...', 'The answer is...', etc.",
        "5. Be concise and clear",
        "6.Give atleast 3 key points in the answer if possible",
        ""
    ]
    
    if history_text:
        prompt_parts.append(f"Previous conversation:\n{history_text}\n")
    
    if context:
        prompt_parts.append(f"Reference material:\n{context}\n")
    
    prompt_parts.append(f"Question: {question}\n")
    prompt_parts.append("Answer:")
    
    prompt = "\n".join(prompt_parts)

    response = await primary_llm.ainvoke(prompt)
    answer = response.content.strip()
    
    # Clean response: remove any echoed question or artifacts
    answer = clean_response(answer, question)
    
    logger.info(f"Generated response length: {len(answer)} chars")
    return {"messages": [AIMessage(content=answer)]}

def clean_response(response: str, question: str) -> str:
    """Remove echoed input and clean artifacts from LLM response."""
    import re
    
    # First pass: Remove ALL quoted text (it's usually echoing the question)
    response = re.sub(r'"[^"]*"', '', response)
    response = re.sub(r'\[[^\]]*\]', '', response)
    
    # Remove standalone YES/NO (from retrieve history check leaking through)
    response = re.sub(r'\bYES\b|\bNO\b', '', response, flags=re.IGNORECASE)
    
    # Split into lines for more processing
    lines = response.split('\n')
    cleaned = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines (will normalize spacing later)
        if not stripped:
            continue
        
        # Skip pure artifacts: only digits, only YES/NO/commas/spaces
        if stripped and all(c in '0123456789YES NO,. \t' for c in stripped):
            continue
        
        # Skip marker lines
        if stripped.lower() in ['user:', 'assistant:', 'question:', 'answer:', 'reply:']:
            continue
        
        # Skip lines with only formatting/punctuation and minimal text
        alpha_count = sum(1 for c in stripped if c.isalpha())
        if len(stripped) > 0 and alpha_count < len(stripped) * 0.3:
            continue
        
        cleaned.append(line)
    
    # Join and normalize whitespace
    result = '\n'.join(cleaned).strip()
    result = re.sub(r'\n\n+', '\n\n', result)  # Max 2 consecutive newlines
    result = re.sub(r'  +', ' ', result)  # Single space max
    
    # Remove leading markdown headers
    result = re.sub(r'^#+\s+', '', result, flags=re.MULTILINE).strip()
    
    # If response is too short after cleaning, it might be corrupted
    if len(result.strip()) < 5:
        return response  # Return original if cleaning was too aggressive
    
    return result