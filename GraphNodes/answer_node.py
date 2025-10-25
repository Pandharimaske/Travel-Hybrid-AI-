import asyncio
import logging
from state import AgentState
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import get_llm

# ---------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Optional: File handler for logs
file_handler = logging.FileHandler("logs/synthesis_node.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------
SYNTHESIS_PROMPT_TEMPLATE = """
You are an expert Vietnam travel assistant. Your job is to synthesize a single,
comprehensive, and helpful answer for the user.

You have been provided with the user's original question and may have one or 
two sources of context to help you answer.

**User's Original Question:**
{question}

---
**Context from Vector Search (General info, descriptions, recommendations):**
{vector_context}

---
**Context from Graph Search (Specific facts, lists, properties like address/price):**
{graph_context}

---
**Your Task:**
1. Analyze the user's question and all available context.
2. Base your answer *only* on the provided contexts. Do not make up information.
3. If both contexts are provided, combine them into one seamless answer.
4. If only one context is provided, use that to answer the question.
5. If both context sections are empty or say 'No context', it means no 
   information was found, or the user asked a simple greeting (like 'Hello'). 
   In this case, provide a friendly, conversational response.
6. Do not mention "Vector Search" or "Graph Search" in your final answer.
   Just present the information as a helpful assistant.

**Final Answer:**
"""

# ---------------------------------------------------------------------
# Chain Initialization
# ---------------------------------------------------------------------
synthesis_prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT_TEMPLATE)
llm = get_llm(temperature=0.2)
synthesis_chain = synthesis_prompt | llm | StrOutputParser()

logger.info("✅ Synthesis chain initialized successfully.")

# ---------------------------------------------------------------------
# Async Node Function
# ---------------------------------------------------------------------
async def synthesize_answer_node(state: AgentState) -> dict:
    """
    Asynchronously synthesizes the final answer using LLM based on
    both vector and graph contexts.
    """
    logger.info("--- Synthesizing Final Answer ---")

    question = state.get("question", "")
    v_context = state.get("vector_search_context", "No context provided.")
    g_context = state.get("graph_search_context", "No context provided.")

    if not v_context:
        v_context = "No context provided."
    if not g_context:
        g_context = "No context provided."

    logger.info(f"🧠 Synthesizing for: {question}")
    logger.debug(f"Vector context: {v_context[:120]}...")
    logger.debug(f"Graph context: {g_context[:120]}...")

    try:
        # Run the synchronous LLM chain in a separate thread
        answer = await asyncio.to_thread(
            synthesis_chain.invoke,
            {"question": question, "vector_context": v_context, "graph_context": g_context}
        )
        logger.info("✅ Synthesis complete.")
        return {"answer": answer}

    except Exception as e:
        logger.exception(f"❌ Error during synthesis: {e}")
        return {"answer": f"Sorry, an error occurred while generating the answer: {e}"}