import asyncio
import logging
from pydantic import BaseModel, Field
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from state import AgentState
from utils import get_llm


# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# 1. Router Schema
# -----------------------------
class RouterDecision(BaseModel):
    route: Literal["pinecone", "cypher", "both", "none"] = Field(
        ...,
        description="Chosen route for query handling."
    )
    reasoning: str = Field(..., description="Explanation for chosen route.")


# -----------------------------
# 3. Router Node
# -----------------------------

# Router prompt
system_prompt = """
You are an expert router for a Vietnam travel assistant. Your sole responsibility 
is to analyze the user's query and decide the best tool to use based on the 
nature of their question.

You have two data sources:
1.  **Pinecone (Vector Search):** Used for semantic, open-ended, or descriptive
    queries. Good for finding *similar* things or getting *general advice* and
    *recommendations* about cities, hotels, or activities.
2.  **Neo4j (Graph Search):** Used for specific, factual, or relational
    queries. Good for finding *exact matches*, *connections*, or *properties*
    of entities like hotels, cities, and their relationships (e.g., 'HOTEL_IN_CITY').

You must choose one of four routes based on the query:

1.  `pinecone`:
    - Use this for questions about "vibe", "suggestions", "descriptions", "recommendations", or "what is... like".
    - Example: "What are some good budget-friendly hotels in Ho Chi Minh City?"
    - Example: "Tell me about the food scene in Hanoi."
    - Example: "Find hotels similar to the 'La Siesta' hotel."
    - Example: "What is Da Nang like for a family vacation?"

2.  `cypher`:
    - Use this for specific,
      factual questions about entities and their properties.
    - Example: "What is the address of the 'La Siesta' hotel?"
    - Example: "Which hotels are located in the 'Old Quarter' of Hanoi?"
    - Example: "Does the 'InterContinental' hotel have a pool?"
    - Example: "List all hotels in Hanoi."

3.  `both`:
    - Use this for multi-part questions that combine both factual and descriptive needs.
    - Example: "Which hotels are in Hanoi [cypher] and what are they like [pinecone]?"
    - Example: "Tell me the price of the 'Rex Hotel' [cypher] and suggest similar hotels [pinecone]."
    - Example: "List hotels in Da Nang [cypher] and give me a summary of the best ones [pinecone]."

4.  `none`:
    - Use this *only* for greetings, simple conversation, or off-topic questions.
    - Example: "Hello"
    - Example: "Thanks, that was helpful."
    - Example: "What's 2+2?"

Based on the user's question, decide the best route.
"""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# Create the LLM with the structured output
llm = get_llm(temperature=0.0)

llm_with_router_schema = llm.with_structured_output(RouterDecision)

# Create the router chain
router_chain = router_prompt | llm_with_router_schema

async def router_node(state: AgentState) -> dict:
    """
    Determines the next step based on the user's query asynchronously.
    """
    logger.info("--- 1. Calling Router Node ---")

    question = state["question"]

    try:
        router_output: RouterDecision = await router_chain.ainvoke({"question": question})

        logger.info(f"Router Decision: {router_output.route}")
        logger.info(f"Router Reasoning: {router_output.reasoning}")

        return {"router_decision": router_output.route}

    except Exception as e:
        logger.error(f"Router Node failed: {e}")
        return {"router_decision": "none"}