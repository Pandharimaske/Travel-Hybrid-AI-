import asyncio
import logging
import config
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain.prompts import ChatPromptTemplate
from state import AgentState
from utils import get_llm

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/graph_cypher_node.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())  # Also log to console

# ---------------------------------------------------------------------
# 1. Initialize Neo4j Graph
# ---------------------------------------------------------------------
logger.info("Initializing Neo4jGraph client...")

try:
    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        enhanced_schema=True,
    )
    logger.info("✅ Neo4jGraph client initialized successfully.")
except Exception as e:
    logger.exception("❌ Error connecting to Neo4j or refreshing schema.")
    raise SystemExit("Failed to initialize Neo4j. Please check config.py credentials.")

# ---------------------------------------------------------------------
# 2. Cypher Generation Prompt
# ---------------------------------------------------------------------
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to 
answer questions about travel in Vietnam.

**CRITICAL INSTRUCTIONS:**
1. ALWAYS use the node labels `City`, `Hotel`, `Attraction`, or `Activity`.
2. NEVER use the generic `Entity` label.
3. NEVER query for string properties like `city` or `type` on `Hotel`, `Attraction`, or `Activity` nodes.
4. ALWAYS find a node's city by following:
   - (:Hotel)-[:Located_In]->(:City)
   - (:Attraction)-[:Located_In]->(:City)
   - (:Activity)-[:Available_In]->(:City)
5. ALWAYS query nodes by their `name` property (e.g., name: "Hanoi").
6. ALWAYS return id, name, and relevant properties.

Schema:
{schema}

---
Question:
{query}
"""

cypher_prompt = ChatPromptTemplate.from_messages([
    ("system", CYPHER_GENERATION_TEMPLATE)
])

# ---------------------------------------------------------------------
# 3. Create GraphCypherQAChain
# ---------------------------------------------------------------------
try:
    llm = get_llm(temperature=0.0)
    cypher_qa_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        verbose=True,  # Set to False for cleaner logs in production
        allow_dangerous_requests=True,
    )
    logger.info("✅ GraphCypherQAChain initialized successfully.")
except Exception as e:
    logger.exception("❌ Failed to initialize GraphCypherQAChain.")
    raise SystemExit("Failed to initialize Cypher chain.")

# ---------------------------------------------------------------------
# 4. Async Node Function
# ---------------------------------------------------------------------
async def call_cypher_node(state: AgentState) -> dict:
    """
    Asynchronous version of the GraphCypherQAChain node.
    Translates user question → Cypher → executes → natural language answer.
    """
    question = state.get("question", "")
    logger.info(f"--- Executing Cypher Node for question: '{question}' ---")

    if not question.strip():
        logger.warning("⚠️ No question provided to Cypher node.")
        return {"graph_search_context": "No question provided."}

    try:
        # Run blocking chain in background thread
        result = await asyncio.to_thread(cypher_qa_chain.invoke, {"query": question})

        # Extract human-readable result
        answer = result.get("result", "No answer found from graph.")
        logger.info(f"✅ Cypher execution completed. Answer: {answer[:100]}...")

        return {"graph_search_context": str(answer)}

    except Exception as e:
        logger.exception("❌ Error during Cypher QA chain execution.")
        return {"graph_search_context": f"Error running graph query: {e}"}