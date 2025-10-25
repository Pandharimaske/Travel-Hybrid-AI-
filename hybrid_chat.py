import asyncio
import logging
from state import AgentState
from langgraph.graph import StateGraph, END
from GraphNodes.pinecone_node import call_pinecone_node
from GraphNodes.cypher_node import call_cypher_node
from GraphNodes.router_node import router_node
from GraphNodes.answer_node import synthesize_answer_node

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Conditional Router
# -----------------------------
def conditional_router(state: AgentState) -> str:
    """Decides which branch to follow based on router_decision."""
    logger.info("--- 2. Calling Conditional Router ---")
    decision = state["router_decision"]
    logger.info(f"Routing to: '{decision}'")
    return decision

# -----------------------------
# Parallel Search Node
# -----------------------------
async def parallel_search_node(state: AgentState) -> dict:
    """
    Triggers parallel execution of Pinecone and Cypher searches.
    This node doesn't modify the state — it just signals both to start.
    """
    logger.info("--- 3. Triggering Parallel Searches ---")
    return {}

# -----------------------------
# Graph Assembly
# -----------------------------
logger.info("Assembling LangGraph workflow...")
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("pinecone_search", call_pinecone_node)
workflow.add_node("cypher_search", call_cypher_node)
workflow.add_node("synthesize_answer", synthesize_answer_node)
workflow.add_node("parallel_search", parallel_search_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    conditional_router,
    {
        "pinecone": "pinecone_search",
        "cypher": "cypher_search",
        "both": "parallel_search",
        "none": "synthesize_answer"
    }
)

workflow.add_edge("parallel_search", "pinecone_search")
workflow.add_edge("parallel_search", "cypher_search")

workflow.add_edge("pinecone_search", "synthesize_answer")
workflow.add_edge("cypher_search", "synthesize_answer")

workflow.add_edge("synthesize_answer", END)

app = workflow.compile()
logger.info("Graph compiled successfully.")

# -----------------------------
# Main Async Runner
# -----------------------------
async def main():
    logger.info("Vietnam Travel Assistant is ready!")

    print("\n--- Vietnam Travel Assistant ---")
    print("Ask me about hotels, cities, and activities in Vietnam.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("---------------------------------")

    while True:
        query = input("\nUser: ").strip()

        if not query or query.lower() in ("exit", "quit"):
            print("\nAssistant: Goodbye! Have a great day.")
            break

        inputs = {
            "question": query,
            "router_decision": "",
            "vector_search_context": "",
            "graph_search_context": "",
            "answer": ""
        }

        try:
            final_state: AgentState = await app.ainvoke(inputs)
            answer = final_state.get(
                "answer",
                "Sorry, I seem to have lost my train of thought. Could you ask again?"
            )
            print(f"\nAssistant:\n{answer}")

        except Exception as e:
            logger.exception("Error in workflow execution")
            print(f"\nAssistant: [An error occurred: {e}]")
            print("I'm sorry, I ran into a problem. Please try rephrasing your question.")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())