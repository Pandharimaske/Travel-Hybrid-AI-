
from typing import TypedDict, List, Literal, Sequence, Annotated

# -----------------------------
# 1. Agent State
# -----------------------------
class AgentState(TypedDict):
    """
    Defines the state for our hybrid search agent graph.
    It holds all the data passed between nodes.
    """
    
    # The user's input question
    question: str

    # The decision made by the router ('pinecone', 'cypher', 'both', or 'none')
    router_decision: str

    # Context from the vector search tool, formatted as a single string
    vector_search_context: str

    # Context from the graph search tool, returned as a string
    graph_search_context: str

    # The final, human-readable answer to be generated
    answer: str