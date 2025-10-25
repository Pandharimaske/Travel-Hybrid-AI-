import asyncio
import json
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import config
from state import AgentState


# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/vector_search_node.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())  # also log to console


# ---------------------------------------------------------------------
# Config & Initialization
# ---------------------------------------------------------------------
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME
PINECONE_DIM = 384  # must match model (all-MiniLM-L6-v2)

logger.info("Initializing Pinecone client...")
pc = Pinecone(api_key=config.PINECONE_API_KEY)

logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure index exists or create
if INDEX_NAME not in pc.list_indexes().names():
    logger.warning(f"Index {INDEX_NAME} not found. Creating new managed index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=PINECONE_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    logger.info(f"✅ Index {INDEX_NAME} created.")
else:
    logger.info(f"Connecting to existing index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)
logger.info("✅ Pinecone and embedding model initialized successfully.")


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def embed_text(text: str) -> List[float]:
    """Generate embedding for text using the local model."""
    return embed_model.encode(text).tolist()


# ---------------------------------------------------------------------
# Async Pinecone Search Node
# ---------------------------------------------------------------------
async def call_pinecone_node(state: AgentState) -> dict:
    """
    Asynchronous Pinecone vector search node.
    Embeds the user's question, queries Pinecone,
    and returns relevant context as JSON.
    """
    question = state.get("question", "")
    logger.info(f"--- Executing Pinecone Search Node for question: '{question}' ---")

    if not question.strip():
        logger.warning("⚠️ No question provided to Pinecone search node.")
        return {"vector_search_context": json.dumps([{"error": "Empty question"}])}

    try:
        # Run embedding in background thread (non-blocking)
        vec = await asyncio.to_thread(embed_text, question)

        # Run Pinecone query asynchronously as well
        res = await asyncio.to_thread(
            index.query,
            vector=vec,
            top_k=TOP_K,
            include_metadata=True,
            include_values=False,
        )

        matches = res.get("matches", [])
        logger.info(f"✅ Retrieved {len(matches)} matches from Pinecone.")

        # Extract and format metadata for LLM
        context_list = [m["metadata"] for m in matches]
        context_str = json.dumps(context_list, ensure_ascii=False)

        return {"vector_search_context": context_str}

    except Exception as e:
        logger.exception("❌ Error during Pinecone vector search.")
        return {"vector_search_context": json.dumps([{"error": str(e)}])}