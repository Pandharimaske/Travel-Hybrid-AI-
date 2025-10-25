# pinecone_upload.py
import json
import time
import asyncio
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import config


# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 64  # slightly higher for better throughput
MAX_WORKERS = 5  # concurrent upserts

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 1536 for text-embedding-3-small


# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Initialize clients
# -----------------------------
logger.info("Initializing clients...")
pc = Pinecone(api_key=config.PINECONE_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create index if not exists
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    logger.info(f"Creating new Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    logger.info(f"Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)
logger.info(f"Connected to Pinecone index: {INDEX_NAME}")


# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts):
    """Generate embeddings using SentenceTransformer."""
    return model.encode(texts).tolist()


def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


# -----------------------------
# Async helpers
# -----------------------------
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

async def async_upsert(vectors, batch_num):
    """Run Pinecone upsert asynchronously using thread executor."""
    loop = asyncio.get_event_loop()
    start = time.time()
    try:
        await loop.run_in_executor(executor, index.upsert, vectors)
        logger.info(f"✅ Batch {batch_num} uploaded ({len(vectors)} items) in {time.time() - start:.2f}s")
    except Exception as e:
        logger.error(f"❌ Error uploading batch {batch_num}: {e}")


async def upload_batches(all_batches):
    """Upload all batches concurrently."""
    tasks = []
    for i, batch in enumerate(all_batches, start=1):
        tasks.append(async_upsert(batch, i))
    await asyncio.gather(*tasks)


# -----------------------------
# Main upload logic
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", []),
        }
        items.append((node["id"], semantic_text, meta))

    logger.info(f"Preparing to upload {len(items)} items in batches of {BATCH_SIZE}...")

    all_batches = []
    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Preparing batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]
        embeddings = get_embeddings(texts)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]
        all_batches.append(vectors)

    logger.info(f"Starting async upload with {MAX_WORKERS} workers...")
    asyncio.run(upload_batches(all_batches))
    logger.info("🎉 All items uploaded successfully.")


# -----------------------------
if __name__ == "__main__":
    main()