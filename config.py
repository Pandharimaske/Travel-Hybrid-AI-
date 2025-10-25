# config_example.py — copy to config.py and fill with real values.
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"
NEO4J_URI = "neo4j+s://"
NEO4J_DATABASE = "neo4j"

OPENAI_API_KEY = "sk-"# your OpenAI API key

PINECONE_API_KEY="pcsk" # your Pinecone API key
PINECONE_ENV = "us-east-1"   # example
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 384       # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
