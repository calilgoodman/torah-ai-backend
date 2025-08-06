from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from torah_ai_backend.query_rewriter import generate_semantic_query
from InstructorEmbedding import INSTRUCTOR
import os

app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://torahlifeguide.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Paths ===
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chromadb")

# === Ensure ChromaDB directory exists ===
os.makedirs(CHROMA_PATH, exist_ok=True)

# === Initialize ChromaDB and embedding model ===
client = PersistentClient(path=CHROMA_PATH)
embedding_model = INSTRUCTOR("hkunlp/instructor-xl")

# === Embedding Function ===
def embedding_func(texts):
    return embedding_model.encode([
        f"Represent the query for retrieval: {t}" for t in texts
    ])

# === Request Schema ===
class QueryInput(BaseModel):
    prompt: str
    theme: str
    main: str
    sub: str
    sources: list[str]

# === API Route ===
@app.post("/query")
async def query_sources(query: QueryInput):
    semantic_query = generate_semantic_query(
        query.prompt, query.theme, query.main, query.sub
    )

    results = []
    for source in query.sources:
        collection = client.get_collection(name=source)
        query_result = collection.query(
            query_embeddings=embedding_func([semantic_query]),
            n_results=1
        )
        results.append(query_result)

    return {"results": results}
