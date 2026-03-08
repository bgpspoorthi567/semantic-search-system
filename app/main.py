from fastapi import FastAPI
import joblib
import numpy as np

from app.embeddings import Embedder
from app.cache import SemanticCache

# Create FastAPI app
app = FastAPI()

print("Loading models...")

vector_store = joblib.load("vector_store.pkl")
cluster_model = joblib.load("cluster_model.pkl")

embedder = Embedder()
cache = SemanticCache()


@app.post("/query")
def query_api(body: dict):

    query = body["query"]

    embedding = embedder.encode([query])[0]

    entry, sim = cache.lookup(embedding)

    if entry:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(sim),
            "result": entry["result"],
            "dominant_cluster": int(entry["cluster"])
        }

    result = vector_store.search(np.array([embedding]))

    cluster = cluster_model.dominant_cluster(embedding)

    cache.add(query, embedding, result, cluster)

    return {
        "query": query,
        "cache_hit": False,
        "result": result,
        "dominant_cluster": int(cluster)
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "cache cleared"}