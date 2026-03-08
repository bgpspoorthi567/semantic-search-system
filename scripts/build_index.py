import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import joblib

from app.embeddings import Embedder
from app.vector_store import VectorStore
from app.clustering import FuzzyCluster


DATASET_PATH = "20_newsgroups"

documents = []
labels = []

print("Loading dataset...")

for category in os.listdir(DATASET_PATH):

    category_path = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(category_path):
        continue

    for file in os.listdir(category_path):

        file_path = os.path.join(category_path, file)

        try:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()

                documents.append(text)
                labels.append(category)

        except:
            continue

print("Total documents:", len(documents))


print("Generating embeddings...")

embedder = Embedder()

embeddings = embedder.encode(documents)


print("Building FAISS index...")

vector_store = VectorStore(embeddings.shape[1])

vector_store.add(embeddings, documents)


print("Training clustering model...")

cluster_model = FuzzyCluster(n_clusters=20)

cluster_model.fit(embeddings)


print("Saving models...")

joblib.dump(vector_store, "vector_store.pkl")

joblib.dump(cluster_model, "cluster_model.pkl")

print("Index built successfully")