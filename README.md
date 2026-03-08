# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a **semantic search system** built on the **20 Newsgroups dataset**.
Instead of traditional keyword matching, the system retrieves documents based on **semantic similarity** using vector embeddings.

The system also integrates **fuzzy clustering** and a **semantic caching layer** to improve retrieval efficiency and demonstrate real-world ML system design.

The entire system is exposed through a **FastAPI service**, allowing queries to be sent through a REST API.

---

# Key Features

• Semantic document retrieval using **vector embeddings**
• Fast similarity search using **FAISS vector database**
• **Fuzzy clustering** using Gaussian Mixture Model
• **Semantic cache** that detects similar queries using cosine similarity
• REST API service implemented using **FastAPI**

---

# System Architecture

User Query
↓
SentenceTransformer Embedding Model
↓
Semantic Cache Check
↓
FAISS Vector Search
↓
Fuzzy Clustering (Gaussian Mixture Model)
↓
API Response

---

# Dataset

This project uses the **20 Newsgroups dataset**, which contains approximately **20,000 documents across 20 categories**.

Example categories include:

* sci.space
* comp.graphics
* rec.sport.baseball
* talk.politics.misc
* comp.sys.ibm.pc.hardware

The dataset is useful for benchmarking **text classification and retrieval systems**.

---

# Technologies Used

Python
SentenceTransformers
FAISS
Scikit-learn
FastAPI
NumPy

---

# Embedding Model

The system uses:

**SentenceTransformer – all-MiniLM-L6-v2**

Reasons for selecting this model:

* lightweight and efficient
* strong performance on semantic similarity tasks
* suitable for search systems

Each document is converted into a **384-dimensional embedding vector**.

---

# Vector Database

The embeddings are stored in **FAISS (Facebook AI Similarity Search)**.

FAISS enables:

* efficient nearest-neighbor search
* fast similarity queries over large vector datasets
* scalable semantic retrieval

---

# Fuzzy Clustering

To analyze topic structure in the dataset, the system uses a **Gaussian Mixture Model (GMM)**.

Unlike hard clustering algorithms such as K-Means, GMM provides **probabilistic cluster memberships**.

Example:

Document about space shuttle propulsion

Cluster 5 → 0.52
Cluster 12 → 0.31
Cluster 3 → 0.17

This reflects real-world scenarios where documents may belong to multiple topics.

---

# Semantic Cache

Traditional caching works only for **identical queries**.

This project implements a **semantic cache**, which stores previous queries and detects **similar queries using cosine similarity** between embeddings.

Example:

Query 1:
"space shuttle launch"

Query 2:
"space shuttle mission"

Even though the queries are not identical, their embeddings are similar, allowing the cached result to be reused.

This reduces computation and improves system performance.

---

# API Endpoints

The system exposes three REST endpoints.

## POST /query

Submit a query to retrieve semantically similar documents.

Example request:

```json
{
  "query": "space shuttle launch"
}
```

Example response:

```json
{
  "query": "space shuttle launch",
  "cache_hit": false,
  "result": [...],
  "dominant_cluster": 16
}
```

---

## GET /cache/stats

Returns statistics about the semantic cache.

---

## DELETE /cache

Clears all cached queries.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/semantic-search-system.git
cd semantic-search-system
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Build the Index

Generate embeddings and clustering model:

```bash
python scripts/build_index.py
```

This step creates:

vector_store.pkl
cluster_model.pkl

---

# Run the API

Start the FastAPI service:

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

---

# Future Improvements

Possible enhancements include:

* approximate FAISS indexes for larger datasets
* persistent cache storage
* cluster visualization
* Docker containerization for deployment

---

# Author

BATTULA GURUPRASAD SPOORTH – Semantic Search System
