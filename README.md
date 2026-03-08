# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a **semantic search system** built on the **20 Newsgroups dataset**.
Instead of relying on traditional keyword-based search, the system retrieves documents based on **semantic similarity** using vector embeddings.

The system also integrates **fuzzy clustering** and a **semantic caching layer** to improve search efficiency and demonstrate the design of a real-world ML system.

The application is exposed through a **FastAPI service**, allowing users to query the system through REST endpoints.

---

# System Architecture

User Query
↓
SentenceTransformer Embedding Model
↓
Semantic Cache Check
↓
Vector Similarity Search (FAISS)
↓
Fuzzy Clustering (Gaussian Mixture Model)
↓
API Response

---

# Dataset

This project uses the **20 Newsgroups Dataset**, a widely used benchmark dataset for text classification and retrieval tasks.

The dataset contains approximately **20,000 documents across 20 topic categories**, including:

* sci.space
* comp.graphics
* rec.sport.baseball
* talk.politics.misc
* comp.sys.ibm.pc.hardware

Dataset Source:

https://archive.ics.uci.edu/

Download link:

[https://archive.ics.uci.edu/dataset/113/twenty+newsgroups]

After downloading, extract the dataset into the project directory so that the folder structure looks like:

```
semantic-search-system/
│
├── 20_newsgroups/
├── app/
├── scripts/
```

---

# Technologies Used

* Python
* SentenceTransformers
* FAISS
* Scikit-learn
* FastAPI
* NumPy

---

# Embedding Model

The system uses the **SentenceTransformer model `all-MiniLM-L6-v2`**.

Reasons for choosing this model:

* Lightweight and efficient
* Strong semantic representation for sentences
* Suitable for similarity search tasks

Each document is converted into a **384-dimensional embedding vector**.

---

# Vector Database

The embeddings are stored in **FAISS (Facebook AI Similarity Search)**.

FAISS enables:

* Efficient nearest-neighbor search
* Fast similarity queries across large datasets
* Scalable semantic retrieval

When a user query is received, its embedding is compared against all stored vectors to retrieve the **most semantically similar documents**.

---

# Fuzzy Clustering

To analyze topic structure within the dataset, the system uses a **Gaussian Mixture Model (GMM)**.

Unlike hard clustering algorithms like K-Means, GMM provides **probabilistic cluster memberships**.

Example:

A document discussing space shuttle propulsion may belong to:

Cluster 5 → 0.52
Cluster 12 → 0.31
Cluster 3 → 0.17

This reflects real-world scenarios where documents may belong to multiple related topics.

---

# Semantic Cache

Traditional caching systems only work for **identical queries**.

This project implements a **semantic cache**, which stores previous queries and detects **similar queries using cosine similarity** between embeddings.

Example:

Query 1

```
space shuttle launch
```

Query 2

```
space shuttle mission
```

Even though the queries are not identical, their embeddings are similar, allowing the system to reuse cached results.

This reduces repeated computation and improves performance.

---

# API Endpoints

The system exposes three REST endpoints.

---

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

Example response:

```json
{
  "total_entries": 3,
  "hit_count": 1,
  "miss_count": 2,
  "hit_rate": 0.33
}
```

---

## DELETE /cache

Clears the semantic cache.

Example response:

```json
{
  "message": "cache cleared"
}
```

---

# Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/semantic-search-system.git
cd semantic-search-system
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment (Windows):

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Build the Index

Generate document embeddings and clustering model:

```
python scripts/build_index.py
```

This step creates:

```
vector_store.pkl
cluster_model.pkl
```

---

# Run the API

Start the FastAPI server:

```
uvicorn app.main:app --reload
```

Open the interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

# Future Improvements

Possible enhancements include:

* using approximate FAISS indexes for larger datasets
* adding persistent cache storage
* visualizing cluster distributions
* deploying the system using Docker

---

# Author

AI/ML Engineer Project – Semantic Search System
