from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, threshold=0.85):

        self.cache = []
        self.threshold = threshold

        self.hit = 0
        self.miss = 0

    def lookup(self, embedding):

        for entry in self.cache:

            sim = cosine_similarity(
                [embedding],
                [entry["embedding"]]
            )[0][0]

            if sim >= self.threshold:
                self.hit += 1
                return entry, sim

        self.miss += 1
        return None, None

    def add(self, query, embedding, result, cluster):

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = self.hit + self.miss

        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit,
            "miss_count": self.miss,
            "hit_rate": self.hit / total if total else 0
        }

    def clear(self):
        self.cache = []
        self.hit = 0
        self.miss = 0