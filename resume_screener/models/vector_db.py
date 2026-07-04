import faiss
import numpy as np

class HybridVectorDB:
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        # Use Inner Product for Cosine Similarity (vectors must be normalized)
        self.index = faiss.IndexHNSWFlat(embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = 16
        self.candidates = []
        
    def normalize_vector(self, vec):
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def add_resume(self, candidate_name, embedding, skills):
        emb_norm = self.normalize_vector(embedding).reshape(1, -1)
        self.index.add(emb_norm)
        
        self.candidates.append({
            "name": candidate_name,
            "skills": set(skills),
            "original_embedding": embedding
        })
        
    def hybrid_search(self, jd_embedding, jd_skills, k=10, alpha=0.7):
        if not self.candidates:
            return []
            
        k = min(k, len(self.candidates))
        jd_norm = self.normalize_vector(jd_embedding).reshape(1, -1)
        
        # Dense search using FAISS HNSW
        distances, indices = self.index.search(jd_norm, k)
        
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
                
            candidate = self.candidates[idx]
            
            # Vector score (normalized inner product equates to cosine similarity, max 1.0)
            vector_score = distances[0][rank]
            
            # Sparse (Skill) score
            if not jd_skills:
                skill_score = 1.0
            else:
                overlap = candidate["skills"].intersection(jd_skills)
                skill_score = len(overlap) / len(jd_skills)
                
            # Hybrid Score calculation
            hybrid_score = (alpha * vector_score) + ((1 - alpha) * skill_score)
            
            results.append({
                "candidate_idx": idx,
                "name": candidate["name"],
                "vector_score": float(vector_score),
                "skill_score": float(skill_score),
                "hybrid_score": float(hybrid_score)
            })
            
        # Sort by hybrid score descending
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results
