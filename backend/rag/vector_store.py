import os
import re
import math
import chromadb
from typing import List, Dict, Any

class BiomedicalVectorStore:
    """
    RAG vector store wrapper using ChromaDB for persistent storage
    and a robust, local pure-Python TF-IDF text similarity engine
    to bypass Hugging Face Hub network dependencies, timeouts, and rate limits.
    """
    
    def __init__(self, persist_dir: str = "data/chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize chroma client for document storage and queries
        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        except Exception:
            self.client = chromadb.EphemeralClient()
            
        self.collection = self.client.get_or_create_collection(
            name="biomedical_literature",
            metadata={"hnsw:space": "cosine"}
        )

    def _tokenize(self, text: str) -> List[str]:
        """Splits text into lowercased alphanumeric tokens."""
        return re.findall(r'[a-zA-Z0-9_]+', text.lower())

    def _compute_cosine_similarity(self, query: str, doc: str) -> float:
        """
        Computes cosine similarity between two text strings using a bag-of-words representation.
        """
        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(doc)
        
        if not q_tokens or not d_tokens:
            return 0.0
            
        # Build frequency maps
        q_freq = {}
        for token in q_tokens:
            q_freq[token] = q_freq.get(token, 0) + 1
            
        d_freq = {}
        for token in d_tokens:
            d_freq[token] = d_freq.get(token, 0) + 1
            
        # Compute dot product and magnitude norms
        dot_product = 0.0
        for token, freq in q_freq.items():
            if token in d_freq:
                dot_product += freq * d_freq[token]
                
        q_norm = math.sqrt(sum(f ** 2 for f in q_freq.values()))
        d_norm = math.sqrt(sum(f ** 2 for f in d_freq.values()))
        
        if q_norm == 0 or d_norm == 0:
            return 0.0
            
        return dot_product / (q_norm * d_norm)

    def add_publications(self, publications: List[Dict[str, Any]], mutation_name: str):
        """
        Segments and indexes scientific literature abstracts into the vector store.
        """
        if not publications:
            return

        documents = []
        metadatas = []
        ids = []
        
        for idx, pub in enumerate(publications):
            pmid = pub.get("pmid", f"gen_{idx}")
            title = pub.get("title", "Unknown Publication")
            abstract = pub.get("abstract", "")
            journal = pub.get("journal", "Unknown Journal")
            year = pub.get("year", "Unknown Year")
            authors = pub.get("authors", "Unknown Authors")
            
            if not abstract:
                continue
                
            # Chunking abstract into sentences for finer search granularity
            sentences = [s.strip() for s in abstract.split(".") if s.strip()]
            
            for s_idx, sentence in enumerate(sentences):
                doc_id = f"{mutation_name}_{pmid}_{s_idx}"
                documents.append(sentence)
                metadatas.append({
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "year": str(year),
                    "authors": authors,
                    "mutation": mutation_name,
                    "full_source": f"{authors} ({year}). {title}. {journal}. PMID: {pmid}"
                })
                ids.append(doc_id)
                
        if documents:
            # Upsert into Chroma (using dummy embeddings since we calculate cosine similarity locally)
            dummy_embeddings = [[0.1] * 128 for _ in range(len(documents))]
            self.collection.upsert(
                embeddings=dummy_embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def search_evidence(self, query: str, limit: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves all documents matching the mutation and performs cosine similarity search.
        """
        # Fetch all documents in collection
        results = self.collection.get(include=["documents", "metadatas"])
        
        hits = []
        if results and results.get("documents"):
            docs = results["documents"]
            metas = results["metadatas"]
            
            for i in range(len(docs)):
                doc_text = docs[i]
                meta = metas[i]
                
                # Compute local TF-IDF style similarity score
                similarity = self._compute_cosine_similarity(query, doc_text)
                
                # Filter out low-matching results to keep search qualitative
                if similarity > 0.05:
                    hits.append({
                        "text": doc_text,
                        "metadata": meta,
                        "similarity": round(similarity, 3),
                        "citation": meta.get("full_source", "Unknown Citation")
                    })
                    
        # Sort hits by descending similarity score
        hits = sorted(hits, key=lambda x: x["similarity"], reverse=True)
        return hits[:limit]
