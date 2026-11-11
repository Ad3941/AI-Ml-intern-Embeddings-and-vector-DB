import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TopKSearchEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_chunks = []
        self.embedding_matrix = None
        self.metadata = []

    def chunk_text(self, text: str, chunk_size: int = 250, overlap: int = 30) -> list[str]:
        """Splits text into overlapping sliding-window chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += (chunk_size - overlap)
        return [c for c in chunks if c]

    def ingest_documents(self, documents: list[dict]):
        """Step 1 & 2: Chunk documents and create the embedding matrix."""
        print("[1/3] Chunking source documents...")
        all_chunks = []
        all_meta = []

        for doc in documents:
            doc_id = doc.get("id", "doc_default")
            chunks = self.chunk_text(doc["text"])
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({"doc_id": doc_id, "chunk_index": idx})

        self.document_chunks = all_chunks
        self.metadata = all_meta

        print("[2/3] Generating embedding matrix...")
        # Fit vectorizer and transform chunks into a matrix of numerical vectors
        self.embedding_matrix = self.vectorizer.fit_transform(self.document_chunks).toarray()
        print(f"Successfully indexed {len(self.document_chunks)} chunks. Matrix shape: {self.embedding_matrix.shape}")

    def search_top_k(self, query: str, k: int = 2) -> list[dict]:
        """Step 3 & 4: Embed query, compute similarities, and return top-k matches."""
        print(f"\n[3/3] Executing Top-{k} search for query: '{query}'")
        
        # Transform the query using the fitted vectorizer (ensures identical vector space)
        query_vector = self.vectorizer.transform([query]).toarray()

        # Compute cosine similarity between the query vector and all document vectors
        similarity_scores = cosine_similarity(query_vector, self.embedding_matrix).flatten()

        # Sort indices by highest similarity score in descending order
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Extract top-k results
        results = []
        for rank, idx in enumerate(sorted_indices[:k], start=1):
            score = similarity_scores[idx]
            if score <= 0:
                continue  # Skip irrelevant orthogonal results
            results.append({
                "rank": rank,
                "score": round(float(score), 4),
                "chunk_text": self.document_chunks[idx],
                "metadata": self.metadata[idx]
            })

        return results

if __name__ == "__main__":
    # Initialize Engine
    engine = TopKSearchEngine()

    # Sample source database documents
    documents = [
        {
            "id": "doc_vectors",
            "text": "Vector embeddings map unstructured strings into mathematical arrays where geometric proximity represents semantic meaning."
        },
        {
            "id": "doc_chunking",
            "text": "Document chunking strategies break long texts into overlapping windows to preserve local context without breaking thoughts."
        },
        {
            "id": "doc_search",
            "text": "Top-k similarity search computes cosine distances between query vectors and stored embeddings to retrieve relevant text segments."
        }
    ]

    # Ingest data and build the matrix
    engine.ingest_documents(documents)

    # Run Top-K Search Query
    query_string = "How do I calculate cosine distances for retrieval?"
    top_matches = engine.search_top_k(query_string, k=2)

    # Display formatted output
    print("\n--- Search Results ---")
    for match in top_matches:
        print(f"Rank {match['rank']} | Score: {match['score']} | Doc ID: {match['metadata']['doc_id']}")
        print(f"Content: {match['chunk_text']}\n")