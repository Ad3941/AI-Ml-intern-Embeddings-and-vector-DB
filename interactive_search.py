import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class InteractiveSearchEngine:
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
        """Pre-loads and vectorizes documents into memory."""
        print("[Info] Chunking source documents...")
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

        print("[Info] Generating embedding matrix...")
        self.embedding_matrix = self.vectorizer.fit_transform(self.document_chunks).toarray()
        print(f"[Info] Successfully indexed {len(self.document_chunks)} chunks. Ready for queries!\n")

    def search_top_k(self, query: str, k: int = 2) -> list[dict]:
        """Transforms query and returns top-k matching chunks."""
        query_vector = self.vectorizer.transform([query]).toarray()
        similarity_scores = cosine_similarity(query_vector, self.embedding_matrix).flatten()
        sorted_indices = np.argsort(similarity_scores)[::-1]

        results = []
        for rank, idx in enumerate(sorted_indices[:k], start=1):
            score = similarity_scores[idx]
            if score <= 0:
                continue
            results.append({
                "rank": rank,
                "score": round(float(score), 4),
                "chunk_text": self.document_chunks[idx],
                "metadata": self.metadata[idx]
            })
        return results

if __name__ == "__main__":
    # Initialize Engine
    engine = InteractiveSearchEngine()

    # Pre-load sample knowledge base documents
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

    engine.ingest_documents(documents)

    # CLI Loop Banner
    print("==================================================")
    print("      Interactive Top-K Search Engine CLI         ")
    print("      Type your query and press Enter.            ")
    print("      Type 'exit' or 'quit' to terminate.         ")
    print("==================================================")

    # Interactive Command-Line Loop
    while True:
        try:
            user_query = input("\nQuery > ").strip()
            
            # Check for exit commands
            if user_query.lower() in ["exit", "quit"]:
                print("Exiting search engine. Goodbye!")
                break
            
            # Ignore empty inputs
            if not user_query:
                continue

            # Execute search
            top_matches = engine.search_top_k(user_query, k=2)

            if not top_matches:
                print("No relevant results found (Score: 0.0).")
                continue

            # Display formatted search results
            print(f"\n--- Top Results for: '{user_query}' ---")
            for match in top_matches:
                print(f"[{match['rank']}] Score: {match['score']} | Doc ID: {match['metadata']['doc_id']}")
                print(f"    Content: {match['chunk_text']}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting search engine. Goodbye!")
            break