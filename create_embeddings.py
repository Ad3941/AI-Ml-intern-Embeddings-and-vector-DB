import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class DocumentEmbeddingEngine:
    def __init__(self):
        # Initialize vectorizer as our embedding generator
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_chunks = []
        self.embeddings = None
        self.metadata = []

    def chunk_text(self, text: str, chunk_size: int = 250, overlap: int = 30) -> list[str]:
        """Splits text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += (chunk_size - overlap)
        return [c for c in chunks if c]

    def create_embeddings(self, documents: list[dict]):
        """Step-by-step execution to chunk documents and generate vector embeddings."""
        print("[1/2] Chunking source documents...")
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

        print("[2/2] Generating vector embeddings for chunks...")
        # Fit and transform chunks into numerical embedding vectors
        self.embeddings = self.vectorizer.fit_transform(self.document_chunks).toarray()
        
        print(f"Successfully generated embeddings for {len(self.document_chunks)} chunks!")
        return self.embeddings

if __name__ == "__main__":
    # Initialize Engine
    engine = DocumentEmbeddingEngine()

    # Sample source documents to embed
    sample_documents = [
        {
            "id": "doc_1",
            "text": "Vector embeddings represent text as numerical arrays where spatial proximity reflects semantic meaning."
        },
        {
            "id": "doc_2",
            "text": "Chunking strategies ensure long documents are partitioned into focused, context-aware segments."
        }
    ]

    # Execute embedding creation
    vector_matrix = engine.create_embeddings(sample_documents)

    # Inspect outputs
    print(f"\nEmbedding Matrix Shape: {vector_matrix.shape}")
    print(f"Sample Vector Representation (First 5 values of chunk 1): {vector_matrix[0][:5]}")