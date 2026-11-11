# Top-K Text Search

`top_k_search.py` is a small in-memory search example. It splits documents into
overlapping text chunks, represents each chunk with TF-IDF, and ranks the chunks
by cosine similarity to a search query.

Despite the variable name `embedding_matrix`, this example uses TF-IDF vectors,
not a machine-learning embedding model. Its results are based on word overlap.

## Requirements

Python 3.9 or newer, plus NumPy and scikit-learn:

```bash
python -m pip install numpy scikit-learn
```

## Run the example

From this folder, run:

```bash
python top_k_search.py
```

The script indexes three sample documents, searches for
“How do I calculate cosine distances for retrieval?”, and prints up to two
matching chunks with their scores and document IDs.

## Use the search engine

Import `TopKSearchEngine`, ingest documents as dictionaries containing a
`text` field and an optional `id`, then search:

```python
from top_k_search import TopKSearchEngine

engine = TopKSearchEngine()
engine.ingest_documents([
    {"id": "guide", "text": "Text to make searchable goes here."},
])

matches = engine.search_top_k("search phrase", k=2)
```

Each match contains a 1-based `rank`, a cosine similarity `score` rounded to
four decimal places, the matching `chunk_text`, and `metadata` containing the
source `doc_id` and `chunk_index`. Chunks with a zero similarity score are
omitted.

By default, text is split into 250-character chunks with 30 characters of
overlap. The chunk size and overlap can be changed through `chunk_text`.
Search is performed against the documents ingested into the engine in memory;
the example does not persist an index.
