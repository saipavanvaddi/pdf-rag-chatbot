
from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def generate_embeddings(chunks):

    # Extract text from each chunk
    texts = [
        chunk["chunk_text"]
        for chunk in chunks
    ]

    # Convert text into numerical vectors
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings
