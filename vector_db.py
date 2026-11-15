
import chromadb
import uuid

from embeddings import model


# STEP 1: Create persistent ChromaDB client

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# STEP 2: Create or connect to collection

collection = client.get_or_create_collection(
    name="pdf_documents",
    metadata={
        "hnsw:space": "cosine"
    }
)


# STEP 3: Store PDF chunks and embeddings

def store_chunks(chunks):

    if not chunks:
        return 0

    texts = [
        chunk["chunk_text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Remove previous chunks with the same filename
    # This prevents duplicate uploads of the same filename

    filenames = list({
        chunk["filename"]
        for chunk in chunks
    })

    for filename in filenames:

        collection.delete(
            where={
                "filename": filename
            }
        )

    # Generate unique IDs

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    # Prepare metadata

    metadatas = [
        {
            "filename": chunk["filename"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"]
        }
        for chunk in chunks
    ]

    # Store everything in ChromaDB

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return len(chunks)


# STEP 4: Search relevant chunks

def search_chunks(question, top_k=5):

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    # Convert user question into embedding

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    # Search ChromaDB

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=min(top_k, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results
