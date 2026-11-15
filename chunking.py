
def create_chunks(pages, chunk_size=1000, overlap=200):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be between 0 and chunk_size - 1"
        )

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "chunk_text": chunk_text,
                    "chunk_index": len(chunks)
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks
