
import streamlit as st

from pdf_processor import extract_pdf_text
from chunking import create_chunks
from embeddings import generate_embeddings
from vector_db import store_chunks, search_chunks, collection


# Page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# Application heading
st.title("📚 PDF RAG Chatbot")

st.write(
    "Upload your PDF and extract its text."
)


# PDF upload
uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

chunk_size = st.slider(
    "Chunk Size",
    min_value=200,
    max_value=2000,
    value=1000,
    step=100
)

overlap = st.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=500,
    value=200,
    step=50
)


if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button("Extract PDF Text"):

        pdf_bytes = uploaded_file.getvalue()

        pages = extract_pdf_text(
            pdf_bytes,
            uploaded_file.name
        )

        if pages:

            st.success(
                f"Successfully extracted {len(pages)} pages!"
            )

            for page in pages:

                with st.expander(
                    f"Page {page['page_number']}"
                ):

                    st.write(page["text"])

        else:

            st.warning(
                "No selectable text found in this PDF."
            )
    if st.button("Process PDF"):

        pdf_bytes = uploaded_file.getvalue()

        pages = extract_pdf_text(
            pdf_bytes,
            uploaded_file.name
        )

        if not pages:
            st.warning("No selectable PDF text found.")
            st.stop()

        st.success(
            f"Extracted text from {len(pages)} pages."
        )

        chunks = create_chunks(
            pages,
            chunk_size=chunk_size,
            overlap=overlap
        )

        st.success(
            f"Generated {len(chunks)} chunks."
        )

        st.subheader("Generated Chunks")

        for chunk in chunks:

            with st.expander(
                f"Chunk {chunk['chunk_index'] + 1} "
                f"- Page {chunk['page_number']}"
            ):

                st.write(chunk["chunk_text"])

                st.caption(
                    f"Characters: {len(chunk['chunk_text'])}"
                )
    if st.button("Process PDF and Generate Embeddings"):

        # STEP 1: Extract PDF text

        pdf_bytes = uploaded_file.getvalue()

        pages = extract_pdf_text(
            pdf_bytes,
            uploaded_file.name
        )

        if not pages:

            st.warning("No selectable PDF text found.")

            st.stop()

        st.success(
            f"Extracted {len(pages)} pages."
        )


        # STEP 2: Create chunks

        chunks = create_chunks(
            pages,
            chunk_size=chunk_size,
            overlap=overlap
        )

        st.success(
            f"Generated {len(chunks)} chunks."
        )


        # STEP 3: Generate embeddings

        with st.spinner("Generating embeddings..."):

            embeddings = generate_embeddings(chunks)


        st.success("Embeddings generated successfully!")


        # STEP 4: Display results

        st.subheader("Embedding Results")

        st.write("Total chunks:", len(chunks))

        st.write("Total embeddings:", len(embeddings))

        st.write("Vector dimensions:", len(embeddings[0]))

        st.write("Embedding matrix shape:", embeddings.shape)


        # Display first vector

        st.subheader("First Chunk Embedding")

        st.write(
            embeddings[0][:10].tolist()
        )


        # Display chunks

        st.subheader("Generated Chunks")

        for chunk in chunks:

            with st.expander(
                f"Chunk {chunk['chunk_index'] + 1} "
                f"- Page {chunk['page_number']}"
            ):

                st.write(chunk["chunk_text"])

                st.caption(
                    f"Characters: {len(chunk['chunk_text'])}"
                )
    if st.button("Process PDF and Save to Database"):

        with st.spinner("Extracting PDF text..."):

            pages = extract_pdf_text(
                uploaded_file.getvalue(),
                uploaded_file.name
            )

        if not pages:

            st.warning("No selectable text found.")

            st.stop()

        st.success(
            f"Extracted {len(pages)} pages."
        )


        # Create chunks

        chunks = create_chunks(
            pages,
            chunk_size=chunk_size,
            overlap=overlap
        )

        st.success(
            f"Generated {len(chunks)} chunks."
        )


        # Save to ChromaDB

        with st.spinner("Generating embeddings and saving..."):

            saved_count = store_chunks(chunks)

        st.success(
            f"Successfully saved {saved_count} chunks to ChromaDB!"
        )


# --------------------------------
# DATABASE INFORMATION
# --------------------------------

st.divider()

st.subheader("Vector Database Information")

total_chunks = collection.count()

st.metric(
    "Total Stored Chunks",
    total_chunks
)


# --------------------------------
# QUESTION SEARCH
# --------------------------------

st.divider()

st.subheader("Search Your PDF")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What are the maintenance charges?"
)


if st.button("Search Relevant Chunks"):

    if not question.strip():

        st.warning("Please enter a question.")

    elif collection.count() == 0:

        st.warning("Please upload and process a PDF first.")

    else:

        with st.spinner("Searching ChromaDB..."):

            results = search_chunks(
                question,
                top_k=5
            )

        st.success("Search completed!")

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        st.subheader("Retrieved Results")

        for i, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances)
        ):

            with st.expander(
                f"Result {i + 1} - "
                f"{metadata['filename']} - "
                f"Page {metadata['page_number']}"
            ):

                st.write(document)

                st.caption(
                    f"Cosine distance: {distance:.4f}"
                )

                st.caption(
                    f"Chunk index: {metadata['chunk_index']}"
                )
