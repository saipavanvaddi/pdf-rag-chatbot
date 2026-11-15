
import streamlit as st

from pdf_processor import extract_pdf_text


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
