import fitz

def extract_pdf_text(pdf_bytes, filename):

    # Open PDF from uploaded file bytes
    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )
    pages = []

    # Read every page
    for page_number, page in enumerate(pdf):

        # Extract text from current page
        text = page.get_text("text")
        pages.append({
            "filename": filename,
            "page_number": page_number + 1,
            "text": text
        })
    pdf.close()

    return pages