# text_extractor.py

import PyPDF2
import docx
import io

def _extract_text_from_pdf(file_stream):
    """Extracts text from a PDF file stream."""
    pdf_reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def _extract_text_from_docx(file_stream):
    """Extracts text from a DOCX file stream."""
    doc = docx.Document(file_stream)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def _extract_text_from_txt(file_stream):
    """Extracts text from a TXT file stream."""
    # The file_stream needs to be decoded from bytes to string
    return file_stream.read().decode("utf-8")

def extract_text_from_file(uploaded_file):
    """
    Detects the file type and extracts text from the uploaded file.

    Args:
        uploaded_file: An uploaded file object from Streamlit.

    Returns:
        A string containing the extracted text, or None if the file type is unsupported.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # We use io.BytesIO to treat the uploaded file's bytes as a file-like object
    file_stream = io.BytesIO(uploaded_file.getvalue())

    if file_extension == 'pdf':
        return _extract_text_from_pdf(file_stream)
    elif file_extension == 'docx':
        return _extract_text_from_docx(file_stream)
    elif file_extension == 'txt':
        return _extract_text_from_txt(file_stream)
    else:
        # You can add more file types here if needed
        print(f"Unsupported file type: {file_extension}")
        return None