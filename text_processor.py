# text_processor.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def split_and_clean_text(text: str) -> list[str]:
    """
    Splits the text into chunks and performs basic cleaning.

    Args:
        text: The raw text extracted from a document.

    Returns:
        A list of cleaned text chunks.
    """
    # 1. Basic cleaning
    # Replace multiple newlines with a single one to consolidate paragraphs
    text = re.sub(r'\n+', '\n', text).strip()
    # You can add more cleaning steps here, e.g., removing headers/footers if you can identify them.

    # 2. Initialize the text splitter
    # RecursiveCharacterTextSplitter is good at keeping related pieces of text together.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # The target size of each chunk in characters
        chunk_overlap=50,  # The number of characters to overlap between chunks
        length_function=len,
        is_separator_regex=False,
    )

    # 3. Split the text into chunks
    chunks = text_splitter.split_text(text)
    
    # 4. Filter out any very short or empty chunks that might result from splitting
    cleaned_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 10]

    print(f"Original text split into {len(cleaned_chunks)} chunks.")
    return cleaned_chunks