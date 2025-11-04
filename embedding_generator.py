# embedding_generator.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the sentence-transformer model.
# This will be downloaded on the first run and cached for future use.
# You will see the download progress in your terminal.
print("Loading sentence transformer model: all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

def generate_embeddings(text_chunks: list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of text chunks.

    Args:
        text_chunks: A list of strings (text chunks).

    Returns:
        A numpy array of embeddings, where each row is the embedding for a chunk.
    """
    # The model.encode() method is highly optimized for batch processing.
    # It will show a progress bar in the terminal as it works.
    print("Generating embeddings for text chunks...")
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    print("Embeddings generated.")
    return embeddings