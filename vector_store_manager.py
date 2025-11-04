# vector_store_manager.py

import faiss
import numpy as np

def create_and_populate_faiss_index(embeddings: np.ndarray):
    """
    Creates a FAISS index and populates it with the given embeddings.

    Args:
        embeddings: A numpy array of document chunk embeddings.

    Returns:
        A trained and populated FAISS index object.
    """
    # First, get the dimension of the embeddings (e.g., 384 for all-MiniLM-L6-v2)
    dimension = embeddings.shape[1]

    # We are using the IndexFlatL2 index type. This is a basic but effective index
    # that performs an exhaustive search. It's perfect for CPU-based applications
    # with a moderate number of vectors.
    print(f"Creating FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)

    # FAISS requires the embeddings to be in a specific format (float32).
    # We ensure this by converting the numpy array.
    embeddings_float32 = embeddings.astype('float32')

    # Add the embeddings to the index.
    print(f"Adding {embeddings.shape[0]} vectors to the index...")
    index.add(embeddings_float32)

    print(f"FAISS index created successfully. Total vectors in index: {index.ntotal}")
    return index

def search_faiss_index(index, query_embedding: np.ndarray, top_k: int):
    """
    Searches the FAISS index for the most similar vectors to a query embedding.

    Args:
        index: The populated FAISS index.
        query_embedding: The embedding of the user's search query.
        top_k: The number of top results to return.

    Returns:
        A tuple containing the distances and the indices of the top_k most similar chunks.
    """
    # FAISS expects a 2D array for queries, so we reshape our 1D query vector.
    query_embedding_float32 = query_embedding.astype('float32').reshape(1, -1)
    
    # Perform the search
    distances, indices = index.search(query_embedding_float32, top_k)
    
    return distances, indices