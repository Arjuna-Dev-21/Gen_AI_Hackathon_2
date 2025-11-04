# app.py

import streamlit as st
from text_extractor import extract_text_from_file
from text_processor import split_and_clean_text
from embedding_generator import model as embedding_model
from vector_store_manager import create_and_populate_faiss_index, search_faiss_index
from summarizer import summarize_text # <-- Import our new function

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Document Search & Summarization",
    page_icon="📚",
    layout="wide"
)

# --- App Title and Description ---
st.title("📚 AI-Powered Document Search and Summarization")
st.markdown("Welcome! This tool ingests a document, performs a semantic search, and generates a concise summary of the relevant sections.")

# --- Initialize Session State ---
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'retrieved_chunks' not in st.session_state:
    st.session_state.retrieved_chunks = None # To store search results for summarization

# --- File Uploader ---
with st.sidebar:
    st.header("1. Upload Your Document")
    uploaded_file = st.file_uploader(
        "Choose a document (.pdf, .docx, or .txt)",
        type=['pdf', 'docx', 'txt']
    )

    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document... This may take a few minutes."):
                st.session_state.text_chunks = None
                st.session_state.faiss_index = None
                st.session_state.retrieved_chunks = None

                extracted_text = extract_text_from_file(uploaded_file)
                if not extracted_text:
                    st.error("Failed to extract text.")
                    st.stop()

                text_chunks = split_and_clean_text(extracted_text)
                st.session_state.text_chunks = text_chunks

                chunk_embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
                
                faiss_index = create_and_populate_faiss_index(chunk_embeddings)
                st.session_state.faiss_index = faiss_index
            
            st.success("Document processed successfully!")
            st.info("You can now proceed to the search section.")
        else:
            st.warning("Please upload a document first.")

# --- Main Content Area ---
st.header("2. Search and Summarize Your Document")

if st.session_state.faiss_index is None:
    st.info("Please upload and process a document using the sidebar to enable search.")
else:
    query = st.text_input("Enter your search query:", placeholder="e.g., What are the main challenges of AI adoption?")
    top_k = st.slider("Number of relevant chunks to retrieve:", 1, 10, 3)

    if query:
        with st.spinner("Searching for relevant chunks..."):
            query_embedding = embedding_model.encode([query])
            distances, indices = search_faiss_index(st.session_state.faiss_index, query_embedding, top_k)
            
            # Store retrieved chunks in session state
            st.session_state.retrieved_chunks = [st.session_state.text_chunks[i] for i in indices[0]]

            st.subheader("Search Results:")
            if not st.session_state.retrieved_chunks:
                st.warning("No relevant chunks found.")
            else:
                for i, chunk in enumerate(st.session_state.retrieved_chunks):
                    st.info(f"**Result {i+1}**\n\n{chunk}")
        
        # --- NEW SECTION: Summarization Button ---
        if st.session_state.retrieved_chunks:
            st.subheader("3. Generate Summary")
            if st.button("Summarize Retrieved Content"):
                with st.spinner("Generating summary with Llama 3.2... This can take a moment."):
                    summary = summarize_text(query, st.session_state.retrieved_chunks)
                    st.success("Summary generated!")
                    st.markdown("### Summary / Answer")
                    st.write(summary)
        # --- END OF NEW SECTION ---