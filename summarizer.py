# summarizer.py

from transformers import pipeline
import torch
from dotenv import load_dotenv
import os

# Load environment variables to get the Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_HUB_TOKEN not found. Please add it to your .env file.")

# Initialize the text generation pipeline with the Llama 3.2 1B model
# This will download the model on the first run.
# We're using device="cpu" to ensure it runs without a dedicated GPU.
print("Loading summarization model: unsloth/Llama-3.2-1B-Instruct...")
try:
    summarizer_pipeline = pipeline(
        "text-generation",
        model="unsloth/Llama-3.2-1B-Instruct",
        torch_dtype=torch.bfloat16, # Use bfloat16 for better performance
        device="cpu", # Force CPU to avoid potential hardware issues
        token=HF_TOKEN # Pass the token for gated model access
    )
    print("Summarization model loaded successfully.")
except Exception as e:
    print(f"Failed to load the model. Error: {e}")
    # Provide a helpful message if the user hasn't accepted the license
    if "GatedRepo" in str(e):
        print("\n--- IMPORTANT ---")
        print("This model is gated. Please ensure you have accepted the license on Hugging Face:")
        print("https://huggingface.co/unsloth/Llama-3.2-1B-Instruct")
        print("-----------------")
    summarizer_pipeline = None

def summarize_text(query: str, context_chunks: list[str]) -> str:
    """
    Generates a summary/answer based on a query and a list of context chunks.

    Args:
        query: The user's original search query.
        context_chunks: A list of the most relevant text chunks from the document.

    Returns:
        A string containing the generated summary/answer.
    """
    if not summarizer_pipeline:
        return "Error: The summarization model could not be loaded. Please check the terminal for errors."

    # Combine the context chunks into a single string
    context = "\n\n".join(context_chunks)

    # This is our carefully crafted prompt for the Llama model
    # It uses the official Llama 3 prompt format
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Your task is to answer the user's query based *only* on the provided context. If the context does not contain the answer, say that you cannot answer based on the given information."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
    ]
    
    # The pipeline's apply_chat_template method formats the messages correctly
    prompt = summarizer_pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    print("\n--- Generating Summary ---")
    try:
        response = summarizer_pipeline(
            prompt,
            max_new_tokens=256,  # Limit the length of the generated summary
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=summarizer_pipeline.tokenizer.eos_token_id
        )
        
        generated_text = response[0]['generated_text']
        
        # Clean the output to only get the model's response
        # The response will start after the <|start_header_id|>assistant<|end_header_id|> tag
        assistant_tag = "<|start_header_id|>assistant<|end_header_id|>"
        summary = generated_text.split(assistant_tag)[-1].strip()
        
        print("--- Summary Generated Successfully ---")
        return summary

    except Exception as e:
        print(f"An error occurred during summarization: {e}")
        return f"Error: Failed to generate summary. {e}"