#!/usr/bin/env python3
"""
Example usage of parallel sentence encoding functions.
Demonstrates different ways to encode sentences efficiently.
"""

import time
from localquwen import encode_sentences_parallel, encode_sentences_with_threading
from transformers import AutoTokenizer, AutoModel


def main():
    print("Loading model and tokenizer...")
    start_time = time.time()
    
    # Load model and tokenizer once for reuse (more efficient)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Example 1: Basic sentence encoding
    print("\n=== Example 1: Basic Sentence Encoding ===")
    sentences = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Explain machine learning algorithms",
        "What is deep learning?",
        "Describe natural language processing"
    ]
    
    start_time = time.time()
    embeddings_dict = encode_sentences_parallel(
        sentences=sentences,
        model=model,
        tokenizer=tokenizer,
        batch_size=8,
        normalize=True
    )
    encoding_time = time.time() - start_time
    
    print(f"Encoded {len(sentences)} sentences in {encoding_time:.3f} seconds")
    for sentence, embedding in list(embeddings_dict.items())[:2]:  # Show first 2
        print(f"'{sentence[:50]}...': shape {embedding.shape}")
    
    # Example 2: Large batch processing
    print("\n=== Example 2: Large Batch Processing ===")
    large_sentences = [
        f"This is sentence number {i} for testing batch processing efficiency."
        for i in range(100)
    ]
    
    start_time = time.time()
    large_embeddings = encode_sentences_parallel(
        sentences=large_sentences,
        model=model,
        tokenizer=tokenizer,
        batch_size=32,  # Larger batch for efficiency
        normalize=True
    )
    large_encoding_time = time.time() - start_time
    
    print(f"Encoded {len(large_sentences)} sentences in {large_encoding_time:.3f} seconds")
    print(f"Average time per sentence: {large_encoding_time/len(large_sentences)*1000:.2f} ms")
    
    # Example 3: With instruction formatting
    print("\n=== Example 3: Instruction-based Encoding ===")
    query_sentences = [
        "Python programming tutorial",
        "Machine learning best practices", 
        "Data science workflow"
    ]
    
    start_time = time.time()
    instructed_embeddings = encode_sentences_parallel(
        sentences=query_sentences,
        model=model,
        tokenizer=tokenizer,
        task_description="Given a search query, retrieve relevant educational content",
        add_instruction=True,
        batch_size=8,
        normalize=True
    )
    instructed_time = time.time() - start_time
    
    print(f"Encoded {len(query_sentences)} instructed queries in {instructed_time:.3f} seconds")
    
    # Example 4: Threading comparison (for very large datasets)
    print("\n=== Example 4: Threading vs Batch Processing ===")
    test_sentences = [f"Test sentence {i} for performance comparison." for i in range(50)]
    
    # Standard batch processing
    start_time = time.time()
    batch_results = encode_sentences_parallel(
        sentences=test_sentences,
        model=model,
        tokenizer=tokenizer,
        batch_size=16
    )
    batch_time = time.time() - start_time
    
    # Threading approach
    start_time = time.time()
    thread_results = encode_sentences_with_threading(
        sentences=test_sentences,
        model=model,
        tokenizer=tokenizer,
        batch_size=16,
        max_workers=2
    )
    thread_time = time.time() - start_time
    
    print(f"Batch processing: {batch_time:.3f} seconds")
    print(f"Threading approach: {thread_time:.3f} seconds")
    print(f"Speedup: {batch_time/thread_time:.2f}x" if thread_time < batch_time else f"Slower by {thread_time/batch_time:.2f}x")
    
    # Example 5: Memory-efficient processing for very large datasets
    print("\n=== Example 5: Memory-Efficient Processing ===")
    def encode_large_dataset_efficiently(sentences, chunk_size=1000):
        """Process very large datasets in chunks to manage memory."""
        all_embeddings = {}
        
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(sentences)-1)//chunk_size + 1}")
            
            chunk_embeddings = encode_sentences_parallel(
                sentences=chunk,
                model=model,
                tokenizer=tokenizer,
                batch_size=32
            )
            all_embeddings.update(chunk_embeddings)
        
        return all_embeddings
    
    # Simulate a large dataset
    large_dataset = [f"Large dataset sentence {i}" for i in range(500)]
    
    start_time = time.time()
    large_results = encode_large_dataset_efficiently(large_dataset, chunk_size=100)
    large_time = time.time() - start_time
    
    print(f"Processed {len(large_dataset)} sentences in chunks: {large_time:.3f} seconds")
    print(f"Memory-efficient average: {large_time/len(large_dataset)*1000:.2f} ms per sentence")


if __name__ == "__main__":
    main() 