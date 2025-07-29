# Requires transformers>=4.51.0

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def encode_sentences_parallel(
    sentences: List[str],
    model: Optional[AutoModel] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    task_description: str = 'Given a web search query, retrieve relevant passages that answer the query',
    batch_size: int = 32,
    max_length: int = 8192,
    add_instruction: bool = False,
    normalize: bool = True,
    device: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Encode sentences in parallel using batch processing for maximum efficiency.
    
    Args:
        sentences: List of sentences to encode
        model: Pre-loaded model (if None, will load Qwen3-Embedding-0.6B)
        tokenizer: Pre-loaded tokenizer (if None, will load Qwen3-Embedding-0.6B tokenizer)
        task_description: Task description for instruction-based encoding
        batch_size: Number of sentences to process in each batch
        max_length: Maximum token length for each sentence
        add_instruction: Whether to add task instruction to queries
        normalize: Whether to normalize embeddings
        device: Device to use for processing (auto-detected if None)
        
    Returns:
        Dictionary mapping original sentences to their embeddings as numpy arrays
    """
    
    # Load model and tokenizer if not provided
    if model is None:
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
    
    # Set device
    if device is None:
        device = model.device
    model.to(device)
    
    # Prepare sentences (add instruction if requested)
    processed_sentences = []
    if add_instruction:
        processed_sentences = [get_detailed_instruct(task_description, sentence) for sentence in sentences]
    else:
        processed_sentences = sentences.copy()
    
    # Process sentences in batches for optimal parallel processing
    all_embeddings = []
    
    for i in range(0, len(processed_sentences), batch_size):
        batch_sentences = processed_sentences[i:i + batch_size]
        
        # Tokenize batch
        batch_dict = tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalize if requested
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy and add to results
            batch_embeddings = embeddings.cpu().numpy()
            all_embeddings.extend(batch_embeddings)
    
    # Create dictionary mapping original sentences to embeddings
    sentence_embeddings = {
        sentence: embedding 
        for sentence, embedding in zip(sentences, all_embeddings)
    }
    
    return sentence_embeddings


def encode_sentences_with_threading(
    sentences: List[str],
    model: Optional[AutoModel] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    task_description: str = 'Given a web search query, retrieve relevant passages that answer the query',
    batch_size: int = 32,
    max_workers: int = 2,
    max_length: int = 8192,
    add_instruction: bool = False,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Encode sentences using both batch processing and threading for maximum parallelism.
    Note: This approach may not always be faster due to GIL limitations in Python.
    
    Args:
        sentences: List of sentences to encode
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer  
        task_description: Task description for instruction-based encoding
        batch_size: Number of sentences per batch
        max_workers: Number of thread workers
        max_length: Maximum token length
        add_instruction: Whether to add task instruction
        normalize: Whether to normalize embeddings
        
    Returns:
        Dictionary mapping sentences to embeddings
    """
    
    # Load model and tokenizer if not provided
    if model is None:
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
    
    # Split sentences into chunks for different threads
    chunk_size = len(sentences) // max_workers if len(sentences) >= max_workers else len(sentences)
    sentence_chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
    
    def process_chunk(chunk):
        return encode_sentences_parallel(
            chunk, model, tokenizer, task_description, 
            batch_size, max_length, add_instruction, normalize
        )
    
    # Process chunks in parallel using threads
    all_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in sentence_chunks}
        
        for future in future_to_chunk:
            chunk_results = future.result()
            all_results.update(chunk_results)
    
    return all_results


# Example usage and existing code
if __name__ == "__main__":
    # Load model and tokenizer once for reuse
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
    
    # Example sentences to encode
    example_sentences = [
        'What is the capital of China?',
        'Explain gravity',
        'How does photosynthesis work?',
        'What is machine learning?',
        'Describe the water cycle'
    ]
    
    # Encode sentences in parallel
    embeddings_dict = encode_sentences_parallel(
        sentences=example_sentences,
        model=model,
        tokenizer=tokenizer,
        batch_size=16,
        add_instruction=True,
        normalize=True
    )
    
    # Print results
    print("Encoded sentences:")
    for sentence, embedding in embeddings_dict.items():
        print(f"'{sentence}': shape {embedding.shape}")
    
    # Original example code
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    
    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    input_texts = queries + documents
    
    max_length = 8192
    
    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:2] @ embeddings[2:].T)
    
    print("\nOriginal example scores:")
    print(scores.tolist())
    # [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]