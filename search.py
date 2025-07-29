# write a function to search the database for the most similar persona to the query

from typing import List, Tuple, Dict, Any
import sqlite3
import numpy as np
from embbeding.localquwen import encode_sentences_parallel
import os


def similarity_search(vector1: List[float], vector2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def get_embbeding(query: str) -> List[float]:
    """Generate embedding for a given query using the local Qwen model."""
    embeddings_dict = encode_sentences_parallel(
        sentences=[query],
        task_description='Given a persona search query, retrieve relevant persona descriptions',
        batch_size=1,
        add_instruction=True,
        normalize=True
    )
    return embeddings_dict[query].tolist()


def search_persona_db(vector: List[float], top_k: int = 3) -> List[Tuple[str, str, str, float]]:
    """
    Search the main personas database for the most similar personas.
    
    Args:
        vector: Query embedding vector
        top_k: Number of top results to return
        
    Returns:
        List of tuples (main_persona, sub_persona, description, similarity_score)
    """
    db_path = "db/personas.db"
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist")
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT main_persona, sub_persona, persona_description, embedding FROM personas")
    results = cursor.fetchall()
    conn.close()
    
    similarities = []
    for row in results:
        main_persona, sub_persona, description, embedding_bytes = row
        # Convert bytes back to numpy array then to list
        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
        embedding_list = embedding_array.tolist()
        
        # Calculate similarity
        similarity = similarity_search(vector, embedding_list)
        similarities.append((main_persona, sub_persona, description, similarity))
    
    # Sort by similarity score (descending) and return top_k
    similarities.sort(key=lambda x: x[3], reverse=True)
    return similarities[:top_k]


def search_spcific_persona_db(vector: List[float], main_persona: str, top_k: int = 3) -> List[Tuple[str, str, str, float]]:
    """
    Search a specific persona database for the most similar personas.
    
    Args:
        vector: Query embedding vector
        main_persona: Main persona type to search
        top_k: Number of top results to return
        
    Returns:
        List of tuples (main_persona, sub_persona, description, similarity_score)
    """
    db_path = f"db/{main_persona}.db"
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist")
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT main_persona, sub_persona, persona_description, embedding FROM personas")
    results = cursor.fetchall()
    conn.close()
    
    similarities = []
    for row in results:
        main_persona_db, sub_persona, description, embedding_bytes = row
        # Convert bytes back to numpy array then to list
        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
        embedding_list = embedding_array.tolist()
        
        # Calculate similarity
        similarity = similarity_search(vector, embedding_list)
        similarities.append((main_persona_db, sub_persona, description, similarity))
    
    # Sort by similarity score (descending) and return top_k
    similarities.sort(key=lambda x: x[3], reverse=True)
    return similarities[:top_k]


def combine_results(results_list: List[List[Tuple[str, str, str, float]]], top_k: int = 5) -> List[Tuple[str, str, str, float]]:
    """
    Combine results from multiple databases and return the top results.
    
    Args:
        results_list: List of result lists from different databases
        top_k: Number of top results to return
        
    Returns:
        Combined and sorted list of top results
    """
    all_results = []
    for results in results_list:
        all_results.extend(results)
    
    # Sort by similarity score (descending) and return top_k
    all_results.sort(key=lambda x: x[3], reverse=True)
    return all_results[:top_k]


def search_database(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Main search function that searches both general and specific persona databases.
    
    Args:
        query: Search query string
        top_k: Number of top results to return
        
    Returns:
        List of dictionaries containing persona information and similarity scores
    """
    # Get the embedding of the query
    query_vector = get_embbeding(query)
    
    # Search the main personas database
    main_results = search_persona_db(query_vector, top_k=10)  # Get more results initially
    
    # Collect unique main personas from the results for specific searches
    main_personas = list(set([result[0] for result in main_results]))
    
    # Search specific persona databases
    all_results = [main_results]  # Start with main results
    
    for main_persona in main_personas:
        specific_results = search_spcific_persona_db(query_vector, main_persona, top_k=5)
        if specific_results:
            all_results.append(specific_results)
    
    # Combine all results
    final_results = combine_results(all_results, top_k=top_k)
    
    # Format results as dictionaries
    formatted_results = []
    for main_persona, sub_persona, description, similarity in final_results:
        formatted_results.append({
            'main_persona': main_persona,
            'sub_persona': sub_persona,
            'description': description,
            'similarity_score': similarity
        })
    
    return formatted_results


# Example usage and testing function
def test_search(query: str = "trip planner"):
    """Test the search functionality with a sample query."""
    print(f"Searching for: '{query}'")
    print("-" * 50)
    
    results = search_database(query, top_k=5)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Main Persona: {result['main_persona']}")
        print(f"   Sub Persona: {result['sub_persona']}")
        print(f"   Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Description: {result['description'][:200]}...")
        print()
    
    return results


if __name__ == "__main__":
    # Test the search functionality
    test_search()



