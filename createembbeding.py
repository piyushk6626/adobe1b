import json
import os
from typing import List, Dict
from embbeding.localquwen import encode_sentences_parallel
from transformers import AutoTokenizer, AutoModel


def open_json_file(path: str) -> Dict:
    """Open and parse a JSON file"""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {path} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {path}: {e}")
        return {}


def update_json_file(path: str, data: Dict):
    """Update a JSON file with new data"""
    try:
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"Updated: {path}")
    except Exception as e:
        print(f"Error writing to {path}: {e}")


def create_embbeding(list_of_text: List[str]) -> Dict:
    """Create embeddings for a list of text using the local Qwen model"""
    try:
        # Load model and tokenizer once for efficiency
        print("Loading embedding model...")
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
        
        # Create embeddings for the text list
        embeddings_dict = encode_sentences_parallel(
            sentences=list_of_text,
            model=model,
            tokenizer=tokenizer,
            batch_size=16,
            add_instruction=True,
            normalize=True,
            task_description="Create embeddings for persona descriptions to enable semantic search and similarity matching"
        )
        
        return embeddings_dict
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return {}


def get_persona_files(persona_dir: str = "persona") -> List[str]:
    """Get all JSON files from the persona directory"""
    try:
        if not os.path.exists(persona_dir):
            print(f"Error: Directory {persona_dir} not found")
            return []
        
        json_files = [
            os.path.join(persona_dir, filename) 
            for filename in os.listdir(persona_dir) 
            if filename.endswith('.json')
        ]
        
        print(f"Found {len(json_files)} persona files")
        return json_files
    except Exception as e:
        print(f"Error reading persona directory: {e}")
        return []


def main():
    """Main function to process all persona files and add embeddings"""
    print("Starting persona embedding process...")
    
    # Get all persona files
    persona_files = get_persona_files()
    
    if not persona_files:
        print("No persona files found. Exiting.")
        return
    
    # Process files in batches to manage memory
    batch_size = 50  # Process 50 files at a time
    total_files = len(persona_files)
    
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = persona_files[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start // batch_size + 1} ({batch_start + 1}-{batch_end} of {total_files})")
        
        # Collect all text from this batch
        batch_texts = []
        batch_data = []
        file_paths = []
        
        for file_path in batch_files:
            persona_data = open_json_file(file_path)
            if persona_data and 'sub_persona' in persona_data:
                # Skip if embedding already exists
                if 'embedding' in persona_data:
                    print(f"Skipping {file_path} - embedding already exists")
                    continue
                
                batch_texts.append(persona_data['sub_persona'])
                batch_data.append(persona_data)
                file_paths.append(file_path)
        
        if not batch_texts:
            print("No new files to process in this batch")
            continue
        
        # Create embeddings for this batch
        print(f"Creating embeddings for {len(batch_texts)} files...")
        embeddings_dict = create_embbeding(batch_texts)
        
        if not embeddings_dict:
            print("Failed to create embeddings for this batch")
            continue
        
        # Update files with embeddings
        for i, (file_path, persona_data, text) in enumerate(zip(file_paths, batch_data, batch_texts)):
            if text in embeddings_dict:
                # Add embedding to persona data
                persona_data['embedding'] = embeddings_dict[text].tolist()  # Convert numpy array to list for JSON serialization
                
                # Update the file
                update_json_file(file_path, persona_data)
            else:
                print(f"Warning: No embedding found for {file_path}")
    
    print(f"\nCompleted processing all {total_files} persona files!")


if __name__ == "__main__":
    main()