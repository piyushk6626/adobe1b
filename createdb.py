import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def create_database_table(db_path: str) -> None:
    """
    Create SQLite database with the required schema.
    
    Args:
        db_path: Path to the database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table with required columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            main_persona TEXT NOT NULL,
            sub_persona TEXT NOT NULL,
            persona_description TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()


def save_persona_to_db(db_path: str, main_persona: str, sub_persona: str, 
                      persona_description: str, embedding: List[float]) -> None:
    """
    Save persona data to the specified database.
    
    Args:
        db_path: Path to the database file
        main_persona: Main persona type
        sub_persona: Sub persona type  
        persona_description: Detailed persona description
        embedding: 1024-dimensional vector as list of floats
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Convert embedding list to numpy array and then to bytes for storage
    embedding_array = np.array(embedding, dtype=np.float32)
    embedding_bytes = embedding_array.tobytes()
    
    cursor.execute('''
        INSERT INTO personas (main_persona, sub_persona, persona_description, embedding)
        VALUES (?, ?, ?, ?)
    ''', (main_persona, sub_persona, persona_description, embedding_bytes))
    
    conn.commit()
    conn.close()


def process_persona_files() -> None:
    """
    Process all JSON files in the persona directory and create appropriate databases.
    """
    persona_dir = Path("persona")
    
    if not persona_dir.exists():
        raise FileNotFoundError("Persona directory not found")
    
    # Track which databases we've created to avoid recreating tables
    created_dbs = set()
    
    # Process each JSON file
    for json_file in persona_dir.glob("*.json"):
        try:
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            main_persona = data.get("main_persona", "")
            sub_persona = data.get("sub_persona", "")
            persona_description = data.get("details", "")
            embedding = data.get("embedding", [])
            
            # Validate required fields
            if not all([main_persona, sub_persona, persona_description, embedding]):
                print(f"Warning: Missing required fields in {json_file.name}")
                continue
                
            # Validate embedding dimension
            if len(embedding) != 1024:
                print(f"Warning: Invalid embedding dimension in {json_file.name}. Expected 1024, got {len(embedding)}")
                continue
            
            # Determine which database to use
            if main_persona == sub_persona:
                db_path = "personas.db"
            else:
                db_path = f"{main_persona}.db"
            
            # Create database table if not already created
            if db_path not in created_dbs:
                create_database_table(db_path)
                created_dbs.add(db_path)
                print(f"Created database: {db_path}")
            
            # Save persona data to database
            save_persona_to_db(db_path, main_persona, sub_persona, 
                              persona_description, embedding)
            
            print(f"Processed: {json_file.name} -> {db_path}")
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")
            continue


def query_database(db_path: str, main_persona: str = None) -> List[Dict[str, Any]]:
    """
    Query database to retrieve persona data.
    
    Args:
        db_path: Path to the database file
        main_persona: Optional filter by main_persona
        
    Returns:
        List of persona records
    """
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist")
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if main_persona:
        cursor.execute('''
            SELECT id, main_persona, sub_persona, persona_description, embedding
            FROM personas WHERE main_persona = ?
        ''', (main_persona,))
    else:
        cursor.execute('''
            SELECT id, main_persona, sub_persona, persona_description, embedding
            FROM personas
        ''')
    
    results = []
    for row in cursor.fetchall():
        # Convert bytes back to numpy array
        embedding_array = np.frombuffer(row[4], dtype=np.float32)
        
        results.append({
            'id': row[0],
            'main_persona': row[1],
            'sub_persona': row[2],
            'persona_description': row[3],
            'embedding': embedding_array.tolist()
        })
    
    conn.close()
    return results


def get_database_stats() -> None:
    """
    Print statistics about the created databases.
    """
    # Check personas.db (where main_persona == sub_persona)
    if os.path.exists("personas.db"):
        conn = sqlite3.connect("personas.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM personas")
        count = cursor.fetchone()[0]
        print(f"personas.db: {count} records")
        conn.close()
    
    # Check individual persona databases
    for db_file in Path(".").glob("*.db"):
        if db_file.name != "personas.db":
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM personas")
            count = cursor.fetchone()[0]
            print(f"{db_file.name}: {count} records")
            conn.close()


if __name__ == "__main__":
    print("Processing persona JSON files...")
    process_persona_files()
    
    print("\nDatabase Statistics:")
    get_database_stats()
    
    print("\nDone! Databases created successfully.") 