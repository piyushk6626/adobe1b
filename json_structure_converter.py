#!/usr/bin/env python3
"""
JSON Structure Converter

Converts DoclingDocument JSON format to nested hierarchical structure.
Transforms flat text arrays into parent-child relationships with proper heading levels.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")


def extract_page_number(prov_data: List[Dict[str, Any]]) -> int:
    """Extract page number from provenance data."""
    if not prov_data or not isinstance(prov_data, list):
        return 1
    return prov_data[0].get("page_no", 1)


def create_nested_item(text: str, category: str, page: int, sub_categories: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a nested structure item."""
    item = {
        "category": category,
        "text": text.strip(),
        "page": page
    }
    
    if sub_categories is not None and category in ["h1", "h2"]:  # Only h1 and h2 can have sub_categories
        item["sub_categories"] = sub_categories
    elif category in ["h1", "h2"]:  # Only add sub_categories to headers that can have them
        item["sub_categories"] = []
        
    return item


def group_texts_by_context(texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group texts into logical hierarchical structure.
    Since original only has level 1 headers, we'll create artificial hierarchy
    based on content proximity and context.
    h3 tags will not have nested children - any h3 content becomes additional h2 items.
    """
    grouped_items = []
    current_h1 = None
    current_h2 = None
    h1_content = []
    h2_content = []
    
    for text_item in texts:
        label = text_item.get("label", "")
        text_content = text_item.get("text", "").strip()
        page = extract_page_number(text_item.get("prov", []))
        level = text_item.get("level")
        
        if not text_content:
            continue
            
        # Section headers become hierarchical based on content and position
        if label == "section_header" and level == 1:
            # Save previous h1 if exists
            if current_h1:
                # Add any remaining h2 content as separate h2 items (flattened)
                if current_h2 and h2_content:
                    current_h1["sub_categories"].append(create_nested_item(current_h2["text"], "h2", current_h2["page"]))
                    # Add h2_content as additional h2 items instead of nesting under h3
                    for content in h2_content:
                        current_h1["sub_categories"].append(create_nested_item(content["text"], "h2", content["page"]))
                elif current_h2:
                    current_h1["sub_categories"].append(create_nested_item(current_h2["text"], "h2", current_h2["page"]))
                
                # Add any remaining h1 content as h2
                for content in h1_content:
                    current_h1["sub_categories"].append(create_nested_item(content["text"], "h2", content["page"]))
                
                grouped_items.append(current_h1)
            
            # Start new h1
            current_h1 = create_nested_item(text_content, "h1", page, [])
            if "sub_categories" not in current_h1:
                current_h1["sub_categories"] = []
            current_h2 = None
            h1_content = []
            h2_content = []
            
        elif label == "text":
            # Regular text content - decide where to place it
            text_item_info = {"text": text_content, "page": page}
            
            if len(text_content) > 200:  # Long content becomes h2
                # Save previous h2 if exists
                if current_h2 and h2_content:
                    if current_h1:
                        current_h1["sub_categories"].append(create_nested_item(current_h2["text"], "h2", current_h2["page"]))
                        # Add h2_content as additional h2 items instead of nesting under h3
                        for content in h2_content:
                            current_h1["sub_categories"].append(create_nested_item(content["text"], "h2", content["page"]))
                elif current_h2:
                    if current_h1:
                        current_h1["sub_categories"].append(create_nested_item(current_h2["text"], "h2", current_h2["page"]))
                
                # This long content becomes new h2
                current_h2 = text_item_info
                h2_content = []
                
            elif current_h2:
                # Add to current h2's sub-content
                h2_content.append(text_item_info)
            else:
                # Add to h1's direct content
                h1_content.append(text_item_info)
    
    # Handle the last items
    if current_h1:
        # Add any remaining h2 content as separate h2 items (flattened)
        if current_h2 and h2_content:
            current_h1["sub_categories"].append(create_nested_item(current_h2["text"], "h2", current_h2["page"]))
            # Add h2_content as additional h2 items instead of nesting under h3
            for content in h2_content:
                current_h1["sub_categories"].append(create_nested_item(content["text"], "h2", content["page"]))
        elif current_h2:
            current_h1["sub_categories"].append(create_nested_item(current_h2["text"], "h2", current_h2["page"]))
        
        # Add any remaining h1 content as h2
        for content in h1_content:
            current_h1["sub_categories"].append(create_nested_item(content["text"], "h2", content["page"]))
        
        grouped_items.append(current_h1)
    
    return grouped_items


def create_root_structure(grouped_items: List[Dict[str, Any]], document_info: Dict[str, Any]) -> Dict[str, Any]:
    """Create the root structure combining all h1 items."""
    if not grouped_items:
        return {
            "category": "h1",
            "text": "Empty Document",
            "page": 1,
            "sub_categories": []
        }
    
    # If we have multiple h1 items, we need to combine them appropriately
    if len(grouped_items) == 1:
        return grouped_items[0]
    
    # Combine multiple h1s under a root structure
    # Take the first h1's text as the main title and make others sub-categories
    root_item = grouped_items[0].copy()
    
    # Add remaining h1s as h2 sub-categories
    for item in grouped_items[1:]:
        # Convert h1 to h2
        h2_item = {
            "category": "h2",
            "text": item["text"],
            "page": item["page"]
        }
        
        # Convert sub-categories from h2->h3, h3->h4, etc.
        if "sub_categories" in item and item["sub_categories"]:
            h2_item["sub_categories"] = []
            for sub_item in item["sub_categories"]:
                new_sub = sub_item.copy()
                if new_sub["category"] == "h2":
                    new_sub["category"] = "h3"
                elif new_sub["category"] == "h3":
                    new_sub["category"] = "h4"
                h2_item["sub_categories"].append(new_sub)
        
        root_item["sub_categories"].append(h2_item)
    
    return root_item


def convert_docling_to_nested(input_file: str, output_file: str = None) -> Dict[str, Any]:
    """
    Convert DoclingDocument JSON to nested hierarchical structure.
    
    Args:
        input_file: Path to input DoclingDocument JSON file
        output_file: Optional path to save the converted structure
        
    Returns:
        Dict containing the nested structure
    """
    # Load the original document
    doc_data = load_json_file(input_file)
    
    # Extract texts array
    texts = doc_data.get("texts", [])
    if not texts:
        raise ValueError("No texts found in the document")
    
    # Group texts into hierarchical structure
    grouped_items = group_texts_by_context(texts)
    
    # Create the root nested structure
    nested_structure = create_root_structure(grouped_items, doc_data)
    
    # Save to output file if specified
    if output_file:
        save_nested_structure(nested_structure, output_file)
    
    return nested_structure


def save_nested_structure(nested_structure: Dict[str, Any], output_file: str) -> None:
    """Save the nested structure to a JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(nested_structure, file, indent=4, ensure_ascii=False)
    
    print(f"Nested structure saved to: {output_file}")


def main():
    """Main function to demonstrate the conversion."""
    input_file = "scratch/file03.json"
    output_file = "glam_json/file03_auto_nested.json"
    
    try:
        print(f"Converting {input_file} to nested structure...")
        nested_result = convert_docling_to_nested(input_file, output_file)
        
        print(f"Conversion completed successfully!")
        print(f"Root category: {nested_result.get('category')}")
        print(f"Root page: {nested_result.get('page')}")
        print(f"Number of sub-categories: {len(nested_result.get('sub_categories', []))}")
        
        return nested_result
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main() 