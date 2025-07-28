import logging
import math
import re
import time
import json
from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple, Sequence, MutableSequence, Optional, Generator, Iterable, Any, Mapping

import networkx as nx
import numpy as np
import torch
from shapely import Polygon
from torch_geometric.data import Data
import fitz  # PyMuPDF
from PIL import Image

# Assuming these modules are in the same directory or accessible
import models
from GLAM.common import PageEdges, ImageNode, TextNode, get_bytes_per_pixel, PageNodes
from GLAM.models import GLAMGraphNetwork
from dln_glam_prepare import CLASSES_MAP

# --- Constants and Setup ---
INVALID_UNICODE = chr(0xFFFD)
EasyocrTextResult = namedtuple("EasyocrTextResult", ["bbox", "text", "confidence"])
MuPDFTextTraceChar = namedtuple("MuPDFTextTraceChar", ["unicode", "glyph", "origin", "bbox"])
logger = logging.getLogger(__name__)


def analyze_document(pdf_filepath: str, model_filepath = "models/glam_dln.pt") -> str:
    """
    Analyzes a PDF document using the GLAM model to identify and classify layout elements.

    The function processes each page to find text and image nodes, runs them through the
    graph network model to cluster them into logical elements (like titles, paragraphs, lists),
    and saves the results to a JSON file.

    Args:
        pdf_filepath (str): The path to the input PDF file.
        model_filepath (str): The path to the pre-trained `.pt` model file.

    Returns:
        str: The path to the newly created JSON file containing the analysis results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load the pre-trained model
    model = GLAMGraphNetwork(PageNodes.features_len, PageEdges.features_len, 512, len(CLASSES_MAP))
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    model = model.to(device)
    model.eval()

    doc = fitz.Document(pdf_filepath)
    document_results = []

    for page_num, page in enumerate(doc):
        # --- 1. Node Extraction ---
        page_nodes = PageNodes()
        page_dict = page.get_text(
            "dict",
            flags=fitz.TEXT_PRESERVE_IMAGES
        )
        for block in page_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        page_nodes.append(TextNode.from_span(span, text=span["text"]))
            elif block["type"] == 1:  # Image block
                page_nodes.append(ImageNode.from_page_block(block))

        if not page_nodes:
            continue

        # --- 2. Graph Construction ---
        page_edges = PageEdges.from_page_nodes_as_complete_graph(page_nodes)
        node_features = page_nodes.to_node_features()
        edge_index = page_edges.to_edge_index().t()
        edge_features = page_edges.to_edge_features()

        if edge_index.shape[0] == 0:
            continue

        # --- 3. Model Inference ---
        example = Data(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        ).to(device)

        with torch.no_grad():
            node_class_scores, edge_class_scores = model(example)

        # --- 4. Clustering based on Edge Prediction ---
        edge_prob_threshold = 0.5
        graph = nx.Graph()
        for k in range(example.edge_index.shape[1]):
            src_node_i = example.edge_index[0, k].item()
            dst_node_i = example.edge_index[1, k].item()
            edge_prob = edge_class_scores[k].item()

            if edge_prob >= edge_prob_threshold:
                graph.add_edge(src_node_i, dst_node_i, weight=edge_prob)
            else:
                # Add nodes even if they don't have edges above the threshold
                graph.add_node(src_node_i)
                graph.add_node(dst_node_i)
        
        clusters = list(nx.connected_components(graph))
        if not clusters:
            continue

        # --- 5. Classify Clusters and Assemble Results ---
        cluster_classes = torch.stack(
            [node_class_scores[torch.tensor(list(c))].sum(dim=0) for c in clusters]
        ).argmax(dim=1).tolist()

        for i, cluster in enumerate(clusters):
            predicted_class_index = cluster_classes[i]
            predicted_class_name = CLASSES_MAP.get(predicted_class_index, "Unknown")
            
            # Combine text from all text nodes in the cluster
            node_contents = [
                page_nodes[node_i].text if hasattr(page_nodes[node_i], 'text') else ""
                for node_i in sorted(list(cluster))
            ]
            full_text = " ".join(node_contents).strip()

            # Calculate the bounding box for the entire cluster
            min_x = min(page_nodes[node_i].bbox_min_x for node_i in cluster)
            min_y = min(page_nodes[node_i].bbox_min_y for node_i in cluster)
            max_x = max(page_nodes[node_i].bbox_max_x for node_i in cluster)
            max_y = max(page_nodes[node_i].bbox_max_y for node_i in cluster)
            
            # Store result for this cluster
            document_results.append({
                "page": page_num + 1,
                "category": predicted_class_name,
                "bbox": [round(c, 2) for c in [min_x, min_y, max_x, max_y]],
                "text": full_text if predicted_class_name != "Picture" else "[Image Content]"
            })

    # --- 6. Save Results to JSON ---
    output_json_path = pdf_filepath.rsplit('.', 1)[0] + ".json"
    output_json_path = "glam_json/" + output_json_path.split("/")[-1]
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(document_results, f, ensure_ascii=False, indent=4)

    logger.info(f"✅ Document Layout Analysis Results saved to {output_json_path}")
    return output_json_path


def main():
    """
    An example of how to call the analyze_document function.
    """
    pdf_filepath = "file03.pdf"
    model_filepath = "models/glam_dln.pt"

    # Configure logger for console output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Call the primary analysis function
        json_output_path = analyze_document(
            pdf_filepath=pdf_filepath,
            model_filepath=model_filepath,
        )
        print(f"\n✨ Analysis complete. Results have been saved to: {json_output_path}")

    except FileNotFoundError as e:
        logger.error(f"Error: Input file not found. {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == '__main__':
    main()