#!/usr/bin/env python3
"""Convert a plain edge-list .txt file to a .pkl file.

Expected input format (one edge per line):
    source_node target_node

Example:
    0 1
    0 2
"""

import argparse
import pickle


def load_edges_from_txt(input_path: str):
    """Read edges from a whitespace-separated text file.

    Returns:
        list[tuple[int, int]]: List of (source, target) integer pairs.
    """
    edges = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # Skip blank lines to be tolerant of minor formatting noise.
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid format at line {line_number}: '{line}'. "
                    "Each line must contain exactly two node IDs."
                )

            src, dst = int(parts[0]), int(parts[1])
            edges.append((src, dst))

    return edges


def extract_nodes(edges):
    """Build a sorted unique node list from edges."""
    nodes = set()
    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)
    return sorted(nodes)


def main():
    parser = argparse.ArgumentParser(
        description="Convert an edge-list .txt file to a .pkl file"
    )
    parser.add_argument("--input", default="facebook_combined.txt", help="Path to input .txt file")
    parser.add_argument("--output", default="facebook_combined.pkl", help="Path to output .pkl file")
    args = parser.parse_args()

    edges = load_edges_from_txt(args.input)

    nodes = extract_nodes(edges)

    # IMPORTANT: decoder.py expects either:
    # 1) a NetworkX Graph/DiGraph, or
    # 2) a dict with exactly usable 'nodes' and 'edges' keys.
    # We write option (2) for maximum simplicity and portability.
    payload = {
        "nodes": nodes,
        "edges": edges,
    }

    with open(args.output, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved {len(nodes)} nodes and {len(edges)} edges to {args.output}")


if __name__ == "__main__":
    main()
