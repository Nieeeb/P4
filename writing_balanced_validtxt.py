#!/usr/bin/env python3
"""
Script to read a JSON file containing an 'images' array, extract each
'image' → 'file_name', transform the paths by replacing the leading
'frames/' with 'Data/images/train/' and converting remaining slashes
to underscores, then write the resulting list to a text file.
"""

import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert JSON image paths and write to a text file."
    )
    parser.add_argument(
        "json_file",
        help="Path to the JSON file containing an 'images' array."
    )
    parser.add_argument(
        "output_txt",
        help="Path to the output .txt file where modified paths will be written."
    )
    return parser.parse_args()

def transform_path(orig_path: str) -> str:
    """
    Given an original path like
        'frames/20200514/clip_21_2239/image_0106.jpg'
    return
        'Data/images/train/20200514_clip_21_2239_image_0106.jpg'
    """
    prefix = "frames/"
    if not orig_path.startswith(prefix):
        raise ValueError(f"Unexpected path format: {orig_path!r}")
    # strip 'frames/' and replace all remaining '/' with '_'
    core = orig_path[len(prefix):].replace('/', '_')
    return f"Data/images/train/{core}"

def main():
    args = parse_args()

    # Load JSON
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    images = data.get("images", [])
    if not images:
        print("No 'images' array found or it is empty.")
        return

    # Transform each file_name
    transformed = []
    for img in images:
        fname = img.get("file_name")
        if not fname:
            continue
        try:
            new_path = transform_path(fname)
        except ValueError as e:
            print(f"Skipping invalid entry: {e}")
            continue
        transformed.append(new_path)

    # Write to output text file
    out_path = Path(args.output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for line in transformed:
            f.write(line + "\n")

    print(f"Wrote {len(transformed)} paths to {out_path}")

if __name__ == "__main__":
    main()
