#!/usr/bin/env python3
"""
Compute the stride parameter from a JSON file of images.
Each image has a 'date_captured' field in ISO format, e.g. "2020-05-14T22:40:46".

The stride is defined as
    stride = round(total_images / (24 * number_of_days))
where number_of_days is the count of unique calendar dates in the dataset.
"""

import argparse
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute stride = round(total_images / (24 * unique_days)) from a JSON 'images' list."
    )
    parser.add_argument(
        "json_file",
        help="Path to the JSON file containing an 'images' array with 'date_captured' tags."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load JSON
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    images = data.get("images", [])
    if not images:
        print("No 'images' array found or it is empty.")
        return

    # Extract unique dates
    unique_dates = set()
    for img in images:
        date_str = img.get("date_captured")
        if not date_str:
            continue
        try:
            # ISO format parser (Python 3.7+)
            dt = datetime.fromisoformat(date_str)
        except ValueError:
            # Fallback if any timezone or format differences occur
            dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        unique_dates.add(dt.date())

    num_days = len(unique_dates)
    total_images = len(images)

    if num_days == 0:
        print("Could not infer any valid dates from 'date_captured' fields.")
        return

    # Compute stride
    mean_hourly = total_images / (24 * num_days)
    stride = round(mean_hourly)

    # Output results
    print(f"Total images:        {total_images}")
    print(f"Unique days found:   {num_days}")
    print(f"Mean hourly count:   {mean_hourly:.2f}")
    print(f"Computed stride:     {stride}")

if __name__ == "__main__":
    main()
