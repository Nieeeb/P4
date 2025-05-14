import json
import argparse
import random
from collections import defaultdict
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Balance image samples per hour within each day by downsampling to the minimum non-zero hourly count."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input JSON file containing an 'images' list with 'date_captured' timestamps."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to write the output JSON file with balanced 'images'."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility."
    )
    return parser.parse_args()


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def group_by_day(images):
    days = defaultdict(list)
    for img in images:
        # Expect format: YYYY-MM-DDTHH:MM:SS
        ts = img.get('date_captured') or img.get('dat_captured')
        if not ts:
            continue
        day = ts.split('T')[0]
        days[day].append(img)
    return days


def group_by_hour(images):
    hours = defaultdict(list)
    for img in images:
        ts = img.get('date_captured') or img.get('dat_captured')
        if not ts:
            continue
        # Extract hour from timestamp
        try:
            hour = datetime.fromisoformat(ts).hour
        except Exception:
            # Fallback: split
            hour = int(ts.split('T')[1].split(':')[0])
        hours[hour].append(img)
    return hours


def balance_day(images, rng):
    # Group into hours
    hours = group_by_hour(images)
    # Filter out hours with zero samples
    nonzero_counts = [len(imgs) for imgs in hours.values() if len(imgs) > 0]
    if not nonzero_counts:
        return []
    # Minimum non-zero hourly count
    min_count = min(nonzero_counts)
    balanced = []
    for hour, imgs in hours.items():
        n = len(imgs)
        if n > min_count:
            # Downsample
            balanced.extend(rng.sample(imgs, min_count))
        else:
            # Keep all (including those equal to min_count or less)
            balanced.extend(imgs)
    return balanced


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    rng = random

    data = load_json(args.input)
    images = data.get('images', [])

    # Group images by day
    days = group_by_day(images)

    # Balance each day
    balanced_images = []
    for day, imgs in days.items():
        balanced = balance_day(imgs, rng)
        balanced_images.extend(balanced)

    # Replace images list
    data['images'] = balanced_images

    # Save
    save_json(data, args.output)
    print(f"Balanced dataset written to {args.output}. Original count: {len(images)}, new count: {len(balanced_images)}")

if __name__ == '__main__':
    main()
