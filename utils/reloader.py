import json
import os
import shutil
from collections import defaultdict
from tqdm import tqdm

# ----- CONFIGURATION -----
# Path to your JSON file (e.g., for validation split)
json_path = "/ceph/project/DAKI4-thermal-2025/harborfrontv2/Test.json"
# Base directory where images are stored (relative to your current directory)
base_img_dir = "/ceph/project/DAKI4-thermal-2025/harborfrontv2/frames"
# Specify the dataset split ("train", "valid", or "test")
dataset_split = "test"  # Change as needed

# Define the output directory where both images and labels will be saved
images_dir = os.path.join('Data', 'images', dataset_split)
os.makedirs(images_dir, exist_ok=True)


labels_dir = os.path.join('Data', 'labels', dataset_split)

# Create output directories if they don't exist
os.makedirs(labels_dir, exist_ok=True)

# ----- LOAD THE JSON DATA -----
with open(json_path, 'r') as f:
    data = json.load(f)

# ----- BUILD IMAGE INFO DICTIONARY -----
# Map each image id to its metadata.
images_info = {}
for img in data['images']:
    image_id = img['id']
    orig_file = img['file_name']  # e.g. "frames/20210220/clip_36_2025/image_0088.jpg"
    # Remove the folder structure prefix "frames/" since base_img_dir already points there.
    if orig_file.startswith("frames/"):
        relative_path = orig_file[len("frames/"):]
    else:
        relative_path = orig_file

    # Construct new filename by joining the remaining parts with underscores.
    parts = relative_path.split('/')
    new_file = "_".join(parts)
    
    images_info[image_id] = {
        'orig_file': orig_file,
        'relative_path': relative_path,
        'new_file': new_file,
        'width': img['width'],
        'height': img['height']
    }

# ----- GROUP ANNOTATIONS BY IMAGE -----
annotations_by_image = defaultdict(list)
for ann in tqdm(data['annotations']):
    annotations_by_image[ann['image_id']].append(ann)

# ----- PROCESS EACH IMAGE -----
for image_id, info in tqdm(images_info.items()):
    width = info['width']
    height = info['height']
    orig_file = info['orig_file']
    new_file = info['new_file']
    relative_path = info['relative_path']
    
    # Define the full source path by joining the base directory with the relative path.
    src_img_path = os.path.join(base_img_dir, relative_path)
    dst_img_path = os.path.join(images_dir, new_file)
    
    # Copy the image to the output folder with the new name.
    if os.path.exists(src_img_path):
        shutil.copy(src_img_path, dst_img_path)
    else:
        print(f"Warning: {src_img_path} not found.")
    
    # Create a label file with the same base name as the new image file.
    label_file = os.path.splitext(new_file)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_file)
    
    # Get the annotations for this image (if any)
    anns = annotations_by_image.get(image_id, [])
    
    # Write annotations in YOLO format:
    # Each line: <category_id> <center_x> <center_y> <norm_width> <norm_height>
    with open(label_path, 'w') as f:
        for ann in anns:
            cat_id = ann['category_id'] - 1
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height
            f.write(f"{cat_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
