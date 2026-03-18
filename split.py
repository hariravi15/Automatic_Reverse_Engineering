import os
import json
import random
import re

# --- Configuration ---
# Your main dataset folder which contains the 8100 model subfolders.
DATA_ROOT_DIR = r"D:\Dataset\cuboid_final_dataset"

# --- Output Configuration ---
OUTPUT_DIR = r"D:\Dataset\cuboid_final_dataset"  # <-- ADD YOUR DESIRED SAVE FOLDER HERE
OUTPUT_FILENAME = "folder_split_stratified2.json"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# --- Split Ratios ---
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15


# TEST_RATIO is the remainder (0.10)

def group_folders_by_shape(root_dir):
    """
    Scans the root directory, identifies all model folders, and groups them by base shape name.
    Example: "cuboid_with_hole_001" and "cuboid_with_hole_002" are grouped under "cuboid_with_hole".
    """
    shapes_to_folders = {}

    try:
        # 1. List all items in the root directory that are folders.
        all_model_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except FileNotFoundError:
        return None, f"Dataset directory '{root_dir}' not found."

    if not all_model_folders:
        return None, "No model folders found in the root directory."

    # 2. Group folders by parsing their names.
    for folder_name in all_model_folders:
        # This removes trailing numbers and any space/underscore before them.
        # e.g., "cuboid_267" -> "cuboid", "shape name 123" -> "shape name"
        shape_category = re.sub(r'[ _]\d+$', '', folder_name).strip()

        # Add the folder to the corresponding shape category list.
        if shape_category not in shapes_to_folders:
            shapes_to_folders[shape_category] = []
        shapes_to_folders[shape_category].append(folder_name)

    return shapes_to_folders, None


if __name__ == "__main__":
    print(f"Starting stratified folder split for directory: {DATA_ROOT_DIR}")

    # --- Step 1: Group all model folders by their shape type ---
    grouped_folders, error_message = group_folders_by_shape(DATA_ROOT_DIR)

    if error_message:
        print(f"Error: {error_message}")
        exit()

    print(f"Found {len(grouped_folders)} unique shape categories.")

    # --- Step 2: Perform the stratified split for each group ---
    final_train_folders = []
    final_val_folders = []
    final_test_folders = []

    for shape_category, folder_list in grouped_folders.items():
        # Shuffle the list of folders for this specific shape
        random.shuffle(folder_list)

        # Calculate split points for this shape's data
        num_total = len(folder_list)
        num_train = int(TRAIN_RATIO * num_total)
        num_val = int(VAL_RATIO * num_total)

        # Split and append the folder names to the final lists
        final_train_folders.extend(folder_list[:num_train])
        final_val_folders.extend(folder_list[num_train: num_train + num_val])
        final_test_folders.extend(folder_list[num_train + num_val:])

        print(
            f"  - Shape '{shape_category}': Total={num_total} -> Train={num_train}, Val={num_val}, Test={len(folder_list[num_train + num_val:])}")

    # --- Step 3: Final shuffle and summary ---
    # Shuffle the final lists to mix the shapes within each set for better training.
    random.shuffle(final_train_folders)
    random.shuffle(final_val_folders)
    random.shuffle(final_test_folders)

    print("\n--- Total Split Sizes ---")
    print(f"Total Training set size: {len(final_train_folders)} folders")
    print(f"Total Validation set size: {len(final_val_folders)} folders")
    print(f"Total Test set size: {len(final_test_folders)} folders")
    print(f"Total folders processed: {len(final_train_folders) + len(final_val_folders) + len(final_test_folders)}")

    # Prepare the data for JSON serialization
    dataset_split_data = {
        "train_ids": final_train_folders,
        "val_ids": final_val_folders,
        "test_ids": final_test_folders
    }

    # --- Step 4: Save the JSON file ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(dataset_split_data, f, indent=2)
        print(f"\n✅ Successfully created folder split file: {OUTPUT_JSON_PATH}")
    except IOError as e:
        print(f"❌ Error: Could not write to {OUTPUT_JSON_PATH}. Reason: {e}")