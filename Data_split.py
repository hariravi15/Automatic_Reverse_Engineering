import os
import json
import random

# --- Configuration ---
DATA_ROOT_DIR = r"D:\Dataset\cuboid_final_dataset"
FILE_EXTENSION_TO_SCAN = ".json"

# --- Output Configuration ---
OUTPUT_DIR = r"D:\Dataset\cuboid_final_dataset"  # <-- ADD YOUR DESIRED SAVE FOLDER HERE
OUTPUT_FILENAME = "dataset_split.json"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# --- Split Ratios ---
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15


def get_unique_model_ids(root_dir, extension):
    model_ids = set()
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                # This part removes the file extension to get the model ID
                model_id = filename[:-len(extension)]
                model_ids.add(model_id)
    return list(model_ids)


if __name__ == "__main__":
    if not os.path.isdir(DATA_ROOT_DIR):
        print(f"Error: Dataset directory '{DATA_ROOT_DIR}' not found. Please update DATA_ROOT_DIR.")
        exit()

    print(f"Scanning for unique model IDs in: {DATA_ROOT_DIR} with extension {FILE_EXTENSION_TO_SCAN}")
    all_model_ids = get_unique_model_ids(DATA_ROOT_DIR, FILE_EXTENSION_TO_SCAN)

    if not all_model_ids:
        print("No model IDs found. Check your directory, extension, and get_unique_model_ids function.")
        exit()

    print(f"Found {len(all_model_ids)} unique model IDs.")

    # Shuffle the IDs to ensure random distribution
    random.shuffle(all_model_ids)

    # Calculate split points
    num_total = len(all_model_ids)
    num_train = int(TRAIN_RATIO * num_total)
    num_val = int(VAL_RATIO * num_total)

    # Create the splits
    train_ids = all_model_ids[:num_train]
    val_ids = all_model_ids[num_train: num_train + num_val]
    test_ids = all_model_ids[num_train + num_val:]

    print(f"Training set size: {len(train_ids)}")
    print(f"Validation set size: {len(val_ids)}")
    print(f"Test set size: {len(test_ids)}")

    # Prepare the data for JSON serialization
    dataset_split_data = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids
    }

    # --- Save the JSON file ---
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(dataset_split_data, f, indent=2)
        print(f"Successfully created dataset split file: {OUTPUT_JSON_PATH}")

    except IOError as e:
        print(f"Error: Could not write to {OUTPUT_JSON_PATH}. Reason: {e}")