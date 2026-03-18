import pandas as pd
import os
import json
from PIL import Image
import io

# --- Configuration ---
# The name of the Parquet file you uploaded
parquet_file_path = 'C:/Users/hr73/Downloads/test-00000-of-00001.parquet'

# The folders where the script will save the images and the question file
inference_folder = 'inference'
image_folder = os.path.join(inference_folder, 'images')
question_file_path = os.path.join(inference_folder, 'qa_test_data_subset100.jsonl')

# --- Main Script ---
# Create the necessary folders
os.makedirs(image_folder, exist_ok=True)

# Read the Parquet file
print(f"Reading {parquet_file_path}...")
df = pd.read_parquet(parquet_file_path)

# Get the first 100 samples for the test set
subset_df = df.head(100)
print(f"Loaded {len(subset_df)} samples.")

# Open the question file for writing
with open(question_file_path, 'w') as f:
    # Loop through each of the 100 samples
    for index, row in subset_df.iterrows():
        # Get the image data from the 'image' column
        image_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(image_bytes))

        # Create a unique filename for the image
        image_filename = f"{row['deepcad_id']}.png"
        image_save_path = os.path.join(image_folder, image_filename)

        # Save the image as a PNG file
        image.save(image_save_path)

        # Create the JSON data for the question file
        question_data = {
            "question_id": int(row['deepcad_id']),
            "image": os.path.join('images', image_filename),  # Use relative path
            "text": row['prompt']
        }

        # Write the JSON data as a new line in the .jsonl file
        f.write(json.dumps(question_data) + '\n')

print(f"✅ Successfully created 100 images in '{image_folder}'")
print(f"✅ Successfully created the question file at '{question_file_path}'")