import os
import cv2
import numpy as np
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
import torch

DATASET_PATH = "dataset"
MODEL_PATH = "model_patchcore.ckpt"
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Folder(
    name="dataset",
    root=DATASET_PATH,
    normal_dir="train/good",
    abnormal_dir="test/defective",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=2
)


model = Patchcore(
    backbone="wide_resnet50_2",
    pre_trained=True
)


engine = Engine(
    datamodule=dataset,
    default_root_dir="results",
    versioned_dir=False
)


print(" Training model on normal images...")
engine.fit(model)

torch.save(model.state_dict(), MODEL_PATH)
print("Training completed and model saved.")

print("\n Running inference on test images...")

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
model.to(DEVICE)

test_good_path = os.path.join(DATASET_PATH, "test", "good")
test_defect_path = os.path.join(DATASET_PATH, "test", "defective")
test_folders = [test_good_path, test_defect_path]

for folder in test_folders:
    if not os.path.exists(folder):
        continue

    print(f"\n Testing folder: {folder}")

    for file in os.listdir(folder):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            result = model.predict_step(img_tensor, 0)

        anomaly_map = result["anomaly_map"][0].cpu().numpy()
        anomaly_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, binary_mask = cv2.threshold(anomaly_map, 128, 255, cv2.THRESH_BINARY)

        defect_pixels = np.sum(binary_mask > 0)
        total_pixels = binary_mask.size
        defect_percent = (defect_pixels / total_pixels) * 100

        print(f"{file}: Defect = {defect_percent:.2f}%")

        vis = cv2.addWeighted(img_resized, 0.7, cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET), 0.3, 0)
        cv2.putText(vis, f"Defect: {defect_percent:.2f}%", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out_path = os.path.join("results", os.path.basename(file))
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(out_path, vis)

print("\n All test images processed. Check 'results/' for outputs.")
