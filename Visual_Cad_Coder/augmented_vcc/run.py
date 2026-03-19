import random
import torch

from compact import MultiViewDataset
from compact import MultiViewMask2Former
from compact import train


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_iou(pred_mask, gt_mask):
    """Compute IoU between two masks"""

    intersection = (pred_mask & gt_mask).float().sum()
    union = (pred_mask | gt_mask).float().sum()

    if union == 0:
        return torch.tensor(0.0)

    return intersection / union


def evaluate_random_sample(model, dataset, iou_threshold=0.5):

    model.eval()

    idx = random.randint(0, len(dataset) - 1)

    images, gt_masks = dataset[idx]

    images = images.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(images)

    pred_masks = outputs.pred_masks.sigmoid()[0].cpu()

    # binarize predictions
    pred_masks = pred_masks > 0.5

    gt_masks = gt_masks > 0

    correct_segments = 0
    total_segments = gt_masks.shape[0]

    for i in range(total_segments):

        gt = gt_masks[i]

        best_iou = 0

        for j in range(pred_masks.shape[0]):

            pred = pred_masks[j]

            iou = compute_iou(pred, gt)

            if iou > best_iou:
                best_iou = iou

        if best_iou >= iou_threshold:
            correct_segments += 1

    print("\nEvaluation on random object:")
    print("Correct segments:", correct_segments)
    print("Total segments:", total_segments)
    print("Accuracy:", correct_segments / total_segments)


def main():

    dataset_path = "C:/Users/mb01/Desktop/Treburi/Vision/CAD/Sample_dataset_output"

    dataset = MultiViewDataset(dataset_path)

    model = MultiViewMask2Former().to(DEVICE)

 #   model = train(model, dataset)

  #  evaluate_random_sample(model, dataset)


if __name__ == "__main__":
    main()