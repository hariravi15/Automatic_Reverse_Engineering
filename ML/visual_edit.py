import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys


def get_train_transform():
    """Defines the image transformation pipeline from your training script."""
    return transforms.Compose([
        # 1. Data Augmentations to make CAD images look more real
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),

        # 2. Standard Conversion to Tensor
        transforms.ToTensor(),

        # 3. Add Noise and Blur AFTER converting to a tensor
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),  # Add Gaussian noise
        transforms.Lambda(lambda x: x.clamp(0, 1)),  # Ensure pixel values stay in [0, 1] range

        # 4. Normalize with standard ImageNet values
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def denormalize_image(tensor):
    """
    Reverses the normalization on an image tensor to make it viewable.
    The model needs the normalized image, but we need to de-normalize it for display.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    denormalized_tensor = tensor.clone().permute(1, 2, 0)  # C, H, W -> H, W, C
    denormalized_tensor = denormalized_tensor * torch.tensor(std) + torch.tensor(mean)
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)

    return denormalized_tensor.numpy()


def visualize_augmentations(image_path):
    """Loads an image, applies transformations, and displays the result."""
    try:

        original_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        sys.exit(1)

    train_transform = get_train_transform()
    augmented_tensor = train_transform(original_image)
    augmented_image_for_display = denormalize_image(augmented_tensor)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display Original Image
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    # Display Augmented Image
    ax[1].imshow(augmented_image_for_display)
    ax[1].set_title("Augmented Image (How the Model Sees It)")
    ax[1].axis('off')

    plt.suptitle("Data Augmentation Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the effect of PyTorch image transformations on an input image.")
    parser.add_argument("image_path",type=str,help="The file path to the input image.")

    args = parser.parse_args()
    visualize_augmentations(args.image_path)


    #python visual_edit.py "D:\test2\top.png"