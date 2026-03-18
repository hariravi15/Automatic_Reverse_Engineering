import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse


def analyze_domain_gap(real_image_path, synthetic_image_path, output_path):
    """
    Loads a real and a synthetic image, applies the same normalization as the ML model,
    and generates a visual analysis of the differences.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    try:
        real_img = Image.open(real_image_path)
        synthetic_img = Image.open(synthetic_image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your image paths.")
        return

    real_tensor = transform(real_img)
    synthetic_tensor = transform(synthetic_img)

    real_np = real_tensor.squeeze().numpy()
    synthetic_np = synthetic_tensor.squeeze().numpy()

    diff_np = np.abs(real_np - synthetic_np)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Sim-to-Real Domain Gap Analysis', fontsize=16)

    axes[0, 0].imshow(real_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Real Image (Normalized)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(synthetic_np, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Synthetic "Twin" (Normalized)')
    axes[0, 1].axis('off')

    im = axes[1, 0].imshow(diff_np, cmap='magma', vmin=0, vmax=1)
    axes[1, 0].set_title('Absolute Pixel Difference')
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0])

    axes[1, 1].hist(real_np.ravel(), bins=256, range=(0, 1), color='blue', alpha=0.7, label='Real Image')
    axes[1, 1].hist(synthetic_np.ravel(), bins=256, range=(0, 1), color='orange', alpha=0.7, label='Synthetic Image')
    axes[1, 1].set_title('Pixel Intensity Histograms')
    axes[1, 1].set_xlabel('Pixel Intensity (0=Black, 1=White)')
    axes[1, 1].set_ylabel('Pixel Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Analysis plot saved to: {output_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze the domain gap between a real and a synthetic image.")
    # CORRECT DEFINITION: 'help' provides a description.
    parser.add_argument('--real_image', type=str, required=True, help="Path to the real-world image.")
    parser.add_argument('--synthetic_image', type=str, required=True, help="Path to the synthetic 'twin' image.")
    parser.add_argument('--output', type=str, default='domain_gap_analysis.png',
                        help="Path to save the output analysis plot.")
    args = parser.parse_args()

    analyze_domain_gap(args.real_image, args.synthetic_image, args.output)



