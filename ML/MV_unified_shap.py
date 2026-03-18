import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import shap
import os
import pickle
import math
import argparse

class Config:
    DATA_ROOT_DIR = r"D:\Dataset\cuboid_final_dataset"
    JSON_DIR = r"D:\Dataset\cuboid_final_dataset\procedural_json"
    DATASET_SPLIT_JSON_PATH = r"D:\Dataset\cuboid_final_dataset\folder_split_stratified2.json"
    MODEL_SAVE_PATH = r"D:\Dataset\cuboid_final_dataset\mvcnn_grayscale_model.pth"
    VOCAB_SAVE_PATH = r"D:\Dataset\cuboid_final_dataset\vocab_multiview.pkl"
    PLOT_SAVE_PATH = r"D:\Dataset\cuboid_final_dataset\training_loss_plot.png"

    NUM_VIEWS = 6
    IMG_EMBED_DIM = 512
    TRANSFORMER_EMBED_DIM = 256
    TRANSFORMER_FF_DIM = 512
    NUM_HEADS = 8
    NUM_DECODER_LAYERS = 2
    DROPOUT_RATE = 0.2
    VOCAB_SIZE = None  # Will be loaded from vocab file
    MAX_SEQ_LENGTH = 150
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = None
config = Config()

class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {0: config.PAD_TOKEN, 1: config.SOS_TOKEN, 2: config.EOS_TOKEN, 3: config.UNK_TOKEN}
        self.stoi = {token: idx for idx, token in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        pass

    def numericalize(self, text_sequence):
        return [self.stoi.get(token, self.stoi[config.UNK_TOKEN]) for token in text_sequence]

class MVCNN_Encoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        cnn_base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.view_feature_extractor = nn.Sequential(*list(cnn_base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cnn_base.fc.in_features, embed_dim)

    def forward(self, multi_view_images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views = multi_view_images.size(0), multi_view_images.size(1)
        reshaped_images = multi_view_images.view(-1, *multi_view_images.shape[2:])
        view_features = self.view_feature_extractor(reshaped_images)
        pooled_view_features = self.pool(view_features).view(batch_size, num_views, -1)
        combined_features, _ = torch.max(pooled_view_features, dim=1)
        return self.fc(combined_features)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = config.MAX_SEQ_LENGTH):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class CADTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, config.MAX_SEQ_LENGTH)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, memory, trg, tgt_mask, tgt_key_padding_mask):
        trg_embed = self.pos_encoder(self.embedding(trg) * math.sqrt(self.embed_dim))
        output = self.transformer_decoder(tgt=trg_embed, memory=memory, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(output)

class MultiViewImageToCADModel(nn.Module):
    def __init__(self, encoder, decoder, img_dim, transformer_dim, device):
        super().__init__()
        self.mvcnn_encoder, self.transformer_decoder, self.device = encoder, decoder, device
        self.projection = nn.Linear(img_dim, transformer_dim) if img_dim != transformer_dim else nn.Identity()

def run_shap_explanation(image_dir):
    print(f"Using device: {config.DEVICE}")
    if not os.path.exists(config.VOCAB_SAVE_PATH):
        raise FileNotFoundError(f"Vocabulary file not found at {config.VOCAB_SAVE_PATH}. Please train the model first.")

    with open(config.VOCAB_SAVE_PATH, "rb") as f:
        vocab = pickle.load(f)
    config.vocab = vocab
    config.VOCAB_SIZE = len(vocab)
    print(f"Vocabulary loaded. Size: {config.VOCAB_SIZE}")

    print("⚙️  Loading custom model and weights...")
    encoder = MVCNN_Encoder(config.IMG_EMBED_DIM)
    decoder = CADTransformerDecoder(config.VOCAB_SIZE, config.TRANSFORMER_EMBED_DIM, config.NUM_HEADS,config.NUM_DECODER_LAYERS, config.TRANSFORMER_FF_DIM, config.DROPOUT_RATE)
    model = MultiViewImageToCADModel(encoder, decoder, config.IMG_EMBED_DIM,config.TRANSFORMER_EMBED_DIM, config.DEVICE)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_SAVE_PATH}. Please train the model first.")

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    print(" Model loaded successfully.")

    class ShapModelWrapper(nn.Module):
        def __init__(self, model_encoder):
            super().__init__()
            self.encoder = model_encoder

        def forward(self, multi_view_tensor):
            embedding = self.encoder(multi_view_tensor)
            return torch.sum(embedding, dim=1, keepdim=True)

    transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    view_names = ["front.png", "back.png", "top.png", "bottom.png", "left.png", "right.png"]
    images = []
    for view_name in view_names:
        img_path = os.path.join(image_dir, view_name)
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        except FileNotFoundError:
            print(f"Warning: View '{img_path}' not found. Using a black placeholder.")
            images.append(Image.new('RGB', (256, 256), (0, 0, 0)))

    tensors = [transform(img) for img in images]
    input_tensor = torch.stack(tensors).unsqueeze(0).to(config.DEVICE)
    print(f"Input data prepared with shape: {input_tensor.shape}")
    print("\n Calculating SHAP values... This might take a moment.")
    background = torch.randn_like(input_tensor)[:1].repeat(10, 1, 1, 1, 1)
    shap_wrapper = ShapModelWrapper(model.mvcnn_encoder)
    explainer = shap.GradientExplainer(shap_wrapper, background)
    shap_values = explainer.shap_values(input_tensor)
    # Step F: Visualize the explanation (Manual Plotting)
    print("🎨 Generating visualization...")

    # Import matplotlib for direct plotting control


    # The shap_values array is 6D: (1, 6, 3, 256, 256, 1).
    # Squeeze out the singleton dimensions to get a 4D array of shape (6, 3, 256, 256).
    shap_values_squeezed = shap_values.squeeze()

    # Transpose the 4D array from (views, channels, height, width) to (views, height, width, channels).
    # Final shape: (6, 256, 256, 3)
    shap_values_for_plot = np.transpose(shap_values_squeezed, (0, 2, 3, 1))

    # De-normalize the original image for proper display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(config.DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(config.DEVICE)
    pixel_values = (input_tensor * std + mean).clamp(0, 1).squeeze(0).cpu().numpy()
    pixel_values_for_plot = np.transpose(pixel_values, (0, 2, 3, 1))

    # --- Manually Create the Plot with Matplotlib ---

    # Create a figure with 6 rows and 2 columns
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(8, 24))

    for i, view_name in enumerate(view_names):
        # --- Column 1: Original Image ---
        axes[i, 0].imshow(pixel_values_for_plot[i])
        axes[i, 0].set_title(f"Original - {view_name.split('.')[0]}")
        axes[i, 0].axis('off')

        # --- Column 2: SHAP Heatmap Overlay ---
        # Get the SHAP values for the current view
        shap_v = shap_values_for_plot[i]
        # To create a heatmap, we take the sum of the absolute SHAP values across the color channels
        shap_heatmap = np.sum(np.abs(shap_v), axis=-1)

        # Normalize the heatmap for better color contrast
        max_val = np.percentile(shap_heatmap, 99.9)
        if max_val > 0:
            shap_heatmap /= max_val

        # Plot the original image first
        axes[i, 1].imshow(pixel_values_for_plot[i])
        # Overlay the heatmap with some transparency
        axes[i, 1].imshow(shap_heatmap, cmap='viridis', alpha=0.5)
        axes[i, 1].set_title(f"SHAP Overlay - {view_name.split('.')[0]}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # We are temporarily commenting out the crashing code to see the debug info.
    # The `explainer` has already returned a 4D numpy array: (views, channels, height, width).
    # We transpose it directly to (views, height, width, channels) for plotting.
    # shap_values_for_plot = np.transpose(shap_values, (0, 2, 3, 1))

    # De-normalize the original image for proper display
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(config.DEVICE)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(config.DEVICE)
    # This part was already correct, squeezing the batch dim before transposing.
    # pixel_values = (input_tensor * std + mean).clamp(0, 1).squeeze(0).cpu().numpy()
    # pixel_values_for_plot = np.transpose(pixel_values, (0, 2, 3, 1))

    # Plot the SHAP values for all 6 views
    # shap.image_plot(
    #     shap_values=list(shap_values_for_plot),
    #     pixel_values=list(pixel_values_for_plot),
    #     labels=[[f'View: {name.split(".")[0]}'] for name in view_names],
    #     show=True
    # )

    # --- END OF NEW DEBUG CODE ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for the Multi-View CAD Model.")
    parser.add_argument('--image_dir', type=str, required=True,help="Path to the directory containing the 6 multi-view images for a single object.")
    args = parser.parse_args()
    run_shap_explanation(args.image_dir)



    #python MV_unified_shap.py --image_dir "D:\test3"