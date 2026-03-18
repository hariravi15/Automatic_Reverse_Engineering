import torch
import torch.nn as nn
import torchvision.models as models
import torch.onnx
import os
import math


# --- 1. CONFIGURATION ---
class Config:
    TRANSFORMER_EMBED_DIM = 256
    TRANSFORMER_FF_DIM = 512
    NUM_HEADS = 8
    NUM_DECODER_LAYERS = 3
    DROPOUT_RATE = 0.2
    NUM_VIEWS = 6
    MAX_SEQ_LENGTH = 150
    # Update this path if needed
    MODEL_SAVE_PATH = r"D:\Dataset\cuboid_final_dataset\mvcnn_attention_model.pth"


config = Config()


# --- 2. MODEL DEFINITIONS ---
class MVCNN_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        cnn_base = models.resnet34(weights=None)
        self.view_feature_extractor = nn.Sequential(*list(cnn_base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = cnn_base.fc.in_features

    def forward(self, multi_view_images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views = multi_view_images.size(0), multi_view_images.size(1)
        reshaped_images = multi_view_images.view(-1, *multi_view_images.shape[2:])
        view_features = self.view_feature_extractor(reshaped_images)
        pooled_view_features = self.pool(view_features).view(batch_size, num_views, -1)
        return pooled_view_features


# Decoder classes (Required for loading, even if unused)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 150):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x): return x


class CADTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.token_type_embed = nn.Embedding(3, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, memory, trg, tgt_mask, tgt_key_padding_mask): return None


class MultiViewImageToCADModel(nn.Module):
    def __init__(self, encoder, decoder, img_feature_dim, transformer_dim, device):
        super().__init__()
        self.mvcnn_encoder = encoder
        self.transformer_decoder = decoder
        self.device = device
        self.projection = nn.Linear(img_feature_dim, transformer_dim)

    def forward(self, images, sequences): return None


# --- 3. THE EXPORT FUNCTION ---
def export_single_view_for_visualization():
    print("\n--- Starting Single-View ONNX Export ---")

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"ERROR: Model file not found at {config.MODEL_SAVE_PATH}")
        return

    # Initialize Dummy Model
    dummy_vocab_size = 10
    encoder = MVCNN_Encoder()
    decoder = CADTransformerDecoder(dummy_vocab_size, 256, 8, 3, 512, 0.2)
    full_model = MultiViewImageToCADModel(encoder, decoder, 512, 256, torch.device('cpu'))

    print(f"Loading weights from: {config.MODEL_SAVE_PATH}")
    try:
        state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=torch.device('cpu'))
        # Filter out decoder weights to avoid size mismatch
        filtered_dict = {k: v for k, v in state_dict.items() if
                         "transformer_decoder" not in k and "projection" not in k}
        full_model.load_state_dict(filtered_dict, strict=False)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # --- CHANGED: Extract ONLY the Feature Extractor ---
    # This part takes a single image [1, 3, H, W] instead of 6 views
    my_feature_extractor = full_model.mvcnn_encoder.view_feature_extractor
    my_feature_extractor.eval()

    # --- CHANGED: Standard Image Input Shape ---
    # 1 Batch, 3 Channels, 256 Height, 256 Width
    dummy_input = torch.randn(1, 3, 256, 256)

    output_filename = "mvcnn_single_view_vis.onnx"

    print(f"Exporting to {output_filename}...")
    torch.onnx.export(
        my_feature_extractor,  # We export only the CNN backbone
        dummy_input,
        output_filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['feature_maps'],
        dynamic_axes={'input_image': {0: 'batch_size'},
                      'feature_maps': {0: 'batch_size'}}
    )

    print(f"Success! File saved at: {os.path.abspath(output_filename)}")
    print("INSTRUCTIONS:")
    print("1. Drag this file into Zetane Viewer.")
    print("2. Drag a SINGLE image (like top.png) into the input.")
    print(
        "3. IMPORTANT: If the image is huge (1109px), Zetane might ask to resize it. Try to use a smaller 256x256 image if possible.")


if __name__ == "__main__":
    export_single_view_for_visualization()