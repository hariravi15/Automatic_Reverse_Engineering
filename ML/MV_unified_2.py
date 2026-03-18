import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import json
import os
from PIL import Image
from collections import Counter
from typing import List, Dict, Tuple, Any
import math
import pickle
import random
import matplotlib.pyplot as plt
import argparse
import numpy as np


#0. Configuration ---
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
    VOCAB_SIZE = None
    MAX_SEQ_LENGTH = 150
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 70
    EARLY_STOPPING_PATIENCE = 5
    GRAD_CLIP_NORM = 1.0
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = None
    NUM_WORKERS = 0


config = Config()
print(f"Using device: {config.DEVICE}")



def parse_json_to_sequence(json_data: Dict[str, Any]) -> List[str]:

    sequence: List[str] = []

    for step in json_data.get("steps", []):
        # Ignore raw string steps like '<sos>' or '<eos>'
        if not isinstance(step, dict):
            continue

        op = step.get("op")
        if not op:
            continue
        if op == "plane":
            sequence.append(f"plane={step.get('plane', 'XY')}")
        elif op == "start_sketch":
            sequence.append("ENTITY_START__Sketch")
        elif op == "end_sketch":
            sequence.append("ENTITY_END__Sketch")

        # --- Sketch Primitives (Curves) ---
        elif op == "sketch_circle":
            sequence.append("CURVE_START__Circle")
            if "center_xy" in step and isinstance(step["center_xy"], list) and len(step["center_xy"]) == 2:
                sequence.append(f"center_x={step['center_xy'][0]}")
                # --- CORRECTED THIS LINE ---
                sequence.append(f"center_y={step['center_xy'][1]}")
            if "dim" in step:
                sequence.append(f"placeholder={step['dim']}")
            sequence.append("CURVE_END__Circle")

        elif op == "sketch_rectangle":
            sequence.append("CURVE_START__Rectangle")
            if "center_xy" in step and isinstance(step["center_xy"], list) and len(step["center_xy"]) == 2:
                sequence.append(f"center_x={step['center_xy'][0]}")
                sequence.append(f"center_y={step['center_xy'][1]}")
            if "dim" in step:
                sequence.append(f"placeholder={step['dim']}")
            sequence.append("CURVE_END__Rectangle")

        elif op == "sketch_line":
            sequence.append("CURVE_START__Line")
            if "center_xy" in step and isinstance(step["center_xy"], list) and len(step["center_xy"]) == 2:
                sequence.append(f"center_x={step['center_xy'][0]}")
                sequence.append(f"center_y={step['center_xy'][1]}")
            if "dim" in step:
                sequence.append(f"placeholder={step['dim']}")
            sequence.append("CURVE_END__Line")

        # --- 3D Operations (Corrected Logic) ---
        elif op == "start_extrude":
            sequence.append("ENTITY_START__Extrude")
        elif op == "extrude":
            sequence.append("operation_type=NewBody")  # Add material
            if "dim" in step:
                sequence.append(f"placeholder_distance={step['dim']}")
        elif op == "extrude_cut":
            sequence.append("operation_type=Cut")  # Remove material
            if "dim" in step:
                sequence.append(f"placeholder_distance={step['dim']}")
        elif op == "end_extrude":
            sequence.append("ENTITY_END__Extrude")

        # --- Fillet Operations ---
        elif op == "start_fillet":
            sequence.append("ENTITY_START__Fillet")
        elif op == "fillet_all_edges":
            sequence.append("fillet_type=all_edges")
            if "dim" in step:
                sequence.append(f"placeholder_radius={step['dim']}")
        elif op == "fillet_all_vertical_corner":
            sequence.append("fillet_type=all_vertical")
            if "dim" in step:
                sequence.append(f"placeholder_radius={step['dim']}")
        elif op == "fillet_all_vertical_edges":
            sequence.append("fillet_type=all_vertical")
            if "dim" in step:
                sequence.append(f"placeholder_radius={step['dim']}")
        elif op == "fillet_all_horizontal_edges":
            sequence.append("fillet_type=all_horizontal")
            if "dim" in step:
                sequence.append(f"placeholder_radius={step['dim']}")
        elif op == "fillet_all_top_corner":
            sequence.append("fillet_type=all_top")
            if "dim" in step:
                sequence.append(f"placeholder_radius={step['dim']}")
        elif op == "fillet_all_bottom_corner":
            sequence.append("fillet_type=all_bottom")
            if "dim" in step:
                sequence.append(f"placeholder_radius={step['dim']}")

        elif op == "end_fillet":
            sequence.append("ENTITY_END__Fillet")

    return sequence if sequence else [config.UNK_TOKEN]


class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {0: config.PAD_TOKEN, 1: config.SOS_TOKEN, 2: config.EOS_TOKEN, 3: config.UNK_TOKEN}
        self.stoi = {token: idx for idx, token in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list: List[List[str]]):
        frequencies, idx = Counter(), len(self.itos)
        for sentence in sentence_list:
            for word in sentence: frequencies[word] += 1
        for word, count in frequencies.items():
            if count >= 1 and word not in self.stoi: self.stoi[word], self.itos[idx], idx = idx, word, idx + 1
        config.VOCAB_SIZE = len(self.itos)

    def numericalize(self, text_sequence: List[str]) -> List[int]:
        return [self.stoi.get(token, self.stoi[config.UNK_TOKEN]) for token in text_sequence]


# --- 2. Data Loading ---
class MultiViewCADDataset(Dataset):
    def __init__(self, data_root_dir: str, json_dir: str, file_ids: List[str], vocab: Vocabulary, transform: Any,
                 num_views: int):
        self.data_root_dir, self.json_dir = data_root_dir, json_dir
        self.vocab, self.transform, self.num_views = vocab, transform, num_views
        self.view_names = ["top.png", "bottom.png", "left.png", "right.png", "front.png" , "back.png"][:num_views]
        self.file_ids = [fid for fid in file_ids if os.path.exists(os.path.join(json_dir, fid + ".json"))]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        model_id = self.file_ids[index]
        model_view_dir = os.path.join(self.data_root_dir, model_id)
        images = []
        for view_name in self.view_names:
            img_path = os.path.join(model_view_dir, view_name)
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform: image = self.transform(image)
                images.append(image)
            except FileNotFoundError:
                images.append(torch.zeros(3, 256, 256))
        images_tensor = torch.stack(images)
        json_path = os.path.join(self.json_dir, model_id + ".json")
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        token_sequence = parse_json_to_sequence(json_data)
        numericalized_seq = [self.vocab.stoi[config.SOS_TOKEN]] + self.vocab.numericalize(token_sequence) + [
            self.vocab.stoi[config.EOS_TOKEN]]
        padded_sequence = torch.full((config.MAX_SEQ_LENGTH,), fill_value=self.vocab.stoi[config.PAD_TOKEN],
                                     dtype=torch.long)
        seq_len = min(len(numericalized_seq), config.MAX_SEQ_LENGTH)
        padded_sequence[:seq_len] = torch.tensor(numericalized_seq[:seq_len], dtype=torch.long)
        return images_tensor, padded_sequence


# --- 3. Model Definition ---
class MVCNN_Encoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        cnn_base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        #original_weights = cnn_base.conv1.weight.data
        #new_weights = original_weights.mean(dim=1, keepdim=True)
        #cnn_base.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #cnn_base.conv1.weight.data = new_weights
        self.view_feature_extractor = nn.Sequential(*list(cnn_base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cnn_base.fc.in_features, embed_dim)

    def forward(self, multi_view_images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views = multi_view_images.size(0), multi_view_images.size(1)
        reshaped_images = multi_view_images.view(-1, *multi_view_images.shape[2:])
        view_features = self.view_feature_extractor(reshaped_images)
        pooled_view_features = self.pool(view_features).view(batch_size, num_views, -1)
        #combined_features, _ = torch.max(pooled_view_features, dim=1)
        #return self.fc(combined_features)
        return pooled_view_features


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
        # --- THIS IS THE CORRECTED LINE ---
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, config.MAX_SEQ_LENGTH)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                                                   dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    # In CADTransformerDecoder:
    def forward(self, memory, trg, tgt_mask, tgt_key_padding_mask):
        trg_embed = self.pos_encoder(self.embedding(trg) * math.sqrt(self.embed_dim))

        # Ask the decoder for the attention weights by setting need_weights=True
        # It will return the cross-attention from the last layer.
        output, attention = self.transformer_decoder(
            tgt=trg_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
            need_weights=True,  # This is the key change
            average_attn_weights=False  # Get weights for each head
        )

        # Now return both the final output and the attention weights
        return self.fc_out(output), attention


class MultiViewImageToCADModel(nn.Module):
    def __init__(self, encoder, decoder, img_dim, transformer_dim, device):
        super().__init__()
        self.mvcnn_encoder, self.transformer_decoder, self.device = encoder, decoder, device
        self.projection = nn.Linear(img_dim, transformer_dim) #if img_dim != transformer_dim else nn.Identity()

    def make_tgt_mask(self, sz: int) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(sz).to(self.device)

    def forward(self, images, sequences):
        # The memory is now a sequence of 6 vectors, not a single vector.
        # Shape of encoder_output: (batch, 6, img_feature_dim)
        encoder_output = self.mvcnn_encoder(images)
        # Shape of memory: (batch, 6, transformer_dim)
        memory = self.projection(encoder_output)

        decoder_in = sequences[:, :-1]
        tgt_pad_mask = (decoder_in == config.vocab.stoi[config.PAD_TOKEN])
        tgt_mask = self.make_tgt_mask(decoder_in.size(1))

        # The decoder will now return (output, attention_weights)
        output, attention_weights = self.transformer_decoder(memory, decoder_in, tgt_mask, tgt_pad_mask)
        return output

    def generate_sequence(self, images, vocab):
        self.eval()
        sos, eos = vocab.stoi[config.SOS_TOKEN], vocab.stoi[config.EOS_TOKEN]
        with torch.no_grad():
            memory = self.projection(self.mvcnn_encoder(images)).unsqueeze(1)
            ids = [sos]
            for i in range(config.MAX_SEQ_LENGTH - 1):
                trg_tensor = torch.LongTensor([ids]).to(self.device)
                tgt_mask = self.make_tgt_mask(trg_tensor.size(1))
                logits = self.transformer_decoder(memory, trg_tensor, tgt_mask, None)[0, -1, :]
                if i == 0:
                    k = min(5, len(vocab))
                    probs, indices = torch.topk(torch.softmax(logits, dim=-1), k)
                    print("\n--- DEBUG: Top 5 predictions for first token ---")
                    for p, idx in zip(probs, indices): print(
                        f"Token: '{vocab.itos.get(idx.item(), '?')}', Prob: {p.item():.4f}")
                    print("---------------------------------------------")
                pred_id = logits.argmax(dim=-1).item()
                ids.append(pred_id)
                if pred_id == eos: break
        return [vocab.itos.get(idx, "?") for idx in ids]


# --- 4. Helper Functions ---
def load_multi_view_images(dir_path, transform, view_names):
    images = []
    for name in view_names:
        img_path = os.path.join(dir_path, name)
        try:
            image = Image.open(img_path).convert("RGB")
            if transform: image = transform(image)
            images.append(image)
        except FileNotFoundError:
            print(f"Warning: View '{img_path}' not found. Using a black image placeholder.")
            images.append(torch.zeros(3, 256, 256))
    if not images: return None
    return torch.stack(images).unsqueeze(0)


def tokens_to_json_script(tokens):
    return {"status": "implement `tokens_to_json_script`", "generated_tokens": tokens}


def plot_loss_history(train_hist, val_hist, save_path):
    plt.figure(figsize=(10, 6));
    plt.plot(train_hist, label='Training Loss', marker='o');
    plt.plot(val_hist, label='Validation Loss', marker='o')
    plt.title('Training & Validation Loss');
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)
    plt.savefig(save_path);
    print(f"Loss plot saved to: {save_path}");
    plt.close()


# --- 5. Main Execution Logic ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    if len(loader) == 0: return 0.0
    for batch_idx, (images, sequences) in enumerate(loader):
        images, sequences = images.to(config.DEVICE), sequences.to(config.DEVICE)
        optimizer.zero_grad()
        predictions = model(images, sequences)
        targets = sequences[:, 1:].reshape(-1)
        preds = predictions.reshape(-1, config.VOCAB_SIZE)
        loss = criterion(preds, targets)
        loss.backward()
        if config.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 2 == 0: print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    return total_loss / len(loader)


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    if len(loader) == 0: return 0.0
    with torch.no_grad():
        for images, sequences in loader:
            images, sequences = images.to(config.DEVICE), sequences.to(config.DEVICE)
            predictions = model(images, sequences)
            targets = sequences[:, 1:].reshape(-1)
            preds = predictions.reshape(-1, config.VOCAB_SIZE)
            loss = criterion(preds, targets)
            total_loss += loss.item()
    return total_loss / len(loader)




def train_model(args):
    print("\n--- Starting Training Mode ---")
    split_data = json.load(open(config.DATASET_SPLIT_JSON_PATH))
    train_ids, val_ids = split_data.get('train_ids', []), split_data.get('val_ids', [])

    print("Building vocabulary from training data...")
    all_seqs = []
    for fid in train_ids:
        json_path = os.path.join(config.JSON_DIR, fid + ".json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                all_seqs.append(parse_json_to_sequence(json.load(f)))
        else:
            print(f"Warning: JSON file not found: {json_path}")
    if not all_seqs: raise ValueError("No JSON files found. Check `JSON_DIR` and `train_ids`.")

    vocab = Vocabulary();
    vocab.build_vocabulary(all_seqs)
    with open(config.VOCAB_SAVE_PATH, "wb") as f:
        pickle.dump(vocab, f)
    config.vocab, config.VOCAB_SIZE = vocab, len(vocab)
    print(f"Vocabulary built and saved. Size: {config.VOCAB_SIZE}")


    train_transform = transforms.Compose([
        # Data Augmentations to make CAD images look more real
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),

        # Standard Conversions
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),

        # Add Noise and Blur AFTER converting to a tensor
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)), # Add Gaussian noise
        transforms.Lambda(lambda x: x.clamp(0, 1)), # Ensure pixel values stay in [0, 1] range

        # Step 5: Normalize with standard ImageNet values, as ResNet was trained on them.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    val_transform = transforms.Compose([
        #transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Use the same normalization for validation and testing.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = MultiViewCADDataset(config.DATA_ROOT_DIR, config.JSON_DIR, train_ids, vocab, train_transform,
                                        config.NUM_VIEWS)
    val_dataset = MultiViewCADDataset(config.DATA_ROOT_DIR, config.JSON_DIR, val_ids, vocab, val_transform,
                                      config.NUM_VIEWS)

    #if len(train_dataset) < config.BATCH_SIZE:
        #print(
            #f"Warning: BATCH_SIZE ({config.BATCH_SIZE}) is larger than training dataset size ({len(train_dataset)}). Adjusting drop_last to False.")
        #drop_last_flag = False
    #else:
        #drop_last_flag = True

    #train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=drop_last_flag,
                              #num_workers=config.NUM_WORKERS)
    # In the train_model function

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    encoder = MVCNN_Encoder(config.IMG_EMBED_DIM)

    # The original encoder was modified to take 1 channel. We must do that again.
    #original_weights = encoder.view_feature_extractor[0].weight.data
    #new_weights = original_weights.mean(dim=1, keepdim=True)
    #encoder.view_feature_extractor[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #encoder.view_feature_extractor[0].weight.data = new_weights
    model = MultiViewImageToCADModel(
        encoder, # Pass the single, correct encoder instance
        CADTransformerDecoder(config.VOCAB_SIZE, config.TRANSFORMER_EMBED_DIM, config.NUM_HEADS, config.NUM_DECODER_LAYERS, config.TRANSFORMER_FF_DIM, config.DROPOUT_RATE),
        config.IMG_EMBED_DIM, config.TRANSFORMER_EMBED_DIM, config.DEVICE
    ).to(config.DEVICE)


    #model = MultiViewImageToCADModel(MVCNN_Encoder(config.IMG_EMBED_DIM),
                                     #CADTransformerDecoder(config.VOCAB_SIZE, config.TRANSFORMER_EMBED_DIM,
                                                           #config.NUM_HEADS, config.NUM_DECODER_LAYERS,
                                                           #config.TRANSFORMER_FF_DIM, config.DROPOUT_RATE),
                                     #config.IMG_EMBED_DIM, config.TRANSFORMER_EMBED_DIM, config.DEVICE).to(
        #config.DEVICE)
    weights = torch.ones(config.VOCAB_SIZE).to(config.DEVICE)
    weights[vocab.stoi[config.EOS_TOKEN]] = 6.0 # Increase penalty for <eos>
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi[config.PAD_TOKEN], weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


    train_hist, val_hist, best_val_loss = [], [], float('inf')
    epochs_no_improve = 0
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step()
        train_hist.append(train_loss);val_hist.append(val_loss)
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        if len(val_dataset) > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"  New best model saved to {config.MODEL_SAVE_PATH}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {config.EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break

    print("\nTraining Finished.");
    plot_loss_history(train_hist, val_hist, config.PLOT_SAVE_PATH)


def test_model(args):
    print("\n--- Starting Test/Inference Mode ---")
    if not args.test_dir or not os.path.isdir(args.test_dir):
        raise ValueError(f"Test directory not found. Use --test_dir. Path: {args.test_dir}")
    if not os.path.exists(config.VOCAB_SAVE_PATH):
        raise FileNotFoundError(f"Vocabulary file not found: {config.VOCAB_SAVE_PATH}. Please train first.")

    with open(config.VOCAB_SAVE_PATH, "rb") as f:
        vocab = pickle.load(f)
    config.vocab, config.VOCAB_SIZE = vocab, len(vocab)
    print(f"Vocabulary loaded. Size: {len(vocab)}")

    model = MultiViewImageToCADModel(MVCNN_Encoder(config.IMG_EMBED_DIM),
                                     CADTransformerDecoder(config.VOCAB_SIZE, config.TRANSFORMER_EMBED_DIM,
                                                           config.NUM_HEADS, config.NUM_DECODER_LAYERS,
                                                           config.TRANSFORMER_FF_DIM, config.DROPOUT_RATE),
                                     config.IMG_EMBED_DIM, config.TRANSFORMER_EMBED_DIM, config.DEVICE).to(
        config.DEVICE)
    print(f"Loading trained weights from {config.MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    view_names = ["top.png", "bottom.png", "left.png", "right.png" , "front.png" , "back.png"][:config.NUM_VIEWS]

    input_tensor = load_multi_view_images(args.test_dir, transform, view_names).to(config.DEVICE)
    if input_tensor is not None:
        generated_tokens = model.generate_sequence(input_tensor, vocab)
        structured_json = tokens_to_json_script(generated_tokens)
        print("\n--- Generated JSON Script ---\n", json.dumps(structured_json, indent=2))
        output_filename = f"generated_output_for_{os.path.basename(args.test_dir)}.json"
        with open(output_filename, "w") as f: json.dump(structured_json, f, indent=2)
        print(f"\nSaved output to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test the Multi-View CAD Model.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help="Set to 'train' to start training, or 'test' to run inference.")
    parser.add_argument('--test_dir', type=str,
                        help="Path to the directory with multi-view images for testing (e.g., 'D:\\Dataset\\custom_dataset\\test'). Required for test mode.")
    args = parser.parse_args()

    if os.name == 'nt':
        config.NUM_WORKERS = 0
    else:
        config.NUM_WORKERS = 2

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)