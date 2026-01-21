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
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import numpy as np


# config
class Config:
    DATA_ROOT_DIR = r"D:\Dataset\cuboid_final_dataset"
    JSON_DIR = r"D:\Dataset\cuboid_final_dataset\procedural_json"
    DATASET_SPLIT_JSON_PATH = r"D:\Dataset\cuboid_final_dataset\folder_split_stratified2.json"
    MODEL_SAVE_PATH = r"D:\Dataset\cuboid_final_dataset\mvcnn_attention_model.pth"
    VOCAB_SAVE_PATH = r"D:\Dataset\cuboid_final_dataset\vocab_multiview.pkl"
    PLOT_SAVE_PATH = r"D:\Dataset\cuboid_final_dataset\training_loss_plot.png"
    NUM_VIEWS = 6
    TRANSFORMER_EMBED_DIM = 256
    TRANSFORMER_FF_DIM = 512
    NUM_HEADS = 8
    NUM_DECODER_LAYERS = 3
    DROPOUT_RATE = 0.2
    VOCAB_SIZE = None
    MAX_SEQ_LENGTH = 150
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 150
    EARLY_STOPPING_PATIENCE = 25
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
        if not isinstance(step, dict): continue
        op = step.get("op")
        if not op: continue
        if op == "plane":
            sequence.append(f"plane={step.get('plane', 'XY')}")
        elif op == "start_sketch":
            sequence.append("ENTITY_START__Sketch")
        elif op == "end_sketch":
            sequence.append("ENTITY_END__Sketch")
        elif op == "sketch_circle":
            sequence.append("CURVE_START__Circle")
            if "center_xy" in step and isinstance(step["center_xy"], list) and len(step["center_xy"]) == 2:
                sequence.append(f"center_x={step['center_xy'][0]}")
                sequence.append(f"center_y={step['center_xy'][1]}")
            if "dim" in step: sequence.append(f"placeholder={step['dim']}")
            sequence.append("CURVE_END__Circle")
        elif op == "sketch_rectangle":
            sequence.append("CURVE_START__Rectangle")
            if "center_xy" in step and isinstance(step["center_xy"], list) and len(step["center_xy"]) == 2:
                sequence.append(f"center_x={step['center_xy'][0]}")
                sequence.append(f"center_y={step['center_xy'][1]}")
            if "dim" in step: sequence.append(f"placeholder={step['dim']}")
            sequence.append("CURVE_END__Rectangle")
        elif op == "sketch_line":
            sequence.append("CURVE_START__Line")
            if "center_xy" in step and isinstance(step["center_xy"], list) and len(step["center_xy"]) == 2:
                sequence.append(f"center_x={step['center_xy'][0]}")
                sequence.append(f"center_y={step['center_xy'][1]}")
            if "dim" in step: sequence.append(f"placeholder={step['dim']}")
            sequence.append("CURVE_END__Line")
        elif op == "start_extrude":
            sequence.append("ENTITY_START__Extrude")
        elif op == "extrude":
            sequence.append("operation_type=NewBody")
            if "dim" in step: sequence.append(f"placeholder_distance={step['dim']}")
        elif op == "extrude_cut":
            sequence.append("operation_type=Cut")
            if "dim" in step: sequence.append(f"placeholder_distance={step['dim']}")
        elif op == "end_extrude":
            sequence.append("ENTITY_END__Extrude")
        elif op == "start_fillet":
            sequence.append("ENTITY_START__Fillet")
        elif op == "fillet_all_edges":
            sequence.append("fillet_type=all_edges")
            if "dim" in step: sequence.append(f"placeholder_radius={step['dim']}")
        elif op == "fillet_all_vertical_corner":
            sequence.append("fillet_type=all_vertical")
            if "dim" in step: sequence.append(f"placeholder_radius={step['dim']}")
        elif op == "fillet_all_vertical_edges":
            sequence.append("fillet_type=all_vertical")
            if "dim" in step: sequence.append(f"placeholder_radius={step['dim']}")
        elif op == "fillet_all_horizontal_edges":
            sequence.append("fillet_type=all_horizontal")
            if "dim" in step: sequence.append(f"placeholder_radius={step['dim']}")
        elif op == "fillet_all_top_corner":
            sequence.append("fillet_type=all_top")
            if "dim" in step: sequence.append(f"placeholder_radius={step['dim']}")
        elif op == "fillet_all_bottom_corner":
            sequence.append("fillet_type=all_bottom")
            if "dim" in step: sequence.append(f"placeholder_radius={step['dim']}")
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


class MultiViewCADDataset(Dataset):
    def __init__(self, data_root_dir: str, json_dir: str, file_ids: List[str], vocab: Vocabulary, transform: Any,
                 num_views: int):
        self.data_root_dir, self.json_dir = data_root_dir, json_dir
        self.vocab, self.transform, self.num_views = vocab, transform, num_views
        self.view_names = ["top.png", "bottom.png", "left.png", "right.png", "front.png", "back.png"][:num_views]
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


# Model_Definition
class MVCNN_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        cnn_base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.view_feature_extractor = nn.Sequential(*list(cnn_base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # The raw feature dimension from ResNet34 before the FC layer is 512
        self.feature_dim = cnn_base.fc.in_features

    def forward(self, multi_view_images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views = multi_view_images.size(0), multi_view_images.size(1)
        reshaped_images = multi_view_images.view(-1, *multi_view_images.shape[2:])
        view_features = self.view_feature_extractor(reshaped_images)
        pooled_view_features = self.pool(view_features).view(batch_size, num_views, -1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.dropout(x + self.pe[:, :x.size(1)])


class CADTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # ---------- NEW ----------
        self.token_type_embed = nn.Embedding(3, embed_dim)  # 0=global 1=in_sketch 2=in_extrude
        # -------------------------

        self.pos_encoder = PositionalEncoding(embed_dim, dropout, config.MAX_SEQ_LENGTH)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=ff_dim, dropout=dropout,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def _token_type(tok: str) -> int:
        if tok.startswith(("CURVE_", "center_", "placeholder=camera_1", "ENTITY_START__Sketch", "ENTITY_END__Sketch")):
            return 1
        if tok.startswith(("operation_type=", "placeholder_distance=", "fillet_type=", "ENTITY_START__Extrude",
                           "ENTITY_END__Extrude",
                           "ENTITY_START__Fillet", "ENTITY_END__Fillet")):
            return 2
        return 0
    # In CADTransformerDecoder
    def forward(self, memory, trg, tgt_mask, tgt_key_padding_mask):
        B, seq_len = trg.size()

        def _token_type(tok: str) -> int:
            if tok.startswith(("CURVE_", "center_", "placeholder=camera_1",
                               "ENTITY_START__Sketch", "ENTITY_END__Sketch")):
                return 1
            if tok.startswith(("operation_type=", "placeholder_distance=", "fillet_type=",
                               "ENTITY_START__Extrude", "ENTITY_END__Extrude",
                               "ENTITY_START__Fillet", "ENTITY_END__Fillet")):
                return 2
            return 0

        type_ids = torch.zeros(B, seq_len, dtype=torch.long, device=trg.device)
        for b in range(B):
            for pos in range(seq_len):
                tok = config.vocab.itos.get(trg[b, pos].item(), config.PAD_TOKEN)
                type_ids[b, pos] = _token_type(tok)

        type_emb = self.token_type_embed(type_ids)
        tok_emb  = self.embedding(trg) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(tok_emb + type_emb)

        out = self.transformer_decoder(tgt=x, memory=memory,
                                       tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(out)


class MultiViewImageToCADModel(nn.Module):
    def __init__(self, encoder, decoder, img_feature_dim, transformer_dim, device):
        super().__init__()
        self.mvcnn_encoder, self.transformer_decoder, self.device = encoder, decoder, device
        self.projection = nn.Linear(img_feature_dim, transformer_dim)

    def make_tgt_mask(self, sz: int) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(sz).to(self.device)

    def forward(self, images, sequences):
        encoder_output = self.mvcnn_encoder(images)
        memory = self.projection(encoder_output)
        decoder_in = sequences[:, :-1]
        tgt_pad_mask = (decoder_in == config.vocab.stoi[config.PAD_TOKEN])
        tgt_mask = self.make_tgt_mask(decoder_in.size(1))
        output = self.transformer_decoder(memory, decoder_in, tgt_mask, tgt_pad_mask)
        return output

    def generate_sequence_beam_search(self, images, vocab, beam_width=5, max_len=config.MAX_SEQ_LENGTH):
        self.eval()
        sos_idx = vocab.stoi[config.SOS_TOKEN]
        eos_idx = vocab.stoi[config.EOS_TOKEN]
        pad_idx = vocab.stoi[config.PAD_TOKEN]

        with torch.no_grad():
            encoder_output = self.mvcnn_encoder(images)
            memory = self.projection(encoder_output)
            initial_beam = (torch.LongTensor([[sos_idx]]).to(self.device), 0.0)
            beams = [initial_beam]
            completed_beams = []

            for _ in range(max_len - 1):
                new_beams = []
                all_candidates = []

                for seq_tensor, score in beams:
                    if seq_tensor[0, -1].item() == eos_idx:
                        completed_beams.append((seq_tensor, score))
                        continue


                    tgt_mask = self.make_tgt_mask(seq_tensor.size(1))
                    # output, _ = self.transformer_decoder(memory, seq_tensor, tgt_mask, None)
                    output = self.transformer_decoder(memory, seq_tensor, tgt_mask, None)

                    next_token_logits = output[0, -1, :]  # Shape: (vocab_size,)
                    next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

                    top_log_probs, top_indices = torch.topk(next_token_log_probs, beam_width)


                    for i in range(beam_width):
                        next_token_idx = top_indices[i].item()
                        next_token_log_prob = top_log_probs[i].item()
                        new_seq_tensor = torch.cat([seq_tensor, torch.LongTensor([[next_token_idx]]).to(self.device)],dim=1)
                        new_score = score + next_token_log_prob
                        all_candidates.append((new_seq_tensor, new_score))

                # sorted_candidates = sorted(all_candidates, key=lambda x: x[1] / x[0].size(1), reverse=True) # Example length normalization
                sorted_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
                beams = sorted_candidates[:beam_width]
                if not beams:
                    break
            completed_beams.extend(beams)
            # completed_beams.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
            completed_beams.sort(key=lambda x: x[1], reverse=True)
            best_seq_tensor = completed_beams[0][0].squeeze(0)  # Remove batch dim
            tokens = [vocab.itos.get(idx.item(), "?") for idx in best_seq_tensor]
            final_tokens = []
            for token in tokens:
                if token == config.SOS_TOKEN:
                    continue
                final_tokens.append(token)
                if token == config.EOS_TOKEN:
                    break
            return final_tokens


# Helper_Functions ---
def visualize_attention(tokens, attention_weights, view_names):
    if not attention_weights:
        print("No attention weights to visualize.")
        return
    # Stack along the sequence length dimension and average across heads
    attn_matrix = torch.stack(attention_weights).squeeze(2).mean(dim=1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, max(6, len(tokens) // 2)))
    im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')

    ax.set_xticks(np.arange(len(view_names)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(view_names)
    ax.set_yticklabels(tokens)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax, orientation='vertical', label="Attention Weight")
    ax.set_title("Decoder Cross-Attention")
    plt.xlabel("Input Image View")
    plt.ylabel("Generated Token")
    plt.tight_layout()
    plt.show()


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


# Main_Execution_Logic
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
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = MultiViewCADDataset(config.DATA_ROOT_DIR, config.JSON_DIR, train_ids, vocab, train_transform,
                                        config.NUM_VIEWS)
    val_dataset = MultiViewCADDataset(config.DATA_ROOT_DIR, config.JSON_DIR, val_ids, vocab, val_transform,
                                      config.NUM_VIEWS)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    encoder = MVCNN_Encoder()
    decoder = CADTransformerDecoder(config.VOCAB_SIZE, config.TRANSFORMER_EMBED_DIM, config.NUM_HEADS,
                                    config.NUM_DECODER_LAYERS, config.TRANSFORMER_FF_DIM, config.DROPOUT_RATE)

    # Pass the correct feature dimension to the main model
    model = MultiViewImageToCADModel(
        encoder,
        decoder,
        img_feature_dim=encoder.feature_dim,
        transformer_dim=config.TRANSFORMER_EMBED_DIM,
        device=config.DEVICE
    ).to(config.DEVICE)

    weights = torch.ones(config.VOCAB_SIZE).to(config.DEVICE)
    weights[vocab.stoi[config.EOS_TOKEN]] = 8.0
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi[config.PAD_TOKEN], weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


    train_hist, val_hist, best_val_loss = [], [], float('inf')
    epochs_no_improve = 0
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step()
        train_hist.append(train_loss);
        val_hist.append(val_loss)
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        if len(val_dataset) > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"  New best model saved to {config.MODEL_SAVE_PATH}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {config.EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break
    print("\nTraining Finished.")
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

    encoder = MVCNN_Encoder()
    decoder = CADTransformerDecoder(config.VOCAB_SIZE, config.TRANSFORMER_EMBED_DIM, config.NUM_HEADS,
                                    config.NUM_DECODER_LAYERS, config.TRANSFORMER_FF_DIM, config.DROPOUT_RATE)
    model = MultiViewImageToCADModel(
        encoder,
        decoder,
        img_feature_dim=encoder.feature_dim,
        transformer_dim=config.TRANSFORMER_EMBED_DIM,
        device=config.DEVICE
    ).to(config.DEVICE)

    print(f"Loading trained weights from {config.MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    view_names = ["top.png", "bottom.png", "left.png", "right.png", "front.png", "back.png"][:config.NUM_VIEWS]

    input_tensor = load_multi_view_images(args.test_dir, transform, view_names).to(config.DEVICE)
    if input_tensor is not None:
        generated_tokens = model.generate_sequence_beam_search(input_tensor, vocab, beam_width=5)
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