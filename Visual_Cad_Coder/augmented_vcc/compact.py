# train_multiview_mask2former.py
#
# Offline-friendly version:
# - NO wandb dependency
# - writes metrics to files in --out_dir:
#     split.json
#     type_to_id.json
#     train_metrics.jsonl        (one JSON per optimizer step)
#     eval_metrics.jsonl         (one JSON per evaluation)
#     final_metrics.json         (summary)
#     checkpoints/...
#     final/...
#
# Usage:
#   python train_multiview_mask2former.py --data_root /path/to/data --out_dir runs/exp1
#
# Notes:
# - Expects per object folder:
#     rgb_{view}.png   where view in: bottom, top, left, right, front, back, iso1, iso2
#     mask_{view}.png  colored per-face instance mask (train/test only; not used at inference)
#     labels.json (or any .json in the folder) describing face->type mapping (adapt parse_json_face_types)
#
# - Evaluation metric is lightweight: mean best IoU per GT instance (class-matched).

import os
import json
import argparse
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from transformers import Mask2FormerForUniversalSegmentation


# ---------------------------
# CONFIG
# ---------------------------

VIEWS = ["bottom", "top", "left", "right", "front", "back", "iso1", "iso2"]
RGB_PATTERN = "rgb_{view}.png"
MASK_PATTERN = "mask_{view}.png"

IGNORE_INDEX = 255


# ---------------------------
# IO HELPERS
# ---------------------------

def write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# ---------------------------
# JSON PARSING (ADAPT THIS)
# ---------------------------

def find_json(obj_dir: str) -> Optional[str]:
    candidates = [f for f in os.listdir(obj_dir) if f.lower().endswith(".json")]
    if not candidates:
        return None
    return os.path.join(obj_dir, sorted(candidates)[0])

def parse_json_face_types(json_path: str) -> Dict[int, str]:
    """
    Returns dict: face_index(int) -> type_name(str)

    Adapt to your JSON schema if needed.
    Draft schema:
      labels["body"]["faces"] list
      face["feature"] -> feature id
      labels["features"][str(feature_id)] -> feature_name/type_name
    """
    with open(json_path, "r") as f:
        labels = json.load(f)

    faces = labels["body"]["faces"]
    feature_map = labels["features"]
    face_to_type = {}
    for i, face in enumerate(faces):
        feat_id = face["feature"]
        face_to_type[i] = feature_map[feat_id]["name"] 
    return face_to_type

def build_type_vocab(dataset_root: str) -> Dict[str, int]:
    type_set = set()
    for obj in sorted(os.listdir(dataset_root)):
        obj_dir = os.path.join(dataset_root, obj)
        if not os.path.isdir(obj_dir):
            continue
        jp = find_json(obj_dir)
        if jp is None:
            continue
        for _, t in parse_json_face_types(jp).items():
            type_set.add(t)
    type_list = sorted(type_set)
    return {t: i for i, t in enumerate(type_list)}


# ---------------------------
# MASK UTILITIES
# ---------------------------

def pil_to_np_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"), dtype=np.uint8)

def rgb_to_int(mask_rgb: np.ndarray) -> np.ndarray:
    return (mask_rgb[..., 0].astype(np.int32) << 16) | (mask_rgb[..., 1].astype(np.int32) << 8) | mask_rgb[..., 2].astype(np.int32)

def unique_colors(mask_rgb: np.ndarray) -> np.ndarray:
    flat = mask_rgb.reshape(-1, 3)
    return np.unique(flat, axis=0)

def build_color_to_face_index(mask_rgb: np.ndarray, background_rgb=(0, 0, 0)) -> Dict[int, int]:
    """
    Heuristic mapping color->face index: sort unique colors by packed int.
    Works if your face index ordering aligns with this scheme.
    If not, you need an explicit color->face_id mapping from your exporter.
    """
    uniq = unique_colors(mask_rgb)
    uniq = np.array([c for c in uniq.tolist() if tuple(c) != tuple(background_rgb)], dtype=np.uint8)
    if len(uniq) == 0:
        return {}
    uniq_int = (uniq[:, 0].astype(np.int32) << 16) | (uniq[:, 1].astype(np.int32) << 8) | uniq[:, 2].astype(np.int32)
    order = np.argsort(uniq_int)
    uniq_int_sorted = uniq_int[order]
    return {int(ci): int(i) for i, ci in enumerate(uniq_int_sorted.tolist())}

def colored_mask_to_instance_targets(
    mask_rgb: np.ndarray,
    face_index_to_type: Dict[int, str],
    type_to_id: Dict[str, int],
    image_size: Tuple[int, int],
    background_rgb=(0, 0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    pil = Image.fromarray(mask_rgb, mode="RGB")
    pil = pil.resize((image_size[1], image_size[0]), resample=Image.NEAREST)  # PIL: (W,H)
    mask_rgb = np.array(pil, dtype=np.uint8)

    mask_int = rgb_to_int(mask_rgb)
    color_to_face_index = build_color_to_face_index(mask_rgb, background_rgb=background_rgb)

    masks = []
    classes = []
    for col_int, face_idx in color_to_face_index.items():
        if face_idx not in face_index_to_type:
            continue
        tname = face_index_to_type[face_idx]
        if tname not in type_to_id:
            continue
        cls = type_to_id[tname]
        m = (mask_int == col_int)
        if m.sum() == 0:
            continue
        masks.append(torch.from_numpy(m.astype(np.float32)))
        classes.append(int(cls))

    if len(masks) == 0:
        H, W = mask_int.shape
        return torch.zeros((0, H, W), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    return torch.stack(masks, dim=0), torch.tensor(classes, dtype=torch.int64)


# ---------------------------
# DATASET
# ---------------------------

class MultiViewRGBWithColorMaskDataset(Dataset):
    def __init__(
        self,
        root: str,
        object_ids: List[str],
        type_to_id: Dict[str, int],
        image_size: int = 512,
        background_rgb=(0, 0, 0),
    ):
        self.root = root
        self.object_ids = object_ids
        self.type_to_id = type_to_id
        self.image_size = image_size
        self.background_rgb = background_rgb

        self.img_tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.object_ids[idx]
        obj_dir = os.path.join(self.root, obj)

        jp = find_json(obj_dir)
        if jp is None:
            raise FileNotFoundError(f"No json found in {obj_dir}")
        face_index_to_type = parse_json_face_types(jp)

        images = []
        mask_labels_per_view = []
        class_labels_per_view = []

        for v in VIEWS:
            rgb_path = os.path.join(obj_dir, RGB_PATTERN.format(view=v))
            mask_path = os.path.join(obj_dir, MASK_PATTERN.format(view=v))

            img = Image.open(rgb_path).convert("RGB")
            images.append(self.img_tf(img))

            mask_rgb = pil_to_np_rgb(Image.open(mask_path))
            mask_labels, class_labels = colored_mask_to_instance_targets(
                mask_rgb=mask_rgb,
                face_index_to_type=face_index_to_type,
                type_to_id=self.type_to_id,
                image_size=(self.image_size, self.image_size),
                background_rgb=self.background_rgb,
            )
            mask_labels_per_view.append(mask_labels)
            class_labels_per_view.append(class_labels)

        return {
            "pixel_values": torch.stack(images, dim=0),  # (V,3,H,W)
            "mask_labels": mask_labels_per_view,         # list(V) of (Ni,H,W)
            "class_labels": class_labels_per_view,       # list(V) of (Ni,)
            "object_id": obj,
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)  # (B,V,3,H,W)
    mask_labels = [b["mask_labels"] for b in batch]       # list(B) of list(V)
    class_labels = [b["class_labels"] for b in batch]
    object_ids = [b["object_id"] for b in batch]
    return {"pixel_values": pixel_values, "mask_labels": mask_labels, "class_labels": class_labels, "object_id": object_ids}


# ---------------------------
# SPLIT
# ---------------------------

def train_test_split_objects(root: str, test_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    objs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    rng = np.random.RandomState(seed)
    rng.shuffle(objs)
    n_test = int(round(len(objs) * test_ratio))
    test_ids = objs[:n_test]
    train_ids = objs[n_test:]
    return train_ids, test_ids


# ---------------------------
# METRICS (simple)
# ---------------------------

def iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (a * b).sum().item()
    union = ((a + b) > 0).float().sum().item()
    return float(inter / (union + eps))

@torch.no_grad()
def eval_on_loader(
    model: Mask2FormerForUniversalSegmentation,
    loader: DataLoader,
    device: str,
    mask_threshold: float = 0.5,
    max_queries: int = 50,
) -> Dict[str, float]:
    model.eval()
    total_iou = 0.0
    total_instances = 0
    total_views = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)  # (B,V,3,H,W)
        B, V, C, H, W = pixel_values.shape
        pv = pixel_values.view(B * V, C, H, W)

        out = model(pixel_values=pv)    #CHANGED on cluster
        pred_masks = out.pred_masks.sigmoid()             # (B*V,Q,H,W)
        pred_probs = out.pred_logits.softmax(-1)          # (B*V,Q,classes+1)

        gt_masks_list = []
        gt_classes_list = []
        for b in range(B):
            for v in range(V):
                gt_masks_list.append(batch["mask_labels"][b][v])   # (Ni,H,W)
                gt_classes_list.append(batch["class_labels"][b][v])

        for i in range(B * V):
            total_views += 1
            pm = (pred_masks[i] > mask_threshold).float()  # (Q,H,W)
            pp = pred_probs[i]                              # (Q,K+1)

            conf = pp[:, :-1].max(dim=-1).values
            top_idx = torch.argsort(conf, descending=True)[:max_queries]
            pm = pm[top_idx].cpu()
            pp = pp[top_idx].cpu()
            pred_cls = pp[:, :-1].argmax(dim=-1)  # (Q,)

            gt_masks = gt_masks_list[i]
            gt_classes = gt_classes_list[i]

            if gt_masks.numel() == 0 or gt_masks.shape[0] == 0:
                continue

            for j in range(gt_masks.shape[0]):
                gmask = gt_masks[j].float()
                gcls = int(gt_classes[j].item())
                cand = (pred_cls == gcls).nonzero(as_tuple=False).view(-1)
                if cand.numel() == 0:
                    total_instances += 1
                    continue
                best = 0.0
                for c in cand.tolist():
                    best = max(best, iou(gmask, pm[c]))
                total_iou += best
                total_instances += 1

    mean_iou = total_iou / max(1, total_instances)
    return {
        "mean_iou_per_gt_instance": mean_iou,
        "gt_instances": float(total_instances),
        "views_evaluated": float(total_views),
    }


# ---------------------------
# TRAIN
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/exp") #CHANGED on cluster
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--background_rgb", type=int, nargs=3, default=(0, 0, 0))
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_root = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    train_jsonl = os.path.join(args.out_dir, "train_metrics.jsonl")
    eval_jsonl = os.path.join(args.out_dir, "eval_metrics.jsonl")
    # reset metric files if they exist
    if os.path.exists(train_jsonl):
        os.remove(train_jsonl)
    if os.path.exists(eval_jsonl):
        os.remove(eval_jsonl)

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Split
    train_ids, test_ids = train_test_split_objects(args.data_root, args.test_ratio, args.seed)
    write_json(os.path.join(args.out_dir, "split.json"), {"train": train_ids, "test": test_ids})

    # Vocab
    type_to_id = build_type_vocab(args.data_root)
    num_classes = len(type_to_id)
    write_json(os.path.join(args.out_dir, "type_to_id.json"), type_to_id)

    # Data
    train_ds = MultiViewRGBWithColorMaskDataset(
        args.data_root, train_ids, type_to_id, image_size=args.image_size, background_rgb=tuple(args.background_rgb)
    )
    test_ds = MultiViewRGBWithColorMaskDataset(
        args.data_root, test_ids, type_to_id, image_size=args.image_size, background_rgb=tuple(args.background_rgb)
    )

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True, collate_fn=collate_fn
    )

    # Model
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-coco-instance",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(args.device)   #CHANGED on cluster

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train
    global_step = 0
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_dl):
            step_t0 = time.time()

            pixel_values = batch["pixel_values"].to(args.device)  # (B,V,3,H,W)
            B, V, C, H, W = pixel_values.shape
            pv = pixel_values.view(B * V, C, H, W)

            # Flatten targets to match B*V
            mask_labels = []
            class_labels = []
            for b in range(B):
                for v in range(V):
                    mask_labels.append(batch["mask_labels"][b][v].to(args.device))
                    class_labels.append(batch["class_labels"][b][v].to(args.device))

            out = model(pixel_values=pv, mask_labels=mask_labels, class_labels=class_labels)
            loss = out.loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            lr_now = optim.param_groups[0]["lr"]
            rec = {
                "step": global_step,
                "epoch": ep,
                "loss": float(loss.item()),
                "lr": float(lr_now),
                "batch_size": int(args.batch_size),
                "views": int(V),
                "sec_per_step": float(time.time() - step_t0),
                "sec_total": float(time.time() - t0),
            }
            append_jsonl(train_jsonl, rec)

            if global_step % 20 == 0:
                print(f"[train] ep={ep} step={global_step} loss={rec['loss']:.4f} lr={lr_now:.2e}")

            global_step += 1

        # Eval
        if (ep + 1) % args.eval_every == 0:
            eval_t0 = time.time()
            metrics = eval_on_loader(model, test_dl, device=args.device)
            rec = {
                "epoch": ep,
                "step": global_step,
                "sec_eval": float(time.time() - eval_t0),
                "sec_total": float(time.time() - t0),
                **{f"test/{k}": v for k, v in metrics.items()},
            }
            append_jsonl(eval_jsonl, rec)
            print(f"[eval] ep={ep} mean_iou={rec['test/mean_iou_per_gt_instance']:.4f} "
                  f"gt_inst={int(rec['test/gt_instances'])} views={int(rec['test/views_evaluated'])}")

        # Save
        if (ep + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(ckpt_root, f"epoch{ep}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)

    # Final save
    final_dir = os.path.join(args.out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)

    # Final eval summary
    final_metrics = eval_on_loader(model, test_dl, device=args.device)
    summary = {
        "num_classes": int(num_classes),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "image_size": int(args.image_size),
        "test_ratio": float(args.test_ratio),
        "seed": int(args.seed),
        "background_rgb": list(args.background_rgb),
        "train_objects": len(train_ids),
        "test_objects": len(test_ids),
        "final_checkpoint": final_dir,
        **{f"test/{k}": v for k, v in final_metrics.items()},
    }
    write_json(os.path.join(args.out_dir, "final_metrics.json"), summary)
    print("[done] wrote:", os.path.join(args.out_dir, "final_metrics.json"))


if __name__ == "__main__":
    main()