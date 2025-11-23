# -*- coding: utf-8 -*-
"""캡스톤_시연용_추론_item_seq반환"""

import os
import json
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from ultralytics import YOLO
from PIL import Image

# -------------------------------------------------
# 0) 경로 / 기본 설정
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BEST_YOLO      = os.path.join(BASE_DIR, "best (2).pt")
RESNET_1324_PT = os.path.join(BASE_DIR, "best_model_generalized_fin.pth")

CLASS_JSON_1K  = os.path.join(BASE_DIR, "pill_label_path_sharp_score.json")
CLASS_JSON_324 = os.path.join(BASE_DIR, "class_mapping_from_cache_1324 (2).json")

KCODE_ITEMSEQ_CACHE = os.path.join(BASE_DIR, "kcode_itemseq_cache_fast_1324.json")

YOLO_CONF  = 0.25
YOLO_IOU   = 0.45
YOLO_IMGSZ = 640

CROP_SIZE        = 224
MIN_BOX_SIDE_PX  = 40
SQUARE_SCALE     = 1.3

NUM_CLASSES  = 1324
LABEL_OFFSET = 1000
MAX_PILLS_MULTI = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# -------------------------------------------------
# 1) class_idx → K-code, 그리고 K-code → item_seq
# -------------------------------------------------
def load_label_map_generic(json_path):
    if not json_path or not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "label_to_kcode" in data:
        data = data["label_to_kcode"]

    out = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                key_int = int(k)
            except:
                continue

            val_str = str(v)
            base = os.path.basename(val_str)
            first = base.split("_")[0]

            if first.startswith("K-") and len(first) >= 3:
                kcode = first
            elif first.startswith("K") and len(first) == 7 and first[1:].isdigit():
                kcode = "K-" + first[1:]
            else:
                kcode = first

            out[key_int] = kcode

    return out


LABEL_MAP_1K  = load_label_map_generic(CLASS_JSON_1K)
LABEL_MAP_324 = load_label_map_generic(CLASS_JSON_324)


def class_idx_to_kcode(global_idx: int):
    if global_idx < LABEL_OFFSET:
        return LABEL_MAP_1K.get(global_idx, None)
    local = global_idx - LABEL_OFFSET
    return LABEL_MAP_324.get(local, None)


with open(KCODE_ITEMSEQ_CACHE, "r", encoding="utf-8") as f:
    _item_cache = json.load(f)

KCODE_TO_ITEMSEQ = _item_cache.get("kcode_to_item_seq", {})


def kcode_to_item_seq(kcode: str):
    if kcode is None:
        return None
    return KCODE_TO_ITEMSEQ.get(kcode)


# -------------------------------------------------
# 2) ResNet 1324 + 전처리
# -------------------------------------------------
def build_resnet_1324(num_classes=NUM_CLASSES, model_path=RESNET_1324_PT):
    model = models.resnet152(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_f, num_classes)
    )

    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "model" in state:
            state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"ℹ️ state_dict load: missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(DEVICE)
    model.eval()

    if DEVICE.type == "cuda":
        model.half()

    return model


RESNET_MODEL = build_resnet_1324()

base_transform = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


def preprocess_pil(pil_img):
    t = base_transform(pil_img)
    if DEVICE.type == "cuda":
        return t.half()
    return t


@torch.no_grad()
def predict_resnet_batch_top1(pil_imgs):
    if not pil_imgs:
        return []
    xs = [preprocess_pil(im) for im in pil_imgs]
    x = torch.stack(xs).to(DEVICE)

    logits = RESNET_MODEL(x)
    probs = F.softmax(logits, dim=1)
    top1_prob, top1_idx = torch.topk(probs, 1, dim=1)

    all_results = []
    for i in range(probs.shape[0]):
        all_results.append({
            "idx": int(top1_idx[i, 0].item()),
            "prob": float(top1_prob[i, 0].item()),
        })
    return all_results


# -------------------------------------------------
# 3) YOLO 감지
# -------------------------------------------------
YOLO_DEVICE = 0 if DEVICE.type == "cuda" else "CPU"
YOLO_MODEL = YOLO(BEST_YOLO)


def detect_yolo_boxes(img_path):
    det = YOLO_MODEL(
        img_path,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        device=YOLO_DEVICE,
        verbose=False
    )[0]

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    boxes = []
    confs = []
    if det.boxes is not None and len(det.boxes) > 0:
        xyxy = det.boxes.xyxy.cpu().numpy().tolist()
        conf = det.boxes.conf.cpu().numpy().tolist()
        for (b, c) in zip(xyxy, conf):
            x1, y1, x2, y2 = map(int, b)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W-1, x2)
            y2 = min(H-1, y2)
            if (x2-x1) >= MIN_BOX_SIDE_PX and (y2-y1) >= MIN_BOX_SIDE_PX:
                boxes.append([x1, y1, x2, y2])
                confs.append(float(c))

    return img, boxes, confs


def square_crop_from_bbox(pil_img, xyxy, scale=SQUARE_SCALE):
    W, H = pil_img.size
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh) * scale
    side = max(side, MIN_BOX_SIDE_PX * 1.5)

    half = side / 2.0
    nx1 = int(round(cx - half))
    ny1 = int(round(cy - half))
    nx2 = int(round(cx + half))
    ny2 = int(round(cy + half))

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(W, nx2)
    ny2 = min(H, ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        return None

    return pil_img.crop((nx1, ny1, nx2, ny2))


# -------------------------------------------------
# 4) 다중 알약 inference  (FastAPI에서 호출되는 함수)
# -------------------------------------------------
def infer_pill_image_multi_top1(img_path: str,
                                max_pills: int = MAX_PILLS_MULTI):

    assert os.path.exists(img_path), f"이미지 없음: {img_path}"

    img, boxes, confs = detect_yolo_boxes(img_path)
    results = []

    if not boxes:
        crops = [img.copy()]
        preds = predict_resnet_batch_top1(crops)
        if preds:
            p = preds[0]
            idx = p["idx"]
            prob = p["prob"]
            kcode = class_idx_to_kcode(idx)
            item_seq = kcode_to_item_seq(kcode)
            results.append({
                "pill_index": 0,
                "bbox": None,
                "class_idx": idx,
                "kcode": kcode,
                "item_seq": item_seq,
                "prob": prob,
            })
        return results

    idx_sorted = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    idx_keep   = idx_sorted[:max_pills]

    crops = []
    kept_boxes = []
    for i in idx_keep:
        bbox = boxes[i]
        sq = square_crop_from_bbox(img, bbox, scale=SQUARE_SCALE)
        if sq is None:
            continue
        crops.append(sq)
        kept_boxes.append(bbox)

    if not crops:
        crops = [img.copy()]
        kept_boxes = [None]

    preds = predict_resnet_batch_top1(crops)

    for pill_idx, (bbox, p) in enumerate(zip(kept_boxes, preds)):
        idx = p["idx"]
        prob = p["prob"]
        kcode = class_idx_to_kcode(idx)
        item_seq = kcode_to_item_seq(kcode)
        results.append({
            "pill_index": pill_idx,
            "bbox": bbox,
            "class_idx": idx,
            "kcode": kcode,
            "item_seq": item_seq,
            "prob": prob,
        })

    return results
