# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
YOLO format dataset loader for CoOp training with SAM3.

Supports the standard YOLO annotation format:
- images/{split}/*.jpg
- labels/{split}/*.txt (class_id cx cy w h, normalized)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class YOLOCoOpDataset(Dataset):
    """
    Dataset for loading YOLO-format annotations for CoOp training.
    
    Directory structure expected:
        data_dir/
            classes.txt
            images/
                TRAIN/
                VAL/
                TEST/
            labels/
                TRAIN/
                VAL/
                TEST/
    
    Args:
        data_dir: Root directory containing images/ and labels/
        split: Data split - "TRAIN", "VAL", or "TEST"
        resolution: Target image resolution
        include_negatives: Whether to include images with no annotations
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "TRAIN",
        resolution: int = 1008,
        include_negatives: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.resolution = resolution
        self.include_negatives = include_negatives
        
        self.image_dir = self.data_dir / "images" / split
        self.label_dir = self.data_dir / "labels" / split
        
        # Load class names
        classes_file = self.data_dir / "classes.txt"
        if classes_file.exists():
            self.classes = classes_file.read_text().strip().split("\n")
        else:
            self.classes = ["object"]  # Default
        
        # Find all images
        self.images = self._find_images()
        
        # Image transforms
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        print(f"Loaded {len(self.images)} images from {split} split")
        print(f"Classes: {self.classes}")
    
    def _find_images(self) -> List[Path]:
        """Find all valid image files."""
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        images = []
        
        for ext in extensions:
            images.extend(self.image_dir.glob(f"*{ext}"))
            images.extend(self.image_dir.glob(f"*{ext.upper()}"))
        
        if not self.include_negatives:
            # Filter to only images with non-empty labels
            valid_images = []
            for img_path in images:
                label_path = self.label_dir / f"{img_path.stem}.txt"
                if label_path.exists() and label_path.stat().st_size > 0:
                    valid_images.append(img_path)
            images = valid_images
        
        return sorted(images)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Parse YOLO labels
        boxes, labels = self._parse_yolo_label(label_path)
        
        # Convert from YOLO format (cx, cy, w, h) to (x1, y1, x2, y2) for SAM3
        if len(boxes) > 0:
            boxes_xyxy = self._yolo_to_xyxy(boxes)
        else:
            boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
        
        return {
            "image": image_tensor,
            "boxes": boxes,  # YOLO format (cx, cy, w, h) normalized
            "boxes_xyxy": boxes_xyxy,  # (x1, y1, x2, y2) normalized
            "labels": labels,
            "image_path": str(img_path),
            "orig_size": (orig_h, orig_w),
        }
    
    def _parse_yolo_label(self, label_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse YOLO format label file.
        
        Format: class_id cx cy w h (all normalized 0-1)
        """
        boxes = []
        labels = []
        
        if not label_path.exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        
        content = label_path.read_text().strip()
        if not content:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        
        for line in content.split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([cx, cy, w, h])
                labels.append(class_id)
        
        if boxes:
            return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    
    def _yolo_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert YOLO format (cx, cy, w, h) to (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)


def collate_coop_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for CoOp dataset.
    
    Handles variable number of boxes per image by padding.
    """
    images = torch.stack([item["image"] for item in batch])
    
    # Find max number of boxes
    max_boxes = max(item["boxes"].size(0) for item in batch)
    max_boxes = max(max_boxes, 1)  # At least 1 box slot
    
    batch_size = len(batch)
    
    # Pad boxes and labels
    boxes_padded = torch.zeros(batch_size, max_boxes, 4)
    boxes_xyxy_padded = torch.zeros(batch_size, max_boxes, 4)
    labels_padded = torch.zeros(batch_size, max_boxes, dtype=torch.long)
    num_boxes = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        n = item["boxes"].size(0)
        if n > 0:
            boxes_padded[i, :n] = item["boxes"]
            boxes_xyxy_padded[i, :n] = item["boxes_xyxy"]
            labels_padded[i, :n] = item["labels"]
        num_boxes[i] = n
    
    return {
        "images": images,
        "boxes": boxes_padded,  # [B, max_boxes, 4] - YOLO format
        "boxes_xyxy": boxes_xyxy_padded,  # [B, max_boxes, 4]
        "labels": labels_padded,  # [B, max_boxes]
        "num_boxes": num_boxes,  # [B]
        "image_paths": [item["image_path"] for item in batch],
        "orig_sizes": [item["orig_size"] for item in batch],
    }
