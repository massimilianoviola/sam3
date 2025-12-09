#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
CoOp Training Script for SAM3 Detector.

Trains learnable context vectors to improve detection of specific classes.
Only ~8K parameters are trained while the 848M SAM3 model stays frozen.

Uses SAM3's native detection loss:
- Hungarian matching (L1 + GIoU + focal)
- Box loss: L1 + GIoU
- Classification loss: Focal loss (binary)

Usage:
    python -m sam3.train.train_coop \
        --data_dir /home/cerrion/fallen_bottle_dataset \
        --class_name "A fallen bottle" \
        --n_ctx 8 \
        --epochs 50 \
        --batch_size 8 \
        --output_dir ./coop_checkpoints
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam3.model.sam3_coop_wrapper import build_sam3_coop_detector
from sam3.model.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from sam3.train.data.coop_yolo_dataset import YOLOCoOpDataset, collate_coop_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train CoOp prompts for SAM3")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to YOLO format dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit dataset size for testing (e.g., 5)")
    parser.add_argument("--class_name", type=str, default="A fallen bottle",
                        help="Class name for the prompt")
    
    # CoOp arguments
    parser.add_argument("--n_ctx", type=int, default=8,
                        help="Number of learnable context tokens")
    parser.add_argument("--ctx_init", type=str, default=None,
                        help="Optional text to initialize context")
    parser.add_argument("--class_token_position", type=str, default="end",
                        choices=["end", "front", "middle"],
                        help="Position of class token in prompt")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Learning rate (CoOp paper default: 0.002)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=1,
                        help="Number of warmup epochs")
    
    # Loss weights (SAM3 defaults)
    parser.add_argument("--loss_bbox", type=float, default=5.0,
                        help="L1 box loss weight")
    parser.add_argument("--loss_giou", type=float, default=2.0,
                        help="GIoU loss weight")
    parser.add_argument("--loss_cls", type=float, default=1.0,
                        help="Classification loss weight")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./coop_checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_freq", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--eval_freq", type=int, default=5,
                        help="Run validation every N epochs")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # SAM3 checkpoint
    parser.add_argument("--sam3_checkpoint", type=str, default=None,
                        help="Optional path to SAM3 checkpoint (defaults to HuggingFace)")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to CoOp checkpoint to resume training from")
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="Starting epoch (use with --resume)")
    
    return parser.parse_args()


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Sigmoid Focal Loss for binary classification.
    
    Args:
        inputs: Prediction logits [N]
        targets: Binary targets [N] (0 or 1)
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
        reduction: 'mean' or 'sum' or 'none'
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * ce_loss
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class SAM3DetectionLoss(nn.Module):
    """
    SAM3-style detection loss with Hungarian matching.
    
    Components:
    - Box loss: L1 + GIoU
    - Classification loss: Focal loss
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        loss_bbox: float = 5.0,
        loss_giou: float = 2.0,
        loss_cls: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.loss_cls = loss_cls
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    @torch.no_grad()
    def hungarian_match(self, pred_boxes, pred_logits, gt_boxes, num_boxes):
        """
        Perform Hungarian matching between predictions and GT.
        
        Args:
            pred_boxes: [B, Q, 4] predicted boxes (cx, cy, w, h)
            pred_logits: [B, Q, 1] predicted logits
            gt_boxes: [B, max_gt, 4] ground truth boxes (cx, cy, w, h) 
            num_boxes: [B] number of valid GT boxes per image
            
        Returns:
            List of (pred_indices, gt_indices) per batch element
        """
        from scipy.optimize import linear_sum_assignment
        
        batch_size, num_queries = pred_boxes.shape[:2]
        device = pred_boxes.device
        
        indices = []
        for b in range(batch_size):
            n_gt = num_boxes[b].item()
            if n_gt == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=device),
                               torch.tensor([], dtype=torch.long, device=device)))
                continue
            
            # Get predictions and GT for this batch element
            pred_box = pred_boxes[b]  # [Q, 4]
            pred_score = pred_logits[b].squeeze(-1).sigmoid()  # [Q]
            gt_box = gt_boxes[b, :n_gt]  # [n_gt, 4]
            
            # Classification cost (focal-inspired)
            neg_cost = (1 - self.focal_alpha) * (pred_score ** self.focal_gamma) * (-(1 - pred_score + 1e-8).log())
            pos_cost = self.focal_alpha * ((1 - pred_score) ** self.focal_gamma) * (-(pred_score + 1e-8).log())
            cost_class = (pos_cost - neg_cost).unsqueeze(-1).expand(-1, n_gt)
            
            # L1 cost
            cost_bbox = torch.cdist(pred_box, gt_box, p=1)
            
            # GIoU cost  
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_box),
                box_cxcywh_to_xyxy(gt_box)
            )
            
            # Total cost
            C = (self.cost_class * cost_class + 
                 self.cost_bbox * cost_bbox + 
                 self.cost_giou * cost_giou)
            
            # Hungarian matching
            C_np = C.cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(C_np)
            
            indices.append((
                torch.tensor(pred_idx, dtype=torch.long, device=device),
                torch.tensor(gt_idx, dtype=torch.long, device=device)
            ))
        
        return indices
    
    def forward(self, outputs, targets):
        """
        Compute SAM3 detection loss.
        
        Args:
            outputs: Dict with 'pred_boxes' [B,Q,4] and 'pred_logits' [B,Q,1]
            targets: Dict with 'boxes' [B,max_gt,4], 'num_boxes' [B]
            
        Returns:
            Dict with loss components and total loss
        """
        pred_boxes = outputs["pred_boxes"]  # [B, Q, 4]
        pred_logits = outputs["pred_logits"]  # [B, Q, 1]
        gt_boxes = targets["boxes"]  # [B, max_gt, 4]
        num_boxes = targets["num_boxes"]  # [B]
        
        device = pred_boxes.device
        batch_size, num_queries = pred_boxes.shape[:2]
        
        # Hungarian matching
        indices = self.hungarian_match(pred_boxes, pred_logits, gt_boxes, num_boxes)
        
        # Compute losses
        total_num_boxes = num_boxes.sum().clamp(min=1).float()
        
        loss_bbox = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            n_gt = num_boxes[b].item()
            
            # Classification loss for ALL predictions (matched + unmatched)
            target_cls = torch.zeros(num_queries, device=device)
            if len(pred_idx) > 0:
                target_cls[pred_idx] = 1.0  # Matched predictions are positive
            
            loss_cls += sigmoid_focal_loss(
                pred_logits[b].squeeze(-1),
                target_cls,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum"
            )
            
            if len(pred_idx) == 0:
                continue
            
            # Box losses only for matched predictions
            matched_pred_boxes = pred_boxes[b, pred_idx]  # [n_matched, 4]
            matched_gt_boxes = gt_boxes[b, gt_idx]  # [n_matched, 4]
            
            # L1 loss
            loss_bbox += F.l1_loss(matched_pred_boxes, matched_gt_boxes, reduction="sum")
            
            # GIoU loss
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(matched_pred_boxes),
                box_cxcywh_to_xyxy(matched_gt_boxes)
            )
            loss_giou += (1 - giou.diag()).sum()
        
        # Normalize
        loss_bbox = loss_bbox / total_num_boxes
        loss_giou = loss_giou / total_num_boxes
        loss_cls = loss_cls / (batch_size * num_queries)  # Normalize by total predictions
        
        # Total weighted loss
        total_loss = (
            self.loss_bbox * loss_bbox +
            self.loss_giou * loss_giou +
            self.loss_cls * loss_cls
        )
        
        return {
            "loss": total_loss,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_cls": loss_cls,
        }


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch, warmup_epochs, base_lr, confidence_threshold=0.5):
    """Train for one epoch with detection metrics."""
    model.train()
    total_loss = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    total_cls = 0.0
    num_batches = 0
    
    # Detection metrics accumulators
    total_gt = 0
    total_pred = 0
    num_matched = 0
    sum_iou = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_factor = (epoch * len(dataloader) + batch_idx) / (warmup_epochs * len(dataloader))
            lr = base_lr * max(warmup_factor, 0.01)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        
        images = batch["images"].to(device)
        targets = {
            "boxes": batch["boxes"].to(device),
            "num_boxes": batch["num_boxes"].to(device),
        }
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, targets)
        
        # Compute SAM3 detection loss
        losses = loss_fn(outputs, targets)
        loss = losses["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_bbox += losses["loss_bbox"].item()
        total_giou += losses["loss_giou"].item()
        total_cls += losses["loss_cls"].item()
        num_batches += 1
        
        # Compute detection metrics (no grad already from optimizer.step)
        with torch.no_grad():
            pred_boxes = outputs.get("pred_boxes")
            pred_logits = outputs.get("pred_logits")
            if pred_boxes is not None and pred_logits is not None:
                batch_size = images.size(0)
                gt_boxes = targets["boxes"]
                num_boxes_t = targets["num_boxes"]
                
                for b in range(batch_size):
                    pred_box = pred_boxes[b]
                    pred_score = pred_logits[b].squeeze(-1).sigmoid()
                    keep = pred_score > confidence_threshold
                    pred_box = pred_box[keep]
                    pred_score = pred_score[keep]
                    
                    n_gt = num_boxes_t[b].item()
                    gt_box = gt_boxes[b, :n_gt]
                    
                    total_gt += n_gt
                    total_pred += len(pred_score)
                    
                    if len(pred_score) > 0 and n_gt > 0:
                        ious = box_iou_simple(pred_box, gt_box)
                        gt_matched = torch.zeros(n_gt, dtype=torch.bool, device=device)
                        for pred_idx in pred_score.argsort(descending=True):
                            best_gt_idx = ious[pred_idx].argmax().item()
                            best_iou = ious[pred_idx, best_gt_idx].item()
                            if best_iou >= 0.5 and not gt_matched[best_gt_idx]:
                                gt_matched[best_gt_idx] = True
                                num_matched += 1
                                sum_iou += best_iou
        
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "bbox": f"{losses['loss_bbox'].item():.3f}",
            "giou": f"{losses['loss_giou'].item():.3f}",
            "cls": f"{losses['loss_cls'].item():.3f}",
        })
    
    precision = num_matched / total_pred if total_pred > 0 else 0
    recall = num_matched / total_gt if total_gt > 0 else 0
    avg_iou = sum_iou / num_matched if num_matched > 0 else 0
    
    return {
        "loss": total_loss / max(num_batches, 1),
        "loss_bbox": total_bbox / max(num_batches, 1),
        "loss_giou": total_giou / max(num_batches, 1),
        "loss_cls": total_cls / max(num_batches, 1),
        "precision": precision,
        "recall": recall,
        "avg_iou": avg_iou,
    }


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, confidence_threshold=0.5):
    """Validate the model with loss and detection metrics."""
    model.eval()
    total_loss = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    total_cls = 0.0
    num_batches = 0
    
    # Detection metrics accumulators
    all_detections = []  # (confidence, is_tp, iou)
    total_gt = 0
    total_pred = 0
    sum_iou = 0.0
    num_matched = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        images = batch["images"].to(device)
        targets = {
            "boxes": batch["boxes"].to(device),
            "num_boxes": batch["num_boxes"].to(device),
        }
        
        outputs = model(images, targets)
        losses = loss_fn(outputs, targets)
        
        total_loss += losses["loss"].item()
        total_bbox += losses["loss_bbox"].item()
        total_giou += losses["loss_giou"].item()
        total_cls += losses["loss_cls"].item()
        num_batches += 1
        
        # Compute detection metrics
        pred_boxes = outputs.get("pred_boxes")  # [B, Q, 4]
        pred_logits = outputs.get("pred_logits")  # [B, Q, 1]
        
        if pred_boxes is None or pred_logits is None:
            continue
        
        batch_size = images.size(0)
        gt_boxes = targets["boxes"]
        num_boxes = targets["num_boxes"]
        
        for b in range(batch_size):
            pred_box = pred_boxes[b]  # [Q, 4]
            pred_score = pred_logits[b].squeeze(-1).sigmoid()  # [Q]
            
            # Filter by confidence
            keep = pred_score > confidence_threshold
            pred_box = pred_box[keep]
            pred_score = pred_score[keep]
            
            n_gt = num_boxes[b].item()
            gt_box = gt_boxes[b, :n_gt]  # [n_gt, 4]
            
            total_gt += n_gt
            total_pred += len(pred_score)
            
            if len(pred_score) == 0:
                continue
            if n_gt == 0:
                for score in pred_score:
                    all_detections.append((score.item(), False, 0.0))
                continue
            
            # Compute IoU
            ious = box_iou_simple(pred_box, gt_box)  # [N, M]
            gt_matched = torch.zeros(n_gt, dtype=torch.bool, device=device)
            
            sorted_indices = pred_score.argsort(descending=True)
            for pred_idx in sorted_indices:
                pred_ious = ious[pred_idx]
                best_gt_idx = pred_ious.argmax().item()
                best_iou = pred_ious[best_gt_idx].item()
                
                if best_iou >= 0.5 and not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    all_detections.append((pred_score[pred_idx].item(), True, best_iou))
                    sum_iou += best_iou
                    num_matched += 1
                else:
                    all_detections.append((pred_score[pred_idx].item(), False, best_iou))
    
    # Compute detection metrics
    if all_detections:
        sorted_dets = sorted(all_detections, key=lambda x: -x[0])
        tp_cumsum = 0
        precisions = []
        recalls = []
        for conf, is_tp, iou in sorted_dets:
            if is_tp:
                tp_cumsum += 1
            precision = tp_cumsum / (len(precisions) + 1) if precisions else (1 if is_tp else 0)
            recall = tp_cumsum / total_gt if total_gt > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
        
        final_precision = tp_cumsum / total_pred if total_pred > 0 else 0
        final_recall = tp_cumsum / total_gt if total_gt > 0 else 0
        avg_iou = sum_iou / num_matched if num_matched > 0 else 0
    else:
        final_precision = 0
        final_recall = 0
        avg_iou = 0
    
    return {
        "loss": total_loss / max(num_batches, 1),
        "loss_bbox": total_bbox / max(num_batches, 1),
        "loss_giou": total_giou / max(num_batches, 1),
        "loss_cls": total_cls / max(num_batches, 1),
        "precision": final_precision,
        "recall": final_recall,
        "avg_iou": avg_iou,
        "matched": num_matched,
        "total_gt": total_gt,
        "total_pred": total_pred,
    }


def box_iou_simple(boxes1, boxes2):
    """Compute IoU between boxes in YOLO format (cx, cy, w, h)."""
    # Convert to xyxy
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    b1_area = boxes1[:, 2] * boxes1[:, 3]
    b2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area + 1e-8
    
    return inter_area / union_area


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CoOp Training for SAM3 Detector (with SAM3 Loss)")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Class name: {args.class_name}")
    print(f"Context tokens: {args.n_ctx}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Loss weights: bbox={args.loss_bbox}, giou={args.loss_giou}, cls={args.loss_cls}")
    print("=" * 60)
    
    # Build model
    print("\nBuilding SAM3 CoOp detector...")
    model = build_sam3_coop_detector(
        class_name=args.class_name,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        class_token_position=args.class_token_position,
        checkpoint_path=args.sam3_checkpoint,
        device=args.device,
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_coop_weights(args.resume)
    
    # Create loss function
    loss_fn = SAM3DetectionLoss(
        loss_bbox=args.loss_bbox,
        loss_giou=args.loss_giou,
        loss_cls=args.loss_cls,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = YOLOCoOpDataset(
        data_dir=args.data_dir,
        split="TRAIN",
        include_negatives=True,
    )
    val_dataset = YOLOCoOpDataset(
        data_dir=args.data_dir,
        split="VAL",
        include_negatives=True,
    )
    
    # Limit dataset size for testing
    if args.limit is not None:
        from torch.utils.data import Subset
        train_indices = list(range(min(args.limit, len(train_dataset))))
        val_indices = list(range(min(args.limit, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        print(f"Dataset limited to {len(train_dataset)} train, {len(val_dataset)} val images")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_coop_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_coop_batch,
        pin_memory=True,
    )
    
    # Optimizer - only CoOp parameters
    optimizer = torch.optim.SGD(
        model.get_trainable_params(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01,
    )
    
    # Fast-forward scheduler if resuming from later epoch
    if args.start_epoch > 1:
        for _ in range(args.start_epoch - 1 - args.warmup_epochs):
            scheduler.step()
        print(f"Scheduler advanced to epoch {args.start_epoch}, LR: {scheduler.get_last_lr()[0]:.5f}")
    
    # Training loop
    best_val_loss = float("inf")
    print("\nStarting training...")
    
    for epoch in range(args.start_epoch, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, loss_fn, args.device, 
            epoch, args.warmup_epochs, args.lr
        )
        
        # Validate (every eval_freq epochs or last epoch)
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            val_losses = validate(model, val_loader, loss_fn, args.device)
        else:
            val_losses = None
        
        # Update scheduler (after warmup)
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train: loss={train_losses['loss']:.3f} (b:{train_losses['loss_bbox']:.2f} g:{train_losses['loss_giou']:.2f} c:{train_losses['loss_cls']:.3f}) | P:{train_losses['precision']:.3f} R:{train_losses['recall']:.3f} IoU:{train_losses['avg_iou']:.2f}")
        if val_losses:
            print(f"  Val:   loss={val_losses['loss']:.3f} (b:{val_losses['loss_bbox']:.2f} g:{val_losses['loss_giou']:.2f} c:{val_losses['loss_cls']:.3f}) | P:{val_losses['precision']:.3f} R:{val_losses['recall']:.3f} IoU:{val_losses['avg_iou']:.2f}")
        print(f"  LR: {current_lr:.5f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_losses and val_losses["loss"] < best_val_loss:
            best_val_loss = val_losses["loss"]
            model.save_coop_weights(output_dir / "best_coop.pth")
            print(f"  ✓ New best model saved!")
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            model.save_coop_weights(output_dir / f"coop_epoch_{epoch}.pth")
    
    # Save final model
    model.save_coop_weights(output_dir / "final_coop.pth")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
