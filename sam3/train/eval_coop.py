#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Evaluation Script for SAM3 CoOp Detector.

Evaluates detection performance on validation set with standard metrics:
- Precision, Recall, F1
- mAP@0.5, mAP@0.5:0.95
- Average IoU

Usage:
    python -m sam3.train.eval_coop \
        --data_dir /home/cerrion/fallen_bottle_dataset \
        --class_name "A fallen bottle" \
        --coop_weights ./coop_checkpoints/best_coop.pth \
        --output_dir ./eval_results

    # Or evaluate baseline (without CoOp, just original text prompt):
    python -m sam3.train.eval_coop \
        --data_dir /home/cerrion/fallen_bottle_dataset \
        --class_name "A fallen bottle" \
        --baseline
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam3.train.data.coop_yolo_dataset import YOLOCoOpDataset, collate_coop_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAM3 CoOp detector")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to YOLO format dataset")
    parser.add_argument("--split", type=str, default="VAL",
                        help="Data split to evaluate on")
    parser.add_argument("--class_name", type=str, default="A fallen bottle",
                        help="Class name for the prompt")
    
    # Model arguments
    parser.add_argument("--coop_weights", type=str, default=None,
                        help="Path to trained CoOp weights")
    parser.add_argument("--n_ctx", type=int, default=8,
                        help="Number of context tokens (must match training)")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate baseline (original text prompt, no CoOp)")
    
    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Confidence threshold for detections (SAM3 default: 0.5)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for matching")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save predictions to JSON file")
    parser.add_argument("--save_samples", type=int, default=0,
                        help="Number of sample images to save with boxes (0 = none)")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    return parser.parse_args()


def box_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes in YOLO format (cx, cy, w, h).
    
    Args:
        boxes1: [N, 4] - first set of boxes
        boxes2: [M, 4] - second set of boxes
        
    Returns:
        [N, M] IoU matrix
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(boxes1.size(0), boxes2.size(0), device=boxes1.device)
    
    # Convert to x1, y1, x2, y2
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Compute intersection
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # Compute union
    b1_area = boxes1[:, 2] * boxes1[:, 3]
    b2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area + 1e-8
    
    return inter_area / union_area


def xyxy_to_yolo(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x1, y1, x2, y2) to YOLO format (cx, cy, w, h)."""
    if boxes.numel() == 0:
        return boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=1)


def draw_sample_image(
    image_path: str,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    output_path: Path,
):
    """Draw GT (green) and pred (red) boxes on image and save."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # Draw GT boxes in green
    for box in gt_boxes:
        cx, cy, bw, bh = box.tolist()
        x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
        x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 12), "GT", fill="green")
    
    # Draw pred boxes in red with scores
    for box, score in zip(pred_boxes, pred_scores):
        cx, cy, bw, bh = box.tolist()
        x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
        x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y2 + 2), f"{score:.2f}", fill="red")
    
    img.save(output_path)


def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    """Compute Average Precision using 11-point interpolation."""
    recalls = [0.0] + list(recalls) + [1.0]
    precisions = [0.0] + list(precisions) + [0.0]
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 11-point interpolation
    ap = 0.0
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p = 0.0
        for r, prec in zip(recalls, precisions):
            if r >= t:
                p = max(p, prec)
        ap += p / 11
    
    return ap


class DetectionEvaluator:
    """Evaluator for object detection with standard metrics."""
    
    def __init__(self, iou_thresholds: List[float] = None):
        if iou_thresholds is None:
            # mAP@0.5:0.95 with step 0.05
            iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        self.iou_thresholds = iou_thresholds
        self.reset()
    
    def reset(self):
        """Reset accumulator."""
        self.all_detections = []  # List of (confidence, is_tp, iou)
        self.total_gt = 0
        self.total_pred = 0
        self.sum_iou = 0.0
        self.num_matched = 0
    
    def update(
        self,
        pred_boxes: torch.Tensor,  # [N, 4] in YOLO format
        pred_scores: torch.Tensor,  # [N]
        gt_boxes: torch.Tensor,  # [M, 4] in YOLO format
    ):
        """Update metrics with predictions from one image."""
        self.total_gt += gt_boxes.size(0)
        self.total_pred += pred_boxes.size(0)
        
        if pred_boxes.size(0) == 0:
            return
        
        if gt_boxes.size(0) == 0:
            # All predictions are false positives
            for score in pred_scores:
                self.all_detections.append((score.item(), False, 0.0))
            return
        
        # Compute IoU matrix
        ious = box_iou_batch(pred_boxes, gt_boxes)  # [N, M]
        
        # Match predictions to GT (greedy matching)
        gt_matched = torch.zeros(gt_boxes.size(0), dtype=torch.bool, device=pred_boxes.device)
        
        # Sort predictions by confidence
        sorted_indices = pred_scores.argsort(descending=True)
        
        for pred_idx in sorted_indices:
            pred_ious = ious[pred_idx]
            best_gt_idx = pred_ious.argmax().item()
            best_iou = pred_ious[best_gt_idx].item()
            
            if best_iou >= 0.5 and not gt_matched[best_gt_idx]:
                # True positive
                gt_matched[best_gt_idx] = True
                self.all_detections.append((pred_scores[pred_idx].item(), True, best_iou))
                self.sum_iou += best_iou
                self.num_matched += 1
            else:
                # False positive
                self.all_detections.append((pred_scores[pred_idx].item(), False, best_iou))
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics."""
        if not self.all_detections:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap50": 0.0,
                "ap50_95": 0.0,
                "avg_iou": 0.0,
                "total_gt": self.total_gt,
                "total_pred": self.total_pred,
            }
        
        # Sort by confidence
        sorted_dets = sorted(self.all_detections, key=lambda x: -x[0])
        
        # Compute precision-recall curve
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        
        for conf, is_tp, iou in sorted_dets:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
            recall = tp_cumsum / self.total_gt if self.total_gt > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AP@0.5
        ap50 = compute_ap(recalls, precisions)
        
        # Final precision/recall at all detections
        final_precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
        final_recall = tp_cumsum / self.total_gt if self.total_gt > 0 else 0
        f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-8)
        
        # Average IoU of matched detections
        avg_iou = self.sum_iou / self.num_matched if self.num_matched > 0 else 0
        
        return {
            "precision": final_precision,
            "recall": final_recall,
            "f1": f1,
            "ap50": ap50,
            "avg_iou": avg_iou,
            "num_matched": self.num_matched,
            "total_gt": self.total_gt,
            "total_pred": self.total_pred,
        }


@torch.no_grad()
def evaluate_coop(
    model,
    dataloader: DataLoader,
    device: str,
    confidence_threshold: float = 0.3,
    save_samples: int = 0,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate the CoOp model on a dataset."""
    model.eval()
    evaluator = DetectionEvaluator()
    predictions = []
    samples_saved = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["images"].to(device)
        gt_boxes = batch["boxes"]  # [B, max_boxes, 4] - YOLO format
        num_boxes = batch["num_boxes"]  # [B]
        batch_size = images.size(0)
        
        # Forward pass
        outputs = model(images)
        
        # Extract predictions
        pred_boxes = outputs.get("pred_boxes")  # [B, num_queries, 4] - cx, cy, w, h
        pred_logits = outputs.get("pred_logits")  # [B, num_queries, 1]
        
        if pred_boxes is None or pred_logits is None:
            continue
        
        # Process each image
        for b in range(batch_size):
            pred_box = pred_boxes[b]  # [num_queries, 4]
            pred_score = pred_logits[b].squeeze(-1).sigmoid()  # [num_queries]
            
            # Handle presence token if available
            if "presence_logit_dec" in outputs:
                presence = outputs["presence_logit_dec"][b].sigmoid()
                pred_score = pred_score * presence
            
            # Filter by confidence
            keep = pred_score > confidence_threshold
            pred_box = pred_box[keep]
            pred_score = pred_score[keep]
            
            # Get GT boxes for this image
            n_gt = num_boxes[b].item()
            gt_box = gt_boxes[b, :n_gt].to(device)  # [n_gt, 4]
            
            # Update evaluator
            evaluator.update(pred_box, pred_score, gt_box)
            
            # Store predictions
            predictions.append({
                "image_path": batch["image_paths"][b],
                "pred_boxes": pred_box.cpu().tolist(),
                "pred_scores": pred_score.cpu().tolist(),
                "gt_boxes": gt_box.cpu().tolist(),
            })
            
            # Save sample images
            if save_samples > 0 and samples_saved < save_samples and output_dir:
                sample_path = output_dir / "samples" / f"sample_{samples_saved:03d}.jpg"
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                draw_sample_image(
                    batch["image_paths"][b],
                    pred_box.cpu(),
                    pred_score.cpu(),
                    gt_box.cpu(),
                    sample_path,
                )
                samples_saved += 1
    
    metrics = evaluator.compute_metrics()
    return metrics, predictions


@torch.no_grad()
def evaluate_baseline(
    dataloader: DataLoader,
    class_name: str,
    device: str,
    confidence_threshold: float = 0.3,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Evaluate the baseline SAM3 model (without CoOp, using original text prompt).
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    # Build model
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    print(f"Using text prompt: '{class_name}'")
    
    evaluator = DetectionEvaluator()
    predictions = []
    
    for batch in tqdm(dataloader, desc="Evaluating baseline"):
        gt_boxes = batch["boxes"]  # [B, max_boxes, 4] - YOLO format
        num_boxes = batch["num_boxes"]  # [B]
        image_paths = batch["image_paths"]
        batch_size = len(image_paths)
        
        # Process each image individually (processor expects raw image)
        for b in range(batch_size):
            # Load raw image from disk (processor applies its own transforms)
            raw_image = Image.open(image_paths[b]).convert("RGB")
            
            # Set image
            state = processor.set_image(raw_image)
            
            # Set text prompt
            state = processor.set_text_prompt(class_name, state)
            
            # Get predictions
            if "boxes" in state and len(state["boxes"]) > 0:
                pred_box = state["boxes"]  # [N, 4] - x1, y1, x2, y2 in pixel coords
                pred_score = state["scores"]  # [N]
                
                # Convert to normalized YOLO format
                pred_box = pred_box / processor.resolution
                pred_box = xyxy_to_yolo(pred_box)
            else:
                pred_box = torch.zeros((0, 4), device=device)
                pred_score = torch.zeros((0,), device=device)
            
            # Get GT boxes for this image
            n_gt = num_boxes[b].item()
            gt_box = gt_boxes[b, :n_gt].to(device)  # [n_gt, 4]
            
            # Update evaluator
            evaluator.update(pred_box, pred_score, gt_box)
            
            # Store predictions
            predictions.append({
                "image_path": batch["image_paths"][b],
                "pred_boxes": pred_box.cpu().tolist(),
                "pred_scores": pred_score.cpu().tolist(),
                "gt_boxes": gt_box.cpu().tolist(),
            })
    
    metrics = evaluator.compute_metrics()
    return metrics, predictions


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SAM3 CoOp Detector Evaluation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Class name: {args.class_name}")
    print(f"Mode: {'Baseline' if args.baseline else 'CoOp'}")
    if not args.baseline:
        print(f"CoOp weights: {args.coop_weights}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = YOLOCoOpDataset(
        data_dir=args.data_dir,
        split=args.split,
        include_negatives=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_coop_batch,
        pin_memory=True,
    )
    
    # Evaluate
    if args.baseline:
        metrics, predictions = evaluate_baseline(
            dataloader=dataloader,
            class_name=args.class_name,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
        )
    else:
        # Build CoOp model
        from sam3.model.sam3_coop_wrapper import build_sam3_coop_detector
        
        print("\nBuilding CoOp detector...")
        model = build_sam3_coop_detector(
            class_name=args.class_name,
            n_ctx=args.n_ctx,
            device=args.device,
        )
        
        # Load weights if provided
        if args.coop_weights:
            model.load_coop_weights(args.coop_weights)
        else:
            print("WARNING: No CoOp weights provided, using random initialization")
        
        metrics, predictions = evaluate_coop(
            model=model,
            dataloader=dataloader,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
            save_samples=args.save_samples,
            output_dir=output_dir,
        )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1']:.4f}")
    print(f"  AP@0.5:       {metrics['ap50']:.4f}")
    print(f"  Avg IoU:      {metrics['avg_iou']:.4f}")
    print(f"  Matched:      {metrics.get('num_matched', 0)}/{metrics['total_gt']} GT boxes")
    print(f"  Predictions:  {metrics['total_pred']} total")
    print("=" * 60)
    
    # Save results
    results = {
        "config": {
            "data_dir": args.data_dir,
            "split": args.split,
            "class_name": args.class_name,
            "mode": "baseline" if args.baseline else "coop",
            "coop_weights": args.coop_weights,
            "confidence_threshold": args.confidence_threshold,
        },
        "metrics": metrics,
    }
    
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    if args.save_predictions:
        pred_path = output_dir / "predictions.json"
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to: {pred_path}")


if __name__ == "__main__":
    main()
