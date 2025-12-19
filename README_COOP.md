# SAM3 CoOp

CoOp detector fine tuning for the SAM3.

## Setup

Ensure you have the required dependencies installed (SAM3 environment).

## Dataset
The dataset must follow the standard YOLO directory structure:

```text
data_dir/
├── classes.txt        # List of class names, one per line
├── images/
│   ├── TRAIN/         # Training images (.jpg, .png, etc.)
│   ├── VAL/           # Validation images
│   └── TEST/          # Test images
└── labels/
    ├── TRAIN/         # YOLO format labels (.txt)
    ├── VAL/           # Validation labels
    └── TEST/          # Test labels
```

## Training

To train with `n_ctx` learnable prompts for a new class:

```bash
python -m sam3.train.train_coop \
    --data_dir /path/to/yolo_dataset \
    --class_name "A fallen bottle" \
    --n_ctx 8 \
    --epochs 50 \
    --batch_size 8 \
    --output_dir ./coop_checkpoints_bottle
```

### Arguments ([train_coop.py](file:///home/cerrion/sam3/sam3/train/train_coop.py))
- `--data_dir`: Path to your dataset in YOLO format.
- `--class_name`: The class name to optimize for.
- `--n_ctx`: Number of learnable context tokens (default: `8`).
- `--lr`: Learning rate (default: `0.002`).
- `--epochs`: Number of training epochs (default: `50`).
- `--batch_size`: Batch size for training (default: `8`).
- `--loss_bbox`: L1 box loss weight (default: `5.0`).
- `--loss_giou`: GIoU loss weight (default: `2.0`).
- `--loss_cls`: Classification loss weight (default: `1.0`).
- `--output_dir`: Directory to save checkpoints (default: `./coop_checkpoints`).
- `--save_freq`: Save checkpoint every N epochs (default: `10`).
- `--eval_freq`: Run validation every N epochs (default: `5`).
- `--num_workers`: Number of data loading workers (default: `4`).
- `--device`: Device to use (default: `cuda`).
- `--resume`: Path to a CoOp checkpoint to resume training.
- `--sam3_checkpoint`: Optional path to a specific SAM3 weight file.

## Evaluation

To evaluate the fine-tuned CoOp model (or the baseline SAM3):

### CoOp Evaluation:
```bash
python -m sam3.train.eval_coop \
    --data_dir /path/to/yolo_dataset \
    --class_name "Fallen Bottle" \
    --split VAL \
    --coop_weights ./coop_checkpoints_bottle/best_coop.pth \
    --output_dir ./eval_results_coop \
    --save_txt \
    --save_samples 10
```

### Baseline Evaluation (Standard SAM3):
```bash
python -m sam3.train.eval_coop \
    --data_dir /path/to/yolo_dataset \
    --class_name "Fallen Bottle" \
    --split VAL \
    --baseline \
    --output_dir ./eval_results_baseline \
    --save_txt \
    --save_samples 10
```

### Key Arguments ([eval_coop.py](file:///home/cerrion/sam3/sam3/train/eval_coop.py))
- `--split`: Data split to evaluate on (default: `VAL`).
- `--batch_size`: Batch size for evaluation (default: `4`).
- `--confidence_threshold`: Minimum confidence to keep a detection (default: `0.5`).
- `--iou_threshold`: IoU threshold for evaluation metrics (default: `0.5`).
- `--baseline`: If set, evaluates the standard SAM3 model without CoOp.
- `--coop_weights`: Path to the trained `.pth` checkpoint.
- `--save_txt`: Saves predictions in YOLO `.txt` format (includes score).
- `--save_samples N`: Saves N visualization images with predicted boxes (default: `0`).
- `--output_dir`: Directory to save evaluation results (default: `./eval_results`).
- `--num_workers`: Number of data loading workers (default: `4`).
- `--device`: Device to use (default: `cuda`).