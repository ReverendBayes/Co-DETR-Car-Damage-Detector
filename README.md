# Car Damage Detection using Co-DETR: 
## A Transformer-Based Learner with CBAM Attention, Hybrid Loss, and Augmented Training

A deep learning–based computer vision training pipeline for car damage detection using a Co-DETR learner enhanced with CBAM Attention, Hybrid Loss, and Albumentations. Trains on Colab to identify and localize car body defects such as scratches, dents, and rust. Includes end-to-end model training and quantitative evaluation.

---

## System Overview

This model implements a detection system using a Swin Transformer backbone, CBAM attention, transformer encoder-decoder architecture, and a custom training loop with focal loss and Hungarian matching. It uses Albumentations-based data augmentation, supports auxiliary losses from intermediate decoder layers, and is optimized for car damage classification and localization tasks on COCO-style CarDD data.

---
This is a deep learning detection training pipeline, and specifically:
* Transformer-based object detection (DETR architecture variant)
* Uses Swin Transformer as a vision backbone (deep convolutional + attention layers)
* Incorporates CBAM, which is a deep attention mechanism
* Trained end-to-end using gradient descent on labeled data
* Uses loss functions (focal loss, GIoU) and learnable weights across multiple layers

---

## Component Table
A Co-DETR Transformer-Based Training Pipeline for a Learner Enhanced with CBAM Attention, Hybrid Loss, and Albumentation-Augmented Training for Vehicle Damage Detection:  

| Component                | Implementation Details | Notes |
|--------------------------|------------------------|-------|
| **Backbone**             | Swin Transformer (`swin_base_patch4_window12_384`) via `timm` | Pretrained on ImageNet; outputs last feature map |
| **Feature Attention**    | CBAM applied post-backbone | Improves spatial/channel-wise focus before transformer |
| **Projection Layer**     | 1×1 Conv2D → `hidden_dim=256` | Reduces backbone output to transformer-compatible shape |
| **Positional Encoding**  | Sinusoidal 1D (`PositionalEncoding`) | Injects spatial information into transformer input |
| **Encoder**              | 6-layer `TransformerEncoder` | `d_model=256`, `nhead=8` |
| **Decoder**              | 6-layer `TransformerDecoder` | Produces class and bbox predictions per query |
| **Prediction Heads**     | Class: `Linear(256 → num_classes)`<br>BBox: 2-layer MLP → `[cx,cy,w,h]` | DETR-style output heads |
| **Label Assignment**     | Hungarian matching (`linear_sum_assignment`) | Minimizes cost across classification, bbox, and IoU |
| **Classification Loss**  | Weighted Focal Loss (`gamma=2.0`, dynamic `alpha`) | Handles class imbalance using inverse frequency |
| **Box Loss**             | GIoU Loss (`torchvision.ops`) | Improves overlap + geometry sensitivity |
| **Augmentations**        | Albumentations: Mosaic, Cutout, affine, blur, color, resize | Increases robustness; supports bbox format |
| **Training Monitoring**  | Custom logging of loss and accuracy + matplotlib plots | Step 9: Logs metrics each epoch, saves graph |
| **Evaluation**           | Accuracy, Precision, Recall, mAP, AUC | Uses `sklearn` for classification metrics |
| **Optimizer**            | AdamW (`lr=1e-4`) | Transformer-friendly, weight decay-aware |
| **LR Scheduler**         | `StepLR` (step=5, gamma=0.5) | Reduces LR over time to stabilize convergence |
| **Early Stopping**       | Stops if mAP stagnates for 10 epochs | Prevents overtraining |
| **Batch/Epoch Settings** | `batch_size=2`, `num_epochs=150` or debug | Optimized for Colab execution memory |

---

## Pipeline by Step

### Step 4: Dataset Loader
I used a COCO-format loader for the Car Damage Dataset (CarDD), filtering out invalid or irrelevant labels (such as crowd annotations or near-zero box areas). The loader outputs tuples of images, bounding boxes, and labels.

- Implements custom `CarDDDataset` class for loading COCO-style annotations and images:
  - Reads bounding boxes and labels from `instances_train2017.json` or `instances_val2017.json`
  - Loads image files from specified directory (`train2017` or `val2017`)
- Applies annotation filtering logic to remove invalid samples:
  - Skips `iscrowd == 1` entries
  - Excludes annotations with width or height ≤ 1
  - Filters out samples with rare class ID 6
- Converts COCO-format `[x, y, w, h]` boxes to Pascal VOC `[x1, y1, x2, y2]`
- Supports Albumentations compatibility:
  - Returns boxes in `pascal_voc` format
  - Exposes bounding boxes and labels via `bbox_params` interface
- Output per sample includes:
  - Transformed image (as NumPy array or tensor)
  - Bounding boxes as list of floats
  - Labels as list of class IDs
  - Image ID for logging or visualization


---

### Step 5: Positional Encoding
I applied sinusoidal positional encodings to the transformer input sequence, enabling the model to distinguish positional relationships in flattened image patches.

- Defines a custom `PositionalEncoding` module using sine and cosine functions
- Applies sinusoidal positional embeddings to input tensor of shape `[B, HW, C]`
- Adds unique encodings per spatial token based on patch position and channel
- Injects positional structure into flattened image features prior to transformer encoding
- Maintains spatial context through the transformer pipeline:
  - Helps model preserve object relationships over flattened sequences

---

### Step 6: Model Architecture
The architecture begins with a Swin Transformer backbone and injects CBAM attention to refine its spatial awareness. The final feature map is projected and flattened, then passed through a transformer encoder and decoder. Object queries are decoded into bounding box coordinates and class logits.

- Uses Swin Transformer backbone from `timm` with `features_only=True`
- Applies CBAM attention module:
  - Channel attention (shared MLP over avg/max pooled channels)
  - Spatial attention (sigmoid gating over pooled spatial map)
- Projects CBAM-enhanced output via `Conv2d(→ hidden_dim=256)` to match transformer input shape
- Flattens `[B, C, H, W]` into `[B, HW, C]` and adds sinusoidal positional encoding
- Processes token sequence with 6-layer `TransformerEncoder` using `nhead=8`, `batch_first=True`
- Initializes decoder with 100 learnable query embeddings (`num_queries=100`)
- Uses 6-layer `TransformerDecoder` to produce query-conditioned predictions
- Outputs per-query predictions:
  - `pred_logits`: class logits (before softmax)
  - `pred_boxes`: normalized `[cx, cy, w, h]` format passed through `sigmoid`
- One-time tensor shape debug print:
  - Logs `features`, `x`, `memory`, and `queries` on first forward pass

An optional SCYLLA-IoU loss function (see: scylla_iou_loss.py) is implemented but not yet activated. This function can replace the GIoU loss to include center-distance and aspect-ratio penalties in box regression.

---

### Step 7: Training Loop
Hungarian matching is used to assign predictions to ground truth targets by minimizing a cost that includes classification, L1 distance, and IoU components. The loop computes classification and bounding box losses across both main and auxiliary decoder outputs. Numerical stability is monitored via gradient checks.

- Hungarian matcher minimizes combined:
  - Classification cost (softmax probabilities + focal loss weighting)
  - L1 distance between predicted and target boxes
  - Generalized IoU loss for spatial misalignment
- Uses weighted cost matrix with weights:
  - 1.0 (classification), 5.0 (bbox L1), 2.0 (GIoU)
- Custom `compute_losses()` function:
  - Matches each sample’s predictions to targets via cost matrix
  - Applies weighted focal loss for classification per matched query
  - Applies GIoU loss after converting boxes to `[x1, y1, x2, y2]` format
- Handles decoder and `aux_outputs` supervision:
  - Computes losses for main decoder and each auxiliary prediction stage
  - Aggregates classification and bbox losses across stages
- Performs:
  - `loss.backward()` followed by `optimizer.step()` and `scheduler.step()`
  - Logs `loss.item()` each epoch before backpropagation
  - Checks all model gradients for NaNs each batch (gradient sanity check)
- Returns average total loss across batches for epoch summary

---

### Step 8: Training Script
This script initializes datasets and loaders with Albumentations transforms. It calculates class frequencies for the focal loss alpha vector, instantiates the model and optimizer, and runs a training loop with early stopping. Evaluation metrics are printed each epoch.

- Initializes Albumentations-based augmentation pipeline for training:
  - Includes `Mosaic`, `Cutout`, `RandomBrightnessContrast`, `HueSaturationValue`, `RGBShift`, `RandomResizedCrop`, `Blur`, and `Normalize`
  - Applies `ToTensorV2()` for PyTorch tensor conversion post-augmentation
- Defines custom `collate_fn` to:
  - Handle Albumentations-transformed samples with bounding boxes
  - Return batched tensors, boxes, labels, and image IDs
- Instantiates training and validation datasets using `CarDDDataset`:
  - Training uses Albumentations transform
  - Validation uses deterministic resize + normalize transform
- Recalculates class frequencies after filtering:
  - Uses `Counter` to tally label occurrences
  - Computes class weighting vector α for use in focal loss
- Initializes:
  - `CoDETR` model with CBAM and `num_queries=100`
  - `WeightedFocalLoss` with dynamic class weights and `gamma=2.0`
  - `AdamW` optimizer (`lr=1e-4`) and `StepLR` scheduler (`step_size=5`, `gamma=0.5`)
- Executes training loop per epoch:
  - Calls `train_one_epoch()` to compute training loss
  - Calls `evaluate()` to compute validation predictions
- Computes validation metrics via `sklearn.metrics`:
  - Accuracy, Precision, Recall, mAP, and AUC (macro averaged)
- Applies early stopping logic:
  - Stops training if macro mAP fails to improve for `patience=10` epochs
- Saves best model checkpoint to:
  - `"/content/drive/MyDrive/Colab Notebooks/cardd_output/co_detr_model.pth"`

---
### Step 9: Logging and Visualization

- Tracks training metrics using:
  - `loss_log = []` and `acc_log = []` initialized at global scope
  - `log_metrics(loss, acc)` function appends values per epoch
- Implements `plot_training_metrics()` for performance visualization:
  - Creates 2×1 subplot with:
    - Left plot: training loss vs. epoch
    - Right plot: validation accuracy vs. epoch
  - Adds grid, labels, and titles for each axis
- Saves output figure to persistent location:
  - `"/content/drive/MyDrive/Colab Notebooks/cardd_output/training_metrics.png"`
- Provides visual diagnostics of learning behavior:
  - Tracks convergence trends
  - Detects training instability or plateau

---

## Optional Features (Already Coded, Not Yet Activated)

| Feature | Description |
|--------|-------------|
| **SCYLLA-IoU Loss** | Drop-in replacement for GIoU with shape + distance components |
| **DSI Metric**      | Damage severity scoring post-inference (can be added in Step 10) |

---

## Design Rationale
This system is built to detect localized car damage in a multi-class COCO-style dataset using minimal supervision and transformer-based reasoning. It leverages pretrained CNN vision priors, enriched via CBAM attention, and token-based query decoding to localize damage. Focal loss mitigates imbalance in class representation, while GIoU penalizes box misalignment more effectively than L1 or IoU alone.

It balances the architectural strength of transformers with CNN-based spatial inductive bias via CBAM. It uses Swin's hierarchical feature extraction and Albumentations' aggressive augmentation to promote generalization. Focal loss addresses dataset imbalance, and GIoU penalizes spatial error in localization. Object detection is framed as a set prediction problem, solved via Hungarian bipartite matching. CBAM attention sharpens activation on damaged regions, while Albumentations synthetic transformations improve robustness to image diversity.

Albumentations augments training diversity through Mosaic, Cutout, and affine transforms, which enhances spatial generalization. The architecture is modular, resilient to data imbalance, and designed to support further enhancements such as SCYLLA-IoU and post-hoc severity scoring. The modularity of the architecture allows plug-and-play replacement of the loss function (e.g., to SCYLLA-IoU) and potential addition of custom severity scoring metrics (e.g., DSI). The model is optimized for fine-grained vehicle damage classification and localization on a moderately imbalanced dataset.


---
📂 Dataset: CarDD
This project uses the Car Damage Detection Dataset (CarDD) from USTC:
* GitHub: https://github.com/CarDD-USTC/CarDD-USTC.github.io
* Project Site: https://cardd-ustc.github.io
* Direct Download: Google Drive Link
Wang, Xinkuang; Li, Wenjing; Wu, Zhongcheng. CarDD: A New Dataset for Vision-Based Car Damage Detection. IEEE Transactions on Intelligent Transportation Systems, 2023. DOI: 10.1109/TITS.2023.3258480

💻 Training Environment
This repo is designed to run in Google Colab and requires the following dependencies:
torch>=1.13
torchvision>=0.14
timm
albumentations
opencv-python
matplotlib
scikit-learn
pycocotools
You can install them in Colab with:
!pip install -q timm albumentations pycocotools
