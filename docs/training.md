# Training

This document explains how to train the 3D localization model.

The model learns to predict two outputs from a 3D medical scan:

• a heatmap representing the target center  
• the bounding box size in millimeters  

---

# Training Script

Training is executed using the following command:

python scripts/train.py

Example:

python scripts/train.py --index-csv data/processed/localizer_index.csv --outdir outputs/run01 --epochs 50 --lr 1e-4

---

# Required Dataset

Training requires a dataset index file:

data/processed/localizer_index.csv

The index file defines the dataset splits:

train  
val  
test  

Each row contains:

• the scan path  
• the bounding box annotation path  

Example:

split,case_id,image,meta  
train,case_0001,data/processed/case_0001/scan.nrrd,data/processed/case_0001/meta.json  
train,case_0002,data/processed/case_0002/scan.nrrd,data/processed/case_0002/meta.json  
val,case_0003,data/processed/case_0003/scan.nrrd,data/processed/case_0003/meta.json  

---

# Training Pipeline

The training pipeline performs the following steps.

### 1. Image Loading

Scans are loaded using SimpleITK.

Supported formats include:

• NRRD  
• NIfTI  
• other SimpleITK compatible formats  

---

### 2. Resampling

Images are resampled to a fixed voxel spacing:

(2.0, 2.0, 2.0) mm

This ensures a consistent spatial resolution across the dataset.

---

### 3. Intensity Normalization

CT intensities are clipped to the following range:

[-150, 350]

After clipping, the image is normalized using:

x = (x − mean) / std

---

### 4. Target Generation

Targets are generated from the bounding box annotation.

Bounding box format:

[xmin, ymin, zmin, xmax, ymax, zmax]

All values are given in millimeters.

The bounding box center is computed as:

center = ((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2)

The center is converted from world coordinates to voxel coordinates.

A Gaussian heatmap is generated at the center location.

Heatmap shape:

(1, Z, Y, X)

The bounding box size is computed as:

size = [width_x, width_y, width_z]

---

# Model Architecture

The model is a 3D U-Net style network.

Input tensor:

(B, 1, Z, Y, X)

Model outputs:

heatmap : (B, 1, Z, Y, X)  
size    : (B, 3)

The architecture includes:

• encoder-decoder structure  
• skip connections  
• instance normalization  
• leaky ReLU activations  

---

# Loss Function

Training optimizes two objectives.

Heatmap loss:

MSE(heat_pred, heat_target)

Size regression loss:

SmoothL1(size_pred, size_target)

Combined loss:

total_loss = heat_loss + λ × size_loss

Default value:

λ = 0.1

---

# Validation Metrics

Validation evaluates localization quality using the following metrics.

Center Error

The Euclidean distance between predicted and ground truth center:

error = ||pred_center − gt_center||

Unit: millimeters

P@20mm

Percentage of cases where:

center_error ≤ 20 mm

3D IoU

Intersection-over-union between predicted and ground truth bounding boxes.

---

# Training Outputs

Training outputs are saved to:

outputs/run_name/

Typical directory:

outputs/run01/

Files generated during training:

best.pt  
last.pt  
history.json  

best.pt → checkpoint with best validation performance  
last.pt → checkpoint from the final epoch  
history.json → training metrics and logs  

---

# Example Training Command

python scripts/train.py --index-csv data/processed/localizer_index.csv --outdir outputs/run01 --epochs 50 --lr 1e-4 --batch-size 1

---

# Hardware Requirements

Recommended setup:

GPU training is recommended  
8-16 GB VRAM depending on volume size  

CPU training is supported but significantly slower.

---

# Summary

Training consists of the following steps:

1. load 3D scans  
2. resample to a fixed voxel spacing  
3. generate Gaussian heatmap targets  
4. train a 3D U-Net localization network  
5. evaluate localization accuracy using center error and IoU
