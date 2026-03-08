# Inference

This document explains how to run inference using the trained 3D localization model.

The inference pipeline predicts:

• the center of the target region  
• the bounding box size  
• the final bounding box in millimeters  

from a 3D medical image volume.

---

# Inference Script

Inference can be executed using:

python scripts/eval.py

or for visualization:

python scripts/visualize_case.py

Example:

python scripts/eval.py --index-csv data/processed/localizer_index.csv --checkpoint outputs/run01/best.pt

---

# Required Inputs

Inference requires the following inputs:

• a trained model checkpoint  
• a dataset index file  
• scan volumes referenced in the dataset index  

Example checkpoint:

outputs/run01/best.pt

Dataset index:

data/processed/localizer_index.csv

---

# Inference Pipeline

The inference pipeline follows the same preprocessing steps used during training.

### 1. Image Loading

Scans are loaded using SimpleITK.

Supported formats include:

• NRRD  
• NIfTI  
• other SimpleITK compatible formats  

---

### 2. Resampling

Images are resampled to the same voxel spacing used during training:

(2.0, 2.0, 2.0) mm

This ensures compatibility with the trained model.

---

### 3. Intensity Normalization

CT intensities are clipped to the range:

[-150, 350]

Then normalized using:

x = (x − mean) / std

---

# Model Prediction

The model receives a 3D tensor:

(B, 1, Z, Y, X)

and predicts:

heatmap : (B, 1, Z, Y, X)  
size    : (B, 3)

The heatmap indicates the predicted center location.

---

# Decoding the Heatmap

The predicted center is obtained by finding the maximum value in the heatmap.

Steps:

1. locate the voxel with maximum activation
2. convert the voxel coordinate to world coordinates
3. obtain the center in millimeters

---

# Bounding Box Prediction

The predicted bounding box is constructed using:

pred_center_mm  
pred_size_mm  

Bounding box format:

[xmin, ymin, zmin, xmax, ymax, zmax]

Computed as:

xmin = cx − width_x / 2  
ymin = cy − width_y / 2  
zmin = cz − width_z / 2  

xmax = cx + width_x / 2  
ymax = cy + width_y / 2  
zmax = cz + width_z / 2  

---

# Optional Margin

During visualization or cropping, a margin can be added to the predicted bounding box.

Example:

margin = 25 mm

This enlarges the predicted region for downstream processing.

---

# Visualization

Predictions can be visualized using:

python scripts/visualize_case.py

The visualization shows:

• axial slice  
• coronal slice  
• sagittal slice  

Each view displays:

• predicted bounding box  
• ground truth bounding box  

---

# Example Command

python scripts/visualize_case.py --index-csv data/processed/localizer_index.csv --checkpoint outputs/run01/best.pt --case 0

---

# Output

Inference returns:

• predicted center in millimeters  
• predicted bounding box  
• predicted bounding box size  

Example:

center_mm = [32.4, -14.8, 65.2]

bbox_mm = [10.2, -40.3, 52.0, 54.6, 12.0, 78.4]

size_mm = [44.4, 52.3, 26.4]

---

# Summary

Inference consists of the following steps:

1. load the trained model checkpoint  
2. load the scan volume  
3. resample to the target voxel spacing  
4. normalize intensities  
5. run the model forward pass  
6. decode the heatmap center  
7. construct the predicted bounding box
