# Dataset Format

This document describes the dataset structure expected by the 3D localization pipeline.

The dataset consists of case folders containing medical images and bounding box annotations, along with a CSV index file defining the dataset splits.

## Folder Structure

Each case is stored inside its own directory.

data/
  processed/
    case_0001/
      scan.nrrd
      meta.json
    case_0002/
      scan.nrrd
      meta.json

Each case directory contains two files:

- **scan.nrrd**: the 3D medical image volume  
- **meta.json**: bounding box annotation for the target region

## Index File

Dataset splits are defined by a CSV file.

data/processed/localizer_index.csv

Example:

split,case_id,image,meta
train,case_0001,data/processed/case_0001/scan.nrrd,data/processed/case_0001/meta.json
train,case_0002,data/processed/case_0002/scan.nrrd,data/processed/case_0002/meta.json
val,case_0003,data/processed/case_0003/scan.nrrd,data/processed/case_0003/meta.json

Columns:

- **split** — dataset split (train / val / test)
- **case_id** — unique patient identifier
- **image** — path to the scan file
- **meta** — path to the annotation file

## Annotation Format

Bounding box annotations are stored inside `meta.json`.

Example:

{
  "bbox_mm": [xmin, ymin, zmin, xmax, ymax, zmax]
}

Example values:

{
  "bbox_mm": [12.5, -24.1, 45.0, 58.3, 16.8, 82.7]
}

## Bounding Box Definition

Bounding boxes follow the format:

[xmin, ymin, zmin, xmax, ymax, zmax]

All coordinates are expressed in **millimeters (mm)** in the **world coordinate system**.

The bounding box is axis-aligned.

xmin ≤ x ≤ xmax  
ymin ≤ y ≤ ymax  
zmin ≤ z ≤ zmax  

## Coordinate System

Coordinates follow the SimpleITK world coordinate convention.

Voxel coordinates are converted to world coordinates using:

world = origin + direction × diag(spacing) × voxel

Where:

- **origin** — world origin of the image  
- **spacing** — voxel spacing in millimeters  
- **direction** — 3×3 direction cosine matrix  

These values are obtained from the image metadata:

img.GetOrigin()  
img.GetSpacing()  
img.GetDirection()  

## Axis Ordering

The project uses two axis conventions.

### NumPy Volumes

Image volumes loaded using SimpleITK follow:

(Z, Y, X)

This corresponds to the array returned by:

sitk.GetArrayFromImage(img)

### Spatial Coordinates

Physical and voxel coordinates are expressed as:

(x, y, z)

This applies to:

- bbox_mm
- center_mm
- voxel coordinates

## Target Representation

The localization model predicts two outputs.

### Heatmap

Shape:

(1, Z, Y, X)

The heatmap is a 3D Gaussian centered at the bounding box center.

### Size

Shape:

(3,)

The size corresponds to the bounding box dimensions in millimeters:

[width_x, width_y, width_z]

## Target Generation

During preprocessing:

1. The bounding box center is computed:

center_mm = [
(xmin + xmax) / 2,
(ymin + ymax) / 2,
(zmin + zmax) / 2
]

2. The center is converted to voxel coordinates on the resampled grid.

3. A Gaussian heatmap is generated at that location.

## Resampling

Before training, images are resampled to a fixed spacing:

target spacing = (2.0, 2.0, 2.0) mm

This ensures that all scans share a consistent voxel grid.

## Summary

The localization dataset requires:

- one 3D scan per case
- a bounding box annotation in millimeters
- a CSV index defining train/validation/test splits
- image formats compatible with SimpleITK (NRRD, NIfTI, etc.)
