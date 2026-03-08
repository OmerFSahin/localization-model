# 3D Medical Image Localization

A PyTorch-based 3D localization framework for detecting anatomical target regions in volumetric medical scans.

The model predicts the spatial location of a target structure using a heatmap-based localization approach and estimates the corresponding bounding box size directly from the 3D volume.

The system is designed for medical imaging pipelines and provides a modular framework for dataset preprocessing, model training, evaluation, and visualization of localization results.

## Project Overview

Localizing anatomical structures in volumetric medical images is a fundamental task in many medical imaging pipelines. 
Accurate localization enables downstream tasks such as segmentation, cropping, detection, and quantitative analysis.

This project implements a **3D heatmap-based localization model** that predicts the center and spatial extent of a target anatomical region directly from a full 3D scan.

The model processes volumetric medical images and produces two outputs:

- a **3D Gaussian heatmap** indicating the predicted center location of the target structure
- the **bounding box size** representing the spatial extent of the region of interest

The predicted center is obtained by decoding the maximum activation of the heatmap, and the bounding box is constructed using the predicted center and size.

The framework includes a complete pipeline for:

- dataset indexing and validation
- volumetric preprocessing and resampling
- target heatmap generation
- model training and evaluation
- inference and visualization of predicted bounding boxes

The implementation is modular and designed to be easily extended for different anatomical targets and medical imaging datasets.

## Key Features

- **3D Heatmap-based Localization** 
  The model predicts a Gaussian heatmap representing the spatial center of the target anatomical region.

- **Bounding Box Size Regression** 
  In addition to the heatmap, the network predicts the physical size of the target region in millimeters.

- **3D U-Net Architecture** 
  A fully convolutional 3D encoder–decoder architecture with skip connections is used for volumetric feature extraction.

- **Medical Imaging Pipeline with SimpleITK** 
  The framework supports common medical imaging formats such as NRRD and NIfTI using SimpleITK.

- **Automatic Dataset Indexing** 
  Utilities are provided to automatically generate dataset splits and index files.

- **Robust Preprocessing Pipeline** 
  Includes resampling to a fixed voxel spacing, intensity normalization, and spatial padding for stable training.

- **Modular Training Framework** 
  Clean separation of datasets, models, training loops, and evaluation utilities for easy experimentation.

- **Visualization Tools** 
  Built-in scripts allow visualization of predicted and ground truth bounding boxes on axial, coronal, and sagittal views.

- **Evaluation Metrics for Localization** 
  Includes center localization error, success rate at a distance threshold (P@20mm), and 3D bounding box IoU.

- **Unit Tests for Core Components** 
  Tests are included for dataset integrity, coordinate transformations, and model forward passes.

## Repository Structure
 ```
localization-model/
├── src/localization/ Core source code for the localization framework.
├── src/localization/data/ Dataset utilities including indexing, preprocessing, and dataset loading.
├── src/localization/transforms/ Image resampling and spatial transformation utilities.
├── src/localization/geometry/ Coordinate conversion functions between voxel and world space.
├── src/localization/targets/ Target generation utilities such as Gaussian heatmap creation.
├── src/localization/models/ Neural network architectures including the 3D U-Net localization model.
├── src/localization/train/ Training utilities including loss functions and the training loop.
├── src/localization/eval/ Evaluation metrics for localization performance.
├── src/localization/inference/ Utilities for decoding model predictions and constructing bounding boxes.
├── src/localization/viz/ Visualization tools for displaying predictions on medical images.
├── scripts/ Command-line scripts for dataset preparation, training, evaluation, and visualization.
├── data/ Dataset directory containing raw data, processed data, and example samples.
├── outputs/ Training outputs including model checkpoints, logs, and experiment results.
├── tests/ Unit tests for dataset validation, coordinate transformations, and model behavior.
├── docs/ Project documentation including dataset format, training guide, and inference instructions.
└── README.md Main documentation for the repository.
 ```
## Installation

Clone the repository:

git clone https://github.com/OmerFSahin/localization-model.git
cd localization-model

Create a Python environment (recommended)

Install the required dependencies:
```
pip install -r requirements.txt
```
Alternatively, install the project in editable mode:
```
pip install -e .
```
This allows importing the project modules directly in Python:

from localization.models.unet3d import LocalizerNet

## Dataset Format

Each dataset case consists of a 3D medical scan and a corresponding bounding box annotation describing the region of interest.

A typical dataset structure is organized as follows:

data/
  processed/
    case_0001/
      scan.nrrd
      meta.json
    case_0002/
      scan.nrrd
      meta.json

Each case directory contains:

scan.nrrd 
The volumetric medical image.

meta.json 
Annotation file containing the bounding box of the target region.

Example annotation format:

{
  "bbox_mm": [xmin, ymin, zmin, xmax, ymax, zmax]
}

The bounding box is expressed in **world coordinates (millimeters)** using the order:

[xmin, ymin, zmin, xmax, ymax, zmax]

This represents the spatial extent of the target region in the scan.

Additional details about coordinate systems, axis ordering, and preprocessing are described in:

docs/dataset_format.md

## Create Dataset Index

Before training, the dataset must be indexed so that the training pipeline knows where to find the scans and annotations.

This repository uses a CSV index file that defines the dataset splits and file locations.

To generate the index file automatically, run:

python scripts/make_index.py

The script scans the dataset directory and creates the following file:

data/processed/localizer_index.csv

The index file contains one row per case with the following fields:

split 
Dataset split identifier (train, val, or test).

case_id 
Unique identifier of the case.

image 
Path to the 3D scan.

meta 
Path to the bounding box annotation file.

Example index file:

split,case_id,image,meta
train,case_0001,data/processed/case_0001/scan.nrrd,data/processed/case_0001/meta.json
train,case_0002,data/processed/case_0002/scan.nrrd,data/processed/case_0002/meta.json
val,case_0003,data/processed/case_0003/scan.nrrd,data/processed/case_0003/meta.json

This file is used by the dataset loader to build training and validation datasets.

## Training

After creating the dataset index, the model can be trained using the training script.

Run the following command:

python scripts/train.py --index-csv data/processed/localizer_index.csv --outdir outputs/run01 --epochs 50 --lr 1e-4

This script performs the full training pipeline including:

• loading the dataset 
• resampling volumes to a fixed voxel spacing 
• generating Gaussian heatmap targets 
• training the 3D localization network 
• evaluating performance on the validation set 

Training outputs are saved to the specified output directory:

outputs/run01/

Typical contents of the output directory:

best.pt 
Checkpoint with the best validation performance.

last.pt 
Checkpoint from the final training epoch.

history.json 
Training history including losses and validation metrics.

The model is trained to minimize two objectives:

heatmap loss 
Mean squared error between predicted and target heatmaps.

size regression loss 
Smooth L1 loss between predicted and ground truth bounding box sizes.

The combined training loss is:

total_loss = heat_loss + λ × size_loss

where λ controls the weight of the size regression loss.

## Evaluation

After training, the model can be evaluated on the validation or test dataset using the evaluation script.

Run evaluation with a trained checkpoint:

python scripts/eval.py --index-csv data/processed/localizer_index.csv --checkpoint outputs/run01/best.pt

The evaluation pipeline loads the trained model and computes localization performance on the dataset.

During evaluation, the model predicts:

• a 3D heatmap representing the center of the target region 
• the bounding box size of the region 

The predicted center is obtained by finding the maximum activation in the heatmap and converting the voxel coordinate to world coordinates.

The following metrics are computed.

Center Error

The Euclidean distance between the predicted center and the ground truth center.

center_error = ||pred_center − gt_center||

The error is measured in millimeters.

P@20mm

Percentage of cases where the localization error is below a threshold of 20 millimeters.

center_error ≤ 20 mm

This metric measures the success rate of the localization model.

3D IoU

Intersection-over-union between the predicted bounding box and the ground truth bounding box.

This evaluates how well the predicted spatial extent of the target region matches the ground truth annotation.

## Inference

Once the model has been trained, it can be used to localize the target region in new 3D medical scans.

Inference consists of running a forward pass through the trained network and decoding the predicted heatmap to obtain the target location.

Run inference using a trained checkpoint:

python scripts/eval.py --index-csv data/processed/localizer_index.csv --checkpoint outputs/run01/best.pt

The inference pipeline performs the same preprocessing steps used during training.

Image Loading 
The 3D scan is loaded using SimpleITK.

Resampling 
The image is resampled to the target voxel spacing used during training.

Intensity Normalization 
CT intensities are clipped and normalized before being passed to the model.

Model Prediction 
The network predicts two outputs:

heatmap 
A 3D heatmap representing the predicted center of the target region.

size 
The predicted bounding box dimensions in millimeters.

Heatmap Decoding 
The predicted center is obtained by locating the maximum activation in the heatmap.

The voxel coordinate of this location is then converted to world coordinates to obtain the predicted center in millimeters.

Bounding Box Construction 
The final bounding box is constructed using the predicted center and predicted size:

xmin = cx − width_x / 2 
ymin = cy − width_y / 2 
zmin = cz − width_z / 2 

xmax = cx + width_x / 2 
ymax = cy + width_y / 2 
zmax = cz + width_z / 2 

This bounding box represents the predicted spatial extent of the target anatomical region.

## Visualization

The repository provides utilities to visualize localization predictions directly on the medical scans.

Visualization helps qualitatively assess how well the predicted bounding box aligns with the ground truth annotation.

To visualize predictions for a specific case, run:

python scripts/visualize_case.py --index-csv data/processed/localizer_index.csv --checkpoint outputs/run01/best.pt --case 0

The visualization tool loads the scan, runs the trained model, and overlays the predicted and ground truth bounding boxes on the image.

The viewer displays three orthogonal views of the volume:

Axial view 
Horizontal slice of the volume showing the predicted bounding box on the XY plane.

Coronal view 
Frontal slice of the volume showing the bounding box on the XZ plane.

Sagittal view 
Side view slice showing the bounding box on the YZ plane.

In each view:

The solid rectangle represents the predicted bounding box. 
The dashed rectangle represents the ground truth bounding box.

This visualization allows quick inspection of localization accuracy and helps identify potential prediction errors.

## Model Architecture

The localization model is based on a **3D U-Net style encoder–decoder architecture** designed for volumetric medical images.

The network processes the entire 3D scan and predicts both the spatial center of the target region and its physical size.

Input

The model receives a normalized volumetric image:

(B, 1, Z, Y, X)

where

B = batch size 
Z, Y, X = spatial dimensions of the 3D volume 

Encoder

The encoder extracts hierarchical volumetric features through a series of convolutional blocks and downsampling operations.

Each encoder stage consists of:

3D convolution 
instance normalization 
leaky ReLU activation 

Spatial resolution is progressively reduced using strided convolutions while increasing the number of feature channels.

Decoder

The decoder reconstructs spatial information using transposed convolutions and skip connections from the encoder.

Skip connections allow the model to combine:

high-level semantic features from deeper layers 
fine spatial details from earlier layers 

This structure improves localization accuracy and stabilizes training.

Output Heads

The network produces two outputs.

Heatmap Head

A 3D convolution layer produces a heatmap representing the probability distribution of the target center.

Output shape:

(B, 1, Z, Y, X)

The predicted center is obtained by finding the voxel with the maximum heatmap activation.

Size Regression Head

Global average pooling is applied to the final decoder features followed by a fully connected layer.

This branch predicts the physical size of the bounding box:

(B, 3)

representing

[width_x, width_y, width_z]

in millimeters.

Bounding Box Prediction

The final bounding box is constructed from:

the decoded heatmap center 
the predicted bounding box size 

This allows the network to localize the anatomical region directly from the volumetric image.

## Metrics

The performance of the localization model is evaluated using several metrics that measure both the accuracy of the predicted center and the quality of the predicted bounding box.

Center Localization Error

The center localization error measures the Euclidean distance between the predicted center and the ground truth center.

center_error = ||pred_center − gt_center||

The error is expressed in millimeters and represents the spatial accuracy of the localization model.

Median Center Error

The median center error across the dataset is often reported as the main localization accuracy metric because it is robust to outliers.

P@20mm (Success Rate)

P@20mm measures the percentage of cases where the predicted center lies within a 20 millimeter radius of the ground truth center.

center_error ≤ 20 mm

This metric represents the practical success rate of the localization system.

Mean Intersection over Union (mIoU)

The predicted bounding box is compared with the ground truth bounding box using the Intersection-over-Union metric.

IoU = intersection_volume / union_volume

The mean IoU across the dataset measures how well the predicted spatial extent matches the ground truth region.

Together, these metrics provide a comprehensive evaluation of the localization performance in both position accuracy and bounding box quality.

## Tests

The repository includes a small set of unit tests to verify the correctness of key components of the localization pipeline.

Tests help ensure that important functionality such as dataset loading, coordinate transformations, and model behavior remain stable as the code evolves.

To run the tests, execute:

pytest

The following tests are included.

Dataset Shape Test

Verifies that the dataset loader returns tensors with the correct shapes for the input volume, heatmap target, and bounding box size.

Model Forward Test

Checks that the model forward pass runs correctly and produces outputs with the expected shapes.

Coordinate Transformation Test

Ensures that the conversion between voxel coordinates and world coordinates is consistent.

These tests provide basic validation of the core functionality of the framework.

## Future Work

Future extensions of this project will investigate multi-modality localization models that can operate across different medical imaging modalities.

As an initial step, modality-specific baseline models will be trained independently:

• a CT-only localization model 
• an MR-only localization model 

Each baseline model will use the same network architecture and training pipeline in order to evaluate localization performance under modality-specific conditions.

These baseline experiments will provide a reference point for analyzing differences between CT and MR localization performance.

Building on these baselines, future work will explore a unified multi-modality localization model capable of learning shared anatomical representations from both CT and MR scans within a single neural network.

Such a model could improve generalization across imaging modalities and enable more flexible medical imaging pipelines.

## License


## Acknowledgements

This project builds on ideas from prior work in medical image analysis and deep learning for volumetric data.

The implementation is inspired by research on 3D convolutional neural networks and U-Net style architectures for medical imaging tasks.

We also acknowledge the developers of the following open-source tools used in this project:

PyTorch 
A deep learning framework used for implementing and training the neural network.

SimpleITK 
A library for medical image processing and handling volumetric medical imaging formats.

NumPy 
A numerical computing library used for array operations and mathematical computations.

Matplotlib 
Used for visualization of model predictions and bounding boxes on medical images.

These tools and the broader open-source community make it possible to develop and share reproducible medical imaging research.
