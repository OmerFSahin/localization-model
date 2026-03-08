# 3D Medical Image Localization

A PyTorch implementation of a 3D localization model for detecting a target anatomical region in volumetric medical scans.

The model predicts:

• a 3D heatmap indicating the target center 
• the bounding box size in millimeters 

from a full 3D medical image volume.

The architecture is based on a 3D U-Net style encoder–decoder network trained with Gaussian heatmap targets.

FEATURES

• 3D U-Net localization architecture 
• Gaussian heatmap target generation 
• bounding box size regression 
• SimpleITK-based medical image pipeline 
• dataset indexing and validation tools 
• visualization utilities for predictions 
• modular PyTorch training pipeline 

REPOSITORY STRUCTURE

localization-model/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml
├─ configs/
│  ├─ data.yaml
│  ├─ train.yaml
│  └─ infer.yaml
├─ src/
│  └─ localization/
│     ├─ data/
│     ├─ transforms/
│     ├─ geometry/
│     ├─ targets/
│     ├─ models/
│     ├─ train/
│     ├─ eval/
│     ├─ inference/
│     └─ viz/
├─ scripts/
│  ├─ make_index.py
│  ├─ check_dataset.py
│  ├─ debug_dataset.py
│  ├─ train.py
│  ├─ eval.py
│  └─ visualize_case.py
├─ data/
│  ├─ samples/
│  ├─ raw/
│  └─ processed/
├─ outputs/
├─ tests/
└─ docs/
   ├─ dataset_format.md
   ├─ training.md
   └─ inference.md

INSTALLATION

Clone the repository:

git clone https://github.com/yourusername/localization-model.git
cd localization-model

Install dependencies:

pip install -r requirements.txt

or with editable install:

pip install -e .

DATASET

Each case contains:

• a 3D scan 
• a bounding box annotation 

Example structure:

data/processed/
   case_0001/
      scan.nrrd
      meta.json

Bounding box format in meta.json:

{
  "bbox_mm": [xmin, ymin, zmin, xmax, ymax, zmax]
}

Coordinates are expressed in millimeters (world coordinate system).

Full specification is described in:

docs/dataset_format.md

CREATE DATASET INDEX

Generate the dataset split file:

python scripts/make_index.py

This creates:

data/processed/localizer_index.csv

TRAINING

Run training using:

python scripts/train.py --index-csv data/processed/localizer_index.csv --outdir outputs/run01 --epochs 50 --lr 1e-4

Training outputs are saved to:

outputs/run01/

Typical files:

best.pt 
last.pt 
history.json 

More details:

docs/training.md

EVALUATION

Evaluate a trained model:

python scripts/eval.py --index-csv data/processed/localizer_index.csv --checkpoint outputs/run01/best.pt

Metrics include:

• center localization error 
• success rate (P@20mm) 
• bounding box IoU 

VISUALIZATION

Visualize predicted bounding boxes:

python scripts/visualize_case.py --index-csv data/processed/localizer_index.csv --checkpoint outputs/run01/best.pt --case 0

The viewer displays:

• axial slice 
• coronal slice 
• sagittal slice 

with predicted and ground truth bounding boxes.

MODEL

The localization model is a 3D U-Net style architecture.

Input tensor:

(B, 1, Z, Y, X)

Outputs:

heatmap : (B,1,Z,Y,X) 
size    : (B,3)

The heatmap represents the predicted center location of the target region.

TESTING

Run unit tests with:

pytest

Included tests:

• dataset shape validation 
• model forward pass 
• coordinate transformation correctness 

LICENSE



ACKNOWLEDGEMENTS

This repository builds on ideas from:

• 3D U-Net architectures for volumetric segmentation 
• heatmap-based object localization methods 
• medical image processing with SimpleITK
