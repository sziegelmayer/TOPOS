# TOPOS

**Target Organ Prediction of Scout Views for Automated CT Scan Planning**

TOPOS is a deep learning-based tool for anatomical region detection and localization from scout scans. It uses a trained nnUNet v2 model to segment 26 anatomical labels from 2D radiographic scout views.

This package provides a simple Python interface for inference and post-processing, and automatically downloads the model checkpoint from Hugging Face.

---

## Features

- Deep learning-based organ localization
- 26 anatomical structures supported
- Compatible with nnUNet v2 (2D configuration)
- Automatically downloads the pretrained checkpoint
- Outputs per-label `.nii.gz` segmentations

---

## Installation

# Create virtual environment
conda create -n topos python=3.11 pip
conda acitvate topos

# Install TOPOS
python -m pip install toposv
