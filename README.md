# Image Analysis Assignment（LU） --by Python

This repository contains the course assignments for the **Image Analysis** course. All tasks were implemented in **Python**.

## ⚙️ Environment Setup

You can choose **either pip or conda**, depending on your setup.

### Option A — Conda

**Recommended** if you use Anaconda.

```
conda env create -f environment.yml
conda activate image-analysis
```

### Option B — pip (GPU version, CUDA 12.9)

For NVIDIA GPU users.
Uses the PyTorch CUDA build specified in requirements.txt.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🧩 Assignment 1: 4.OCR

In assignment1, two segmentation approaches were implemented:

A traditional method (e.g., Otsu thresholding and morphological processing)

A U-Net–based model trained on the M2NIST dataset

The trained U-Net weights are saved as unet_m2nist.pth.
If you wish to switch between segmentation methods, simply modify the call to im2segment in assignment1.py to change the source of segmentation (traditional vs. U-Net).

## 🧩 Assignment 4: task1 setup

In assignment4, for task1, you must first clone the WB_sRGB project into the task1 directory before running the code.

Example:

```
cd assignment4/task1
git clone https://github.com/mahmoudnafifi/WB_sRGB.git
```


Make sure the WB_sRGB folder is placed directly inside task1 so that the code can correctly import its modules.

## 📄 Reports

Each folder includes a short report (.pdf) summarizing the methodology and results.