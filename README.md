Pneumonia Detection from Chest X-Ray Images

A deep learning pipeline to classify chest X-ray images into Normal or Pneumonia using ResNet-18 and served via a FastAPI inference API.

⚠️ For research and learning purposes only. Not for medical use.

1. Project Overview

Chest X-ray imaging is a key tool for diagnosing pneumonia.
This project demonstrates:

Transfer learning with ResNet-18

GPU-enabled model training (PyTorch)

API based inference with FastAPI

File-upload based prediction endpoint

Local reproducibility and deployment readiness

2. Technology Stack
Category	Tools
Model	ResNet-18 (PyTorch)
API Framework	FastAPI + Uvicorn
Dataset	Kaggle Chest X-Ray Pneumonia Dataset
Environment	Conda + Python 3.10
Hardware	CPU / NVIDIA GPU supported
3. Project Structure
medimg-project/
│── checkpoints/            # Saved model weights
│── data/                   # Chest X-ray dataset (excluded from repo)
└── src/
    ├── app.py              # FastAPI service
    ├── train.py            # Training script
    ├── test.py             # Evaluation script
    └── requirements.txt

4. Setup Instructions
Clone repository
```bash
git clone https://github.com/<your-username>/medimg-project.git
cd medimg-project/src

Create Conda environment
```bash
conda create -n medimg python=3.10 -y
conda activate medimg

Install dependencies
```bash
pip install -r requirements.txt

5. Dataset Structure

Download the dataset from Kaggle and arrange like:

data/chest_xray/train/
data/chest_xray/test/

6. Train the Model
python train.py


The trained model is saved to:

checkpoints/model.pth

7. Evaluate the Model
python test.py

8. Run FastAPI Service
uvicorn app:app --reload


API Docs:

http://localhost:8000/docs
