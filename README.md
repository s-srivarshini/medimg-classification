# 🩺 Pneumonia Detection from Chest X-Ray Images

A deep learning system to classify **Normal vs Pneumonia** chest X-rays using **ResNet-18** and served via **FastAPI**.

> ⚠️ Research & educational use only — **not for medical diagnosis**

---

## 📌 1. Project Overview

This project demonstrates:

- ✅ Transfer Learning (ResNet-18)
- ✅ GPU / CUDA support
- ✅ FastAPI inference API
- ✅ Image file upload prediction
- ✅ Clean training + deployment workflow

---

## 🛠️ 2. Tech Stack

| Category | Tools |
|---|---|
Model | ResNet-18 (PyTorch)
API Framework | FastAPI + Uvicorn
Dataset | Kaggle Chest X-Ray Pneumonia Dataset
Environment | Conda + Python 3.10
Hardware | CPU / NVIDIA GPU

---

## 📁 3. Project Structure

medimg-project/
│── checkpoints/ # Saved model weights
│── data/ # Dataset (not included)
└── src/
├── app.py # FastAPI service
├── train.py # Model training
├── test.py # Model evaluation
└── requirements.txt


---

## ⚙️ 4. Setup Instructions

### ✅ Clone the Repo


git clone https://github.com/<your-username>/medimg-project.git
cd medimg-project/src

### ✅ Create Conda Environment
conda create -n medimg python=3.10 -y
conda activate medimg

### ✅ Install Requirements
pip install -r requirements.txt

## 🗂️ 5. Dataset Structure

Download from Kaggle and arrange like:

data/chest_xray/
├── train/
├── val/
└── test/

## 🏋️‍♂️ 6. Train the Model
python train.py


Model saves to:

checkpoints/model.pth

## 🎯 7. Test the Model
python test.py

## 🚀 8. Run the FastAPI Server
uvicorn app:app --reload


Open docs:

http://localhost:8000/docs

## 🧪 9. Make Prediction (cURL)
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@xray_image.jpg"


Example output:

{
  "prediction": "PNEUMONIA"
}


✅ Create Conda Environmen

t
