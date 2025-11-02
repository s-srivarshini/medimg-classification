🩺 Pneumonia Detection from Chest X-Ray Images

This project uses Deep Learning (ResNet-18) and FastAPI to classify chest X-ray images into:

Normal

Pneumonia

Designed for learning, research, and demonstration of AI in medical imaging.

📁 Repository Structure
medimg-project/
│── checkpoints/          # Saved model weights
│── data/                 # Dataset (not included in repo)
└── src/
    ├── train.py          # Model training script
    ├── test.py           # Evaluation script
    ├── app.py            # FastAPI application
    └── requirements.txt

✅ Features
Feature	Details
Model	ResNet-18 (Transfer Learning)
Framework	PyTorch + FastAPI
Inference	REST API for image upload & prediction
GPU Support	Yes (CUDA enabled)
Use Case	Pneumonia detection from chest X-rays
📦 Setup
1️⃣ Clone the repository
git clone https://github.com/<your-username>/medimg-project.git
cd medimg-project/src

2️⃣ Create & activate environment
conda create -n medimg python=3.10 -y
conda activate medimg

3️⃣ Install dependencies
pip install -r requirements.txt

🧠 Training the Model

Place the dataset like:

data/chest_xray/train
data/chest_xray/test


Run training:

python train.py


This creates:

checkpoints/model.pth

📊 Evaluate Model
python test.py

🚀 Run the FastAPI Server
uvicorn app:app --reload


API docs:

http://127.0.0.1:8000/docs

🧪 Test the API (cURL Example)
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpeg"


Example response:

{
  "prediction": "PNEUMONIA"
}

📂 Dataset Used

Chest X-Ray Images (Pneumonia) — Kaggle dataset
Dataset not included due to size.
