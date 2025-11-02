🩺 Medical X-Ray Pneumonia Detection (FastAPI + PyTorch)

A deep-learning based medical imaging project that classifies Chest X-Ray images into:

✅ Normal
✅ Pneumonia

Built using PyTorch, ResNet18, and FastAPI.
Includes training pipeline, inference API & cURL testing.

📂 Project Structure
medimg-project/
 └── src/
     ├── train.py          # Model training script
     ├── test.py           # Evaluation script
     ├── app.py            # FastAPI backend
     └── data/             # Dataset directory
 └── checkpoints/          # Saved model (.pth)

 🚀 Features
Component	Description
Model	ResNet-18 (Transfer Learning)
Accuracy	~73% currently (can be improved)
Framework	PyTorch
Inference	FastAPI REST API


Hardware	GPU Supported (CUDA)
📦 Installation
1️⃣ Clone Repo
git clone https://github.com/<your-username>/medimg-project.git

2️⃣ Create Virtual Environment
conda create -n medimg python=3.10 -y
conda activate medimg

3️⃣ Install Dependencies
pip install -r requirements.txt

📊 Training

Make sure dataset is placed inside:

/data/chest_xray/train
/data/chest_xray/test


Run training:

python train.py


Model saves to:

/checkpoints/model.pth

✅ Testing Model
python test.py

🌐 Running FastAPI Server
uvicorn app:app --reload


API URL:

http://127.0.0.1:8000/predict


Docs UI:

http://127.0.0.1:8000/docs

🧪 Test API with cURL
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_image.jpeg'


Expected Response

{
  "prediction": "PNEUMONIA"
}

📁 Dataset

Dataset used: Chest X-Ray Images (Pneumonia)

📝 Not included in repo due to size.

💡 Future Enhancements

✅ Improve accuracy (ResNet50 / EfficientNet)

✅ Add Streamlit UI

🐳 Docker Deployment

☁️ Deploy to AWS / GCP

👩‍⚕️ Disclaimer

This model is for learning & research only, not certified for clinical use.

👤 Author

Srivarshini Senthil Kumar



cd medimg-project/src
