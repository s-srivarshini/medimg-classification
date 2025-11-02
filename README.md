🩺 Medical X-Ray Pneumonia Detection (Deep Learning + FastAPI)

A deep learning project to classify Chest X-Ray images as Normal or Pneumonia using PyTorch and serve predictions through a FastAPI inference API.

This project demonstrates end-to-end AI system design:

✅ Data preprocessing
✅ CNN model training (ResNet-18)
✅ GPU training support
✅ Model evaluation
✅ REST API for real-time predictions
✅ Curl & Swagger UI testing

📂 Project Structure
medimg-project/
│
├── data/
│   └── chest_xray/ (dataset)
│
├── src/
│   ├── train.py          # Train model
│   ├── test.py           # Evaluate model
│   └── app.py            # FastAPI inference server
│
├── checkpoints/          # Saved model weights
├── requirements.txt
└── README.md

🧠 Model

Architecture: ResNet-18 (Transfer Learning)

Framework: PyTorch

Classes: NORMAL, PNEUMONIA

Evaluation: Accuracy & loss on validation set

🚀 Training

To train the model:

cd src
python train.py


Training auto-detects GPU if available.

✅ Testing Model
cd src
python test.py

🌐 Run FastAPI Server
cd src
uvicorn app:app --reload

🧪 API Usage
✅ Swagger UI

Open in browser:

http://127.0.0.1:8000/docs

✅ cURL Testing
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@YOUR_IMAGE.jpeg;type=image/jpeg'

✅ Example Response
{
  "prediction": "PNEUMONIA"
}

📦 Dependencies

Install:

pip install -r requirements.txt

Key Libraries
Library	Purpose
torch, torchvision	Deep Learning
FastAPI, uvicorn	API backend
Pillow	Image handling
📊 Results
Metric	Value
Training Accuracy	~99%
Test Accuracy	~73%

Note: Real performance improves with more data augmentation & deeper models (ResNet50/EfficientNet).

🎯 Future Enhancements

✅ Deploy on cloud (EC2 / Render / Railways)

✅ Docker support

⏳ Streamlit UI for medical image upload

⏳ Explainability (Grad-CAM heatmaps)

⏳ Model upgrade to EfficientNet

🤝 Contributing

Pull requests are welcome!
