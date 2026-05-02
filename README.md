## 📊 Project Details

| PB No | Title of the Project | Problem Statement | Abstract | Domain |
|------|---------------------|------------------|----------|--------|
| PB-NO-188 | Edge-Based Federated Learning for Real-Time Traffic Prediction in Smart Cities | Centralized machine learning models for urban traffic prediction often require huge amounts of data to be collected and processed in cloud servers, raising concerns about data privacy, latency, and network bandwidth usage. | This project implements a federated learning framework that performs traffic flow prediction on edge devices deployed at various traffic intersections in a smart city environment. Instead of sending raw video or sensor data to the cloud, each device locally trains a lightweight LSTM or CNN model using real-time traffic data. The local models share only their updated parameters with a central server, which aggregates them to create a global model without compromising data privacy. The system predicts congestion trends, optimizes traffic signal timing, and minimizes latency in real-time decision-making. This privacy-preserving, scalable approach represents the next generation of intelligent urban infrastructure. | Federated Learning, Edge Computing, Smart Cities, Artificial Intelligence |





[README.md](https://github.com/user-attachments/files/26377634/README.md)
# 🚦 Smart Traffic Management System
### Federated Learning + YOLOv8 + LSTM | Final Year Project

A real-time intelligent traffic management system that uses **YOLOv8** for vehicle detection and **Federated Learning with LSTM** for adaptive signal timing prediction.

---

## 🧠 Features

- 🔍 **Vehicle Detection** — YOLOv8m detects cars, motorcycles, buses, trucks, and bicycles
- 📊 **Real-time Metrics** — Precision, Recall, F1-score, and Accuracy per frame
- 🧬 **Federated LSTM** — Distributed edge-node learning for traffic prediction
- 🟢 **Congestion Levels** — LOW / MEDIUM / HIGH / CRITICAL with adaptive signal timing
- 📷 **Image Upload** — Upload a traffic image for instant analysis
- 🎥 **Live Webcam** — Real-time detection via webcam feed

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download YOLOv8 model
```bash
# Automatically downloaded on first run, or manually:
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### 3. Run the Flask app
```bash
python demo.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
ArifVS-clone/
├── demo.py              # Main Flask application
├── app.py               # Alternate app entry
├── templates/
│   ├── index.html       # Upload page
│   ├── result.html      # Detection results
│   └── webcam.html      # Live webcam feed
├── static/
│   ├── uploads/         # Uploaded images
│   └── outputs/         # Processed images
└── requirements.txt
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask (Python) |
| Detection | YOLOv8 (Ultralytics) |
| Prediction | LSTM (TensorFlow/Keras) |
| Learning | Federated Averaging |
| Frontend | HTML, CSS, JavaScript |

---

## 👨‍💻 Author

**Final Year Project** — Alliance University  
*Smart Traffic Management using Federated Learning*
