from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ---------------------- YOLO (UNCHANGED) ----------------------
model_yolo = YOLO("yolov8m.pt")

vehicle_classes = ['car','motorcycle','bus','truck','bicycle']

# ---------------------- FEDERATED SETUP ----------------------
EDGE_NODES = 3
history_files = [f"traffic_node_{i}.csv" for i in range(EDGE_NODES)]

time_window = 5
lstm_epochs = 3

def create_lstm_model():
    model = Sequential([
        LSTM(32, input_shape=(time_window, 1), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.01), loss='mse')
    return model

global_model = create_lstm_model()

# ---------------------- DETECTION (DO NOT TOUCH) ----------------------
def detect_vehicles(frame):
    results = model_yolo.predict(frame, conf=0.15, iou=0.45, imgsz=1280, verbose=False)

    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model_yolo.names[cls]

            if name in vehicle_classes:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return count, frame

# ---------------------- UPDATE HISTORY ----------------------
def update_history(node_id, count):
    file = history_files[node_id]

    if os.path.exists(file):
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=['Vehicle_Count'])

    df = pd.concat([df, pd.DataFrame({'Vehicle_Count':[count]})], ignore_index=True)
    df.to_csv(file, index=False)

# ---------------------- LOCAL TRAIN ----------------------
def train_local_model(node_id):
    file = history_files[node_id]

    if not os.path.exists(file):
        return None

    df = pd.read_csv(file)
    values = df['Vehicle_Count'].values

    if len(values) < time_window + 1:
        return None

    X, y = [], []
    for i in range(len(values) - time_window):
        X.append(values[i:i+time_window])
        y.append(values[i+time_window])

    X = np.array(X).reshape(-1, time_window, 1)
    y = np.array(y)

    model = create_lstm_model()
    model.fit(X, y, epochs=lstm_epochs, verbose=0)

    return model.get_weights()

# ---------------------- FEDERATED AVERAGING ----------------------
def federated_averaging(weights_list):
    avg_weights = []
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

# ---------------------- PREDICT ----------------------
def predict_traffic(node_id=0):
    local_weights = []

    for i in range(EDGE_NODES):
        w = train_local_model(i)
        if w is not None:
            local_weights.append(w)

    # fallback
    if not local_weights:
        file = history_files[node_id]
        if os.path.exists(file):
            df = pd.read_csv(file)
            if len(df) > 0:
                return calculate_signal_time(df['Vehicle_Count'].values[-1])
        return 10

    # federated learning
    avg_weights = federated_averaging(local_weights)
    global_model.set_weights(avg_weights)

    df = pd.read_csv(history_files[node_id])
    values = df['Vehicle_Count'].values

    if len(values) < time_window:
        return calculate_signal_time(values[-1])

    data = values[-time_window:].reshape(1, time_window, 1)

    pred = int(global_model.predict(data, verbose=0)[0][0])
    pred = max(pred, 1)

    current = values[-1]

    final_count = int(0.6 * current + 0.4 * pred)

    return calculate_signal_time(final_count)

# ---------------------- SIGNAL ----------------------
def calculate_signal_time(count):
    if count <= 10:
        return 10
    elif count <= 20:
        return 20
    elif count <= 30:
        return 30
    elif count <= 40:
        return 40
    else:
        return 50

# ---------------------- ROUTES ----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files['file']
    if not file:
        return redirect(url_for('index'))

    filename = file.filename
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"processed_{filename}")

    file.save(upload_path)

    frame = cv2.imread(upload_path)
    count, processed_frame = detect_vehicles(frame)

    cv2.imwrite(output_path, processed_frame)

    node_id = 0
    update_history(node_id, count)

    signal_time = predict_traffic(node_id)

    return render_template('result.html',
                           count=count,
                           signal_time=signal_time,
                           file_processed=f"processed_{filename}")

# ---------------------- WEBCAM ----------------------
def gen_frames():
    cap = cv2.VideoCapture(0)

    last_update = time.time()
    current_signal = 10

    while True:
        success, frame = cap.read()
        if not success:
            break

        count, processed_frame = detect_vehicles(frame)

        node_id = np.random.randint(0, EDGE_NODES)
        update_history(node_id, count)

        if time.time() - last_update > 3:
            current_signal = predict_traffic(node_id)
            last_update = time.time()

        cv2.putText(processed_frame, f"Vehicles: {count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(processed_frame, f"Signal: {current_signal}s", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------- RUN ----------------------
if __name__ == "__main__":
    app.run(debug=True)
