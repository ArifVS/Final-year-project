# edge_federated_traffic.py
"""
Edge-Based Federated Learning for Real-Time Traffic Prediction
Supports image, video and webcam inputs. Uses YOLOv8 (ultralytics) for detection,
Keras LSTM for lightweight prediction on each edge, and federated averaging
(across Keras model weights) every aggregation_interval seconds.
Saves per-intersection CSVs with vehicle counts.
"""

import os
import cv2
import time
import threading
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# ---------------------------
# Configuration
# ---------------------------
vehicle_classes = {'car', 'motorbike', 'bus', 'truck', 'auto'}  # names expected from YOLO
time_window = 10               # LSTM input window (timesteps)
lstm_epochs_per_train = 3      # local training epochs (kept small for demo)
aggregation_interval = 30      # seconds between federated aggregation
model_path_yolo = "yolov8n.pt" # ensure this exists or change to available model
CONF_THRESHOLD = 0.25          # detection confidence threshold

# Load YOLO once
model_yolo = YOLO(model_path_yolo)

# ---------------------------
# LSTM Model factory
# ---------------------------
def create_lstm_model(input_shape=(time_window, 1)):
    model = Sequential([
        LSTM(32, input_shape=input_shape, activation='tanh'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(0.005), loss='mse')
    return model

# ---------------------------
# Vehicle Detection (with boxes)
# ---------------------------
def detect_vehicles(frame):
    """
    Uses ultralytics YOLO model to detect objects in a single BGR frame (numpy array).
    Returns (vehicle_count, annotated_frame).
    """
    # Ultralytics expects BGR or RGB depending; passing numpy array is supported
    results = model_yolo.predict(frame, verbose=False)  # returns Results object list
    count = 0
    annotated = frame.copy()
    if len(results) == 0:
        return 0, annotated

    r = results[0]  # for a single frame
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return 0, annotated

    # boxes.xyxy, boxes.conf, boxes.cls are available as tensors/arrays
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)

    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
        if conf < CONF_THRESHOLD:
            continue
        name = r.names.get(int(cls), str(cls))
        if name in vehicle_classes:
            count += 1
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{name} {conf:.2f}", (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return count, annotated

# ---------------------------
# Edge Intersection
# ---------------------------
class EdgeIntersection:
    def __init__(self, name, input_type, source):
        """
        input_type: 1=image, 2=video, 3=webcam
        source: path or webcam index (string or int)
        """
        self.name = name
        self.input_type = input_type
        self.source = source
        self.lstm_model = create_lstm_model()
        self.vehicle_history_full = []   # keep full history for training
        self.recent_window = []          # keep recent window for prediction (max time_window*4)
        self.local_data = []             # for CSV save
        self.lock = threading.Lock()
        self.running = True

        # set up capture or image
        if self.input_type == 1:
            if not os.path.exists(source):
                raise FileNotFoundError(f"Image not found: {source}")
            self.single_frame = cv2.imread(source)
            self.cap = None
        else:
            # video path or webcam index
            try:
                idx = int(source)
                self.cap = cv2.VideoCapture(idx)
            except Exception:
                self.cap = cv2.VideoCapture(source)

        # background training thread control
        self._train_thread = None

    def start(self):
        # start video/image processing in current thread (or separately if you want)
        if self.input_type == 1:
            self.process_frame_loop(single=True)
        else:
            self.process_frame_loop(single=False)

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        # wait for trainer thread
        if self._train_thread and self._train_thread.is_alive():
            self._train_thread.join(timeout=2)

    def process_frame_loop(self, single=False):
        if single:
            self.process_frame(self.single_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.save_data()
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # video finished or webcam disconnected
                break
            self.process_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # stop if user presses 'q'
                self.running = False
                break
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.save_data()

    def process_frame(self, frame):
        # detect vehicles and annotate
        vehicle_count, annotated = detect_vehicles(frame)

        # update histories
        with self.lock:
            self.vehicle_history_full.append(vehicle_count)
            self.recent_window.append(vehicle_count)
            self.local_data.append(vehicle_count)
            # keep recent window bounded (avoid unlimited growth)
            if len(self.recent_window) > max(time_window * 8, 200):
                self.recent_window.pop(0)

        # start training in background if enough samples
        if len(self.vehicle_history_full) >= time_window + 1:
            if not (self._train_thread and self._train_thread.is_alive()):
                self._train_thread = threading.Thread(target=self.train_local_model, daemon=True)
                self._train_thread.start()

        # predict next value using LSTM (if possible)
        pred = self.predict_traffic()

        # compute signal time based on current real-time vehicle_count and predicted value
        signal_time = self.calculate_signal_time(vehicle_count, predicted_next=pred)

        # overlay info and show
        cv2.putText(annotated,
                    f"{self.name} Cur={vehicle_count} Pred={pred} Sig={signal_time}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(self.name, annotated)

    def train_local_model(self):
        """
        Build dataset from vehicle_history_full and train LSTM.
        We form sliding windows of length `time_window` -> next value.
        """
        with self.lock:
            series = list(self.vehicle_history_full)  # copy

        # create windows
        X, y = [], []
        for i in range(len(series) - time_window):
            X.append(series[i:i+time_window])
            y.append(series[i+time_window])
        if len(X) == 0:
            return

        X = np.array(X, dtype=np.float32).reshape(-1, time_window, 1)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        # tiny training for demo
        try:
            self.lstm_model.fit(X, y, epochs=lstm_epochs_per_train, verbose=0)
            print(f"[{self.name}] Trained locally on {len(X)} samples.")
        except Exception as e:
            print(f"[{self.name}] Training error: {e}")

    def predict_traffic(self):
        with self.lock:
            if len(self.recent_window) < time_window:
                return int(self.recent_window[-1]) if self.recent_window else 0
            inp = np.array(self.recent_window[-time_window:], dtype=np.float32).reshape(1, time_window, 1)

        try:
            p = self.lstm_model.predict(inp, verbose=0)
            return int(max(0, round(float(p[0][0]))))
        except Exception:
            return int(self.recent_window[-1])

    def calculate_signal_time(self, vehicle_count, predicted_next=None):
        """
        Rule-based mapping. You can replace with RL later.
        Here we combine current count and prediction a bit for smoother behavior.
        """
        if predicted_next is None:
            predicted_next = vehicle_count
        combined = 0.6 * vehicle_count + 0.4 * predicted_next

        if combined <= 5:
            return 10
        elif combined <= 10:
            return 20
        elif combined <= 20:
            return 30
        elif combined <= 30:
            return 40
        else:
            return 50

    def save_data(self):
        # save local_data to CSV
        if len(self.local_data) == 0:
            return
        df = pd.DataFrame({'Vehicle_Count': self.local_data})
        fname = f"{self.name}_traffic.csv"
        df.to_csv(fname, index=False)
        print(f"✅ {self.name} data saved -> {fname}")

# ---------------------------
# Federated Aggregation
# ---------------------------
def federated_aggregate(models):
    """
    models: list of Keras model instances (same architecture).
    Perform simple average of weights (FedAvg).
    """
    if not models:
        return
    # collect weight lists
    weights_list = [m.get_weights() for m in models]
    # average each weight tensor across models
    new_weights = []
    for weights_tuple in zip(*weights_list):
        new_weights.append(np.mean(np.array(weights_tuple), axis=0))
    # set averaged weights to each model
    for m in models:
        m.set_weights(new_weights)
    print("⚡ Federated aggregation completed.")

# ---------------------------
# Main orchestration
# ---------------------------
def main():
    intersections = []
    try:
        num = int(input("Enter number of intersections: ").strip())
    except Exception:
        print("Invalid number, using 1 intersection.")
        num = 1

    for i in range(num):
        print(f"\nIntersection {i+1}: Choose input type:\n1. Image\n2. Video\n3. Webcam")
        try:
            input_type = int(input("Enter 1, 2, or 3: ").strip())
        except Exception:
            input_type = 3
        if input_type == 1:
            source = input("Enter image path: ").strip()
        else:
            source = input("Enter video path or webcam index (0 for default): ").strip()
        ei = EdgeIntersection(f"Edge-{i+1}", input_type, source)
        intersections.append(ei)

    # start each intersection in its own thread
    threads = []
    for inter in intersections:
        t = threading.Thread(target=inter.start, daemon=True)
        t.start()
        threads.append(t)

    # federated server thread
    def federated_server():
        while True:
            time.sleep(aggregation_interval)
            print(">>> Aggregation triggered...")
            # clone model architecture for safe averaging (use copies if needed)
            models = [inter.lstm_model for inter in intersections if inter is not None]
            try:
                federated_aggregate(models)
            except Exception as e:
                print("Aggregation error:", e)

    agg_thread = threading.Thread(target=federated_server, daemon=True)
    agg_thread.start()

    try:
        # join on UI threads (they are daemonic; allow Ctrl+C to stop)
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for inter in intersections:
            inter.stop()
        print("All stopped.")

if __name__ == "__main__":
    main()
