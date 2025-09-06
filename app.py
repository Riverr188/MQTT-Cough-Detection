import os
import threading
import numpy as np
import tensorflow as tf
import base64
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt

# -----------------------------
# Config
# -----------------------------
BROKER = "mqtt-broker-production-dc66.up.railway.app"
PORT = 1883
TOPIC = "cough/audio"

USERNAME = None
PASSWORD = None

# -----------------------------
# Flask + SocketIO
# -----------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
# Load TFLite model
# -----------------------------
interpreter = tf.lite.Interpreter(model_path="tflite-model/tflite_learn_6.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
expected_length = input_details[0]['shape'][1]

labels = ["Anomalies", "COVID", "Healthy Cough"]

# -----------------------------
# HTML Templates
# -----------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Cough Detection</title>
  <meta http-equiv="refresh" content="2">
</head>
<body>
  <h2>🧪 Real-Time Cough Detection (MQTT)</h2>
  <p><strong>Latest Prediction:</strong> {{ label }} (confidence: {{ confidence }})</p>
</body>
</html>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Cough Detection API</title></head>
<body>
  <h3>✅ Flask API is running</h3>
  <ul>
    <li>Check last result at <a href='/latest'>/latest</a></li>
    <li>View live auto-refresh UI at <a href='/ui'>/ui</a></li>
  </ul>
</body>
</html>
"""

# -----------------------------
# Shared State
# -----------------------------
latest_prediction = {"label": "None", "confidence": 0.0}

# -----------------------------
# Inference
# -----------------------------
def run_inference(audio):
    """Run inference on one full audio buffer"""
    # Pad/trim to expected length
    if len(audio) < expected_length:
        pad = np.zeros(expected_length - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])
    else:
        audio = audio[:expected_length]

    input_tensor = np.expand_dims(audio, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    prediction = dict(zip(labels, map(float, output)))
    top_label = max(prediction, key=prediction.get)
    confidence = prediction[top_label]
    return {"label": top_label, "confidence": confidence}

# -----------------------------
# MQTT Callbacks
# -----------------------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to HiveMQ")
        client.subscribe(TOPIC)
    else:
        print(f"❌ MQTT failed with code {rc}")

def on_message(client, userdata, msg):
    global latest_prediction
    print(f"📩 Received {len(msg.payload)} bytes from {msg.topic}")
    try:
        raw_bytes = base64.b64decode(msg.payload)  # base64 PCM16
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        latest_prediction = run_inference(audio)

        # Broadcast to WebSocket
        socketio.emit("prediction", latest_prediction)

        print(f"🔮 Prediction: {latest_prediction}")
    except Exception as e:
        print("⚠️ Error processing audio:", e)

def start_mqtt():
    def run():
        client = mqtt.Client(protocol=mqtt.MQTTv311)  # force v3.1.1
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(BROKER, PORT, 60)
        client.loop_forever()

    t = threading.Thread(target=run)
    t.daemon = True
    t.start()

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/")
def index():
    return INDEX_TEMPLATE

@app.route("/latest")
def latest():
    return jsonify(latest_prediction)

@app.route("/ui")
def ui():
    return render_template_string(
        HTML_TEMPLATE,
        label=latest_prediction["label"],
        confidence=round(latest_prediction["confidence"], 3),
    )

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    start_mqtt()
    socketio.run(app, host="0.0.0.0", port=8080, debug=True)
