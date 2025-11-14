import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import time
import re

# Try import serial
try:
    import serial
    SERIAL_AVAILABLE = True
except:
    SERIAL_AVAILABLE = False

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(
    page_title="Fruit Ripeness & IoT Monitor",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Fruit Ripeness & Type Detector")

CSV_FILE = 'fruit_analysis_results.csv'

# -------------------- SESSION STATE -------------------- #
if "results_history" not in st.session_state:
    if os.path.exists(CSV_FILE):
        st.session_state.results_history = pd.read_csv(CSV_FILE).to_dict("records")
    else:
        st.session_state.results_history = []

if "result_counter" not in st.session_state:
    st.session_state.result_counter = len(st.session_state.results_history)

# -------------------- LOAD MODEL -------------------- #
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "fruit_ripeness_multitask_finetuned.keras")
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()
if model is None:
    st.stop()

RIPENESS_CLASSES = ["Unripe", "Ripe", "Overripe"]
FRUIT_TYPES = ["Apple", "Orange"]

# -------------------- IMAGE PROCESSING -------------------- #
def preprocess_image(image_pil):
    try:
        img = np.array(image_pil)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype("float32"))
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

# -------------------- PREDICTION -------------------- #
def get_prediction(image_pil):
    try:
        processed = preprocess_image(image_pil)
        if processed is None:
            return None
        pred = model.predict(processed, verbose=0)
        ripeness_probs = pred[0][0]
        fruit_probs = pred[1][0]
        ripeness_idx = int(np.argmax(ripeness_probs))
        fruit_idx = int(np.argmax(fruit_probs))
        return {
            "ripeness": RIPENESS_CLASSES[ripeness_idx],
            "ripeness_conf": float(ripeness_probs[ripeness_idx]),
            "fruit": FRUIT_TYPES[fruit_idx],
            "fruit_conf": float(fruit_probs[fruit_idx]),
            "ripeness_probs": ripeness_probs,
            "fruit_probs": fruit_probs
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# -------------------- SAVE RESULT -------------------- #
def save_result_to_csv(result, source):
    st.session_state.result_counter += 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "ID": st.session_state.result_counter,
        "Timestamp": timestamp,
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Source": source,
        "Fruit_Type": result["fruit"],
        "Fruit_Confidence": round(result["fruit_conf"] * 100, 2),
        "Ripeness": result["ripeness"],
        "Ripeness_Confidence": round(result["ripeness_conf"] * 100, 2),
    }
    st.session_state.results_history.append(entry)
    pd.DataFrame(st.session_state.results_history).to_csv(CSV_FILE, index=False)

# -------------------- FINAL SHELF LIFE LOGIC -------------------- #
def combined_shelf_life(ripeness, temp, hum):
    # Base shelf life (before IoT temp/humidity adjustment)
    base = {
        "Unripe": (5, 7),
        "Ripe": (2, 3),
        "Overripe": (0, 0),
    }

    low, high = base[ripeness]

    # Overripe always zero days
    if ripeness == "Overripe":
        return "0 days — fruit is already overripe"

    # If sensor not available → return base days
    if temp is None or hum is None:
        return f"{low}-{high} days"

    # Environment multiplier
    if temp > 30:  # Hot
        factor = 0.6
    elif 20 <= temp <= 30 and 50 <= hum <= 80:  # Ideal
        factor = 1.0
    elif temp < 20 and 55 <= hum <= 65:  # Cool stable
        factor = 1.2
    else:
        factor = 0.8  # Slightly reduced shelf life

    # Apply factor and convert to whole days
    adj_low = round(low * factor)
    adj_high = round(high * factor)

    # Make sure values stay positive
    adj_low = max(0, adj_low)
    adj_high = max(0, adj_high)

    return f"{adj_low}-{adj_high} days"

# -------------------- TABS -------------------- #
tab1, tab2, tab3, tab4 = st.tabs(
    ["📷 Camera", "📁 Upload", "📋 Results", "📡Shelf Life"]
)

# -------------------- CAMERA TAB -------------------- #
with tab1:
    st.header("📷 Take a Photo")
    picture = st.camera_input("Point camera at fruit")

    if picture:
        img = Image.open(picture)
        st.info("Click Analyze.")

        if st.button("Analyze (Camera)"):
            result = get_prediction(img)
            if result:
                save_result_to_csv(result, "Camera")
                st.session_state["last_ripeness"] = result["ripeness"]
                st.session_state["last_fruit_type"] = result["fruit"]

                st.success("Prediction Saved!")
                st.metric("Ripeness", result["ripeness"])
                st.metric("Fruit Type", result["fruit"])

# -------------------- UPLOAD TAB -------------------- #
with tab2:
    st.header("📁 Upload Image")
    uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)

        if st.button("Analyze (Upload)"):
            result = get_prediction(img)
            if result:
                save_result_to_csv(result, "Upload")
                st.session_state["last_ripeness"] = result["ripeness"]
                st.session_state["last_fruit_type"] = result["fruit"]

                st.success("Prediction Saved!")
                st.metric("Ripeness", result["ripeness"])
                st.metric("Fruit Type", result["fruit"])

# -------------------- RESULTS TAB -------------------- #
with tab3:
    st.header("📋 Recent Results")
    if len(st.session_state.results_history) > 0:
        st.dataframe(
            pd.DataFrame(st.session_state.results_history).tail(10),
            use_container_width=True,
        )
    else:
        st.info("No results yet.")

# -------------------- IOT FUNCTIONS -------------------- #
def read_clean_sensor(esp):
    for _ in range(20):
        raw = esp.readline().decode("ascii", errors="ignore").strip()
        #st.write("RAW:", repr(raw))

        if raw.lower().startswith("ets"):
            continue

        if "Humidity" in raw and "Temperature" in raw:
            return raw

    return None

def parse_temp_hum(line):
    try:
        clean = line.replace("°", "").replace("%", "")
        clean = clean.replace("C", "").replace("c", "")
        clean = clean.strip()
        parts = clean.split()

        hum = None
        temp = None

        for i in range(len(parts)):
            if parts[i].lower().startswith("humidity:"):
                hum = float(parts[i + 1])
            if parts[i].lower().startswith("temperature:"):
                temp = float(parts[i + 1])

        return temp, hum

    except:
        return None, None

# -------------------- IOT TAB -------------------- #
with tab4:
    st.header("Shelf life")

    if st.button("Read Sensor + Calculate Shelf Life"):
        if not SERIAL_AVAILABLE:
            st.error("PySerial not installed")
        elif "last_ripeness" not in st.session_state:
            st.warning("Upload/Analyze a fruit image first!")
        else:
            try:
                esp = serial.Serial("COM5", 115200, timeout=2)
                time.sleep(2)

                raw = read_clean_sensor(esp)

                if raw is None:
                    st.error("No valid sensor data received.")
                else:
                    temp, hum = parse_temp_hum(raw)

                    st.metric(" Temperature (°C)", temp if temp else "N/A")
                    st.metric(" Humidity (%)", hum if hum else "N/A")

                    ripeness = st.session_state["last_ripeness"]
                    shelf = combined_shelf_life(ripeness, temp, hum)

                    st.success(f" Shelf Life (Based on {ripeness} + IoT): {shelf}")

                esp.close()

            except Exception as e:
                st.error(f"Serial error: {e}")

st.markdown("---")
st.markdown("Made with ❤️ using TensorFlow, Streamlit & ESP32 IoT")
