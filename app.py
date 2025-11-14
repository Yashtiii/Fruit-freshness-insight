# app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from datetime import datetime
import os
from ultralytics import YOLO


# Page config
st.set_page_config(
    page_title="Fruit Ripeness Detector",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Fruit Ripeness & Type Detector")

# Define CSV file path
CSV_FILE = 'fruit_analysis_results.csv'

# Initialize session state
if 'results_history' not in st.session_state:
    if os.path.exists(CSV_FILE):
        st.session_state.results_history = pd.read_csv(CSV_FILE).to_dict('records')
    else:
        st.session_state.results_history = []

if 'result_counter' not in st.session_state:
    st.session_state.result_counter = len(st.session_state.results_history)


# Load models
@st.cache_resource
def load_models():
    """Load both YOLO and ripeness classification models"""
    # Use YOLOv8 MEDIUM instead of nano - better accuracy
    yolo_model = YOLO('yolov8m.pt')  # Changed from yolov8n.pt to yolov8m.pt
    
    # Load your ripeness classification model
    ripeness_model_path = 'fruit_ripeness_multitask_finetuned.keras'
    if not os.path.exists(ripeness_model_path):
        st.error(f"❌ Ripeness model not found: {ripeness_model_path}")
        return None, None
    
    ripeness_model = tf.keras.models.load_model(ripeness_model_path)
    
    return yolo_model, ripeness_model


yolo_model, ripeness_model = load_models()

if yolo_model is None or ripeness_model is None:
    st.stop()

RIPENESS_CLASSES = ['Unripe', 'Ripe', 'Overripe']
FRUIT_TYPES = ['Apple', 'Orange']

# YOLO classes for fruits (from COCO dataset)
FRUIT_CLASS_IDS = {
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    # Add more as needed
}

# Add this in your Camera tab before the analyze button
conf_threshold = st.slider("Detection Confidence Threshold", 0.05, 0.50, 0.15, 0.05)

def detect_fruit_with_yolo(image_pil):
    """
    Stage 1: Use YOLO to detect if there's a fruit in the image
    Returns: (has_fruit, cropped_fruit_image, fruit_type_detected, confidence)
    """
    try:
        # Convert PIL to numpy array
        img_array = np.array(image_pil)
        
        # Run YOLO detection - LOWERED confidence threshold for better detection
        results = yolo_model.predict(
        source=img_array,
        classes=[47, 49],
        conf=conf_threshold,  # Use slider value
        verbose=False
        )
        
        # Check if any fruits were detected
        if len(results[0].boxes) == 0:
            return False, None, None, 0.0
        
        # Get the first detected fruit (highest confidence)
        boxes = results[0].boxes
        best_box = boxes[0]
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        
        # Add padding to crop (10% on each side)
        height, width = img_array.shape[:2]
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Crop the fruit region with padding
        cropped_fruit = img_array[y1:y2, x1:x2]
        
        # Check if crop is valid
        if cropped_fruit.size == 0:
            return False, None, None, 0.0
            
        cropped_fruit_pil = Image.fromarray(cropped_fruit)
        
        # Get detected class
        class_id = int(best_box.cls[0])
        detected_fruit = FRUIT_CLASS_IDS.get(class_id, "unknown")
        confidence = float(best_box.conf[0])
        
        return True, cropped_fruit_pil, detected_fruit, confidence
        
    except Exception as e:
        st.error(f"YOLO detection error: {e}")
        return False, None, None, 0.0



def preprocess_image(image_pil):
    """Convert PIL image to preprocessed tensor for ripeness model"""
    try:
        img = np.array(image_pil)
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        img = cv2.resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype('float32'))
        img = np.expand_dims(img, axis=0)
        
        return img
    
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None


def get_ripeness_prediction(image_pil):
    """Stage 2: Get ripeness prediction for detected fruit"""
    try:
        processed = preprocess_image(image_pil)
        if processed is None:
            return None
        
        predictions = ripeness_model.predict(processed, verbose=0)
        ripeness_probs = predictions[0][0]
        fruit_probs = predictions[1][0]
        
        ripeness_idx = int(np.argmax(ripeness_probs))
        fruit_idx = int(np.argmax(fruit_probs))
        
        return {
            'ripeness': RIPENESS_CLASSES[ripeness_idx],
            'ripeness_conf': float(ripeness_probs[ripeness_idx]),
            'fruit': FRUIT_TYPES[fruit_idx],
            'fruit_conf': float(fruit_probs[fruit_idx]),
            'ripeness_probs': ripeness_probs,
            'fruit_probs': fruit_probs
        }
    
    except Exception as e:
        st.error(f"❌ Ripeness prediction error: {e}")
        return None

def analyze_image(image_pil):
    """
    TEMPORARY: Skip YOLO and directly analyze with ripeness model
    """
    # Directly get ripeness prediction without YOLO filtering
    ripeness_result = get_ripeness_prediction(image_pil)
    
    if ripeness_result is None:
        return False, None, "❌ Error analyzing fruit."
    
    # Add dummy YOLO data
    ripeness_result['yolo_detected'] = 'Assumed Fruit'
    ripeness_result['yolo_confidence'] = 1.0
    ripeness_result['cropped_image'] = image_pil  # Use full image
    
    return True, ripeness_result, "✅ Fruit analyzed (YOLO temporarily bypassed)"


def show_yolo_detection(image_pil):
    """Visualize YOLO detections with bounding boxes"""
    img_array = np.array(image_pil)
    
    # Run YOLO
    results = yolo_model.predict(
        source=img_array,
        classes=[47, 49],
        conf=0.25,
        verbose=False
    )
    
    # Draw boxes on image
    annotated_image = results[0].plot()  # This draws boxes automatically!
    
    return Image.fromarray(annotated_image)


def save_result_to_csv(result, source):
    """Save prediction result to CSV and session state"""
    st.session_state.result_counter += 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    result_entry = {
        'ID': st.session_state.result_counter,
        'Timestamp': timestamp,
        'Date': datetime.now().strftime("%Y-%m-%d"),
        'Time': datetime.now().strftime("%H:%M:%S"),
        'Source': source,
        'YOLO_Detected': result.get('yolo_detected', 'N/A'),
        'YOLO_Confidence': round(result.get('yolo_confidence', 0) * 100, 2),
        'Fruit_Type': result['fruit'],
        'Fruit_Confidence': round(result['fruit_conf'] * 100, 2),
        'Ripeness': result['ripeness'],
        'Ripeness_Confidence': round(result['ripeness_conf'] * 100, 2)
    }
    
    st.session_state.results_history.append(result_entry)
    
    df = pd.DataFrame(st.session_state.results_history)
    df.to_csv(CSV_FILE, index=False)


# Main tabs
tab1, tab2, tab3 = st.tabs(["📷 Camera", "📁 Upload", "📋 Recent Results"])

# Tab 1: Camera
with tab1:
    st.header("📷 Take a Photo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        picture = st.camera_input("Point camera at fruit")
        if picture:
            st.info("✓ Photo captured! Click 'Analyze Photo' to get results.")
    
    with col2:
        if picture:
            image = Image.open(picture)
            
            if st.button("🔍 Analyze Photo", key="analyze_camera", type="primary"):
                with st.spinner("🔍 Step 1: Detecting fruit..."):
                    success, result, message = analyze_image(image)
                
                st.info(message)
                
                if success:
                    # Show YOLO detection with bounding box
                    st.subheader("🎯 Detection Result")
                    annotated_img = show_yolo_detection(image)
                    st.image(annotated_img, caption="YOLO Detection with Bounding Box", use_column_width=True)
                    
                    # Show cropped fruit
                    st.subheader("🔍 Cropped Fruit Region")
                    st.image(result['cropped_image'], caption="Region sent to Ripeness Model", width=250)
                    
                    with st.spinner("🔍 Step 2: Analyzing ripeness..."):
                        save_result_to_csv(result, "Camera")
                    
                    st.success("✓ Analysis Complete & Saved!")
                    
                    # Display results
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🍎 Ripeness", result['ripeness'], f"{result['ripeness_conf']:.1%}")
                    with col_b:
                        st.metric("🍊 Fruit Type", result['fruit'], f"{result['fruit_conf']:.1%}")
                    
                    with st.expander("📊 Detailed Scores"):
                        st.write(f"**YOLO Detection:** {result['yolo_detected']} ({result['yolo_confidence']:.1%})")
                        st.write("\n**Ripeness Breakdown:**")
                        for i, cls in enumerate(RIPENESS_CLASSES):
                            st.write(f"  {cls}: {result['ripeness_probs'][i]:.2%}")
                        st.write("\n**Fruit Type Breakdown:**")
                        for i, cls in enumerate(FRUIT_TYPES):
                            st.write(f"  {cls}: {result['fruit_probs'][i]:.2%}")
                    
                    st.info(f"💾 Result #{st.session_state.result_counter} saved!")


# Tab 2: Upload
with tab2:
    st.header("📁 Upload Image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            st.info("✓ Image uploaded! Click 'Analyze Image' to get results.")
    
    with col2:
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            if st.button("🔍 Analyze Image", key="analyze_upload", type="primary"):
                with st.spinner("🔍 Step 1: Detecting fruit..."):
                    success, result, message = analyze_image(image)
                
                st.info(message)
                
                if success:
                    # Show YOLO detection with bounding box
                    st.subheader("🎯 Detection Result")
                    annotated_img = show_yolo_detection(image)
                    st.image(annotated_img, caption="YOLO Detection with Bounding Box", use_column_width=True)
                    
                    # Show cropped fruit
                    st.subheader("🔍 Cropped Fruit Region")
                    st.image(result['cropped_image'], caption="Region sent to Ripeness Model", width=250)
                    
                    with st.spinner("🔍 Step 2: Analyzing ripeness..."):
                        save_result_to_csv(result, "Upload")
                    
                    st.success("✓ Analysis Complete & Saved!")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🍎 Ripeness", result['ripeness'], f"{result['ripeness_conf']:.1%}")
                    with col_b:
                        st.metric("🍊 Fruit Type", result['fruit'], f"{result['fruit_conf']:.1%}")
                    
                    st.info(f"💾 Result #{st.session_state.result_counter} saved!")


# Tab 3: Recent Results
with tab3:
    st.header("📋 Recent Results")
    
    if len(st.session_state.results_history) > 0:
        df = pd.DataFrame(st.session_state.results_history)
        st.dataframe(df.tail(10), use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🗑️ Clear All Data", type="secondary"):
                if os.path.exists(CSV_FILE):
                    os.remove(CSV_FILE)
                st.session_state.results_history = []
                st.session_state.result_counter = 0
                st.rerun()
    else:
        st.info("📭 No results yet. Take a photo or upload an image!")


st.markdown("---")
st.markdown("Made with ❤️ using YOLOv8, TensorFlow & Streamlit")
