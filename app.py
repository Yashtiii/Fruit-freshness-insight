# app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from datetime import datetime
import os


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
    # Load existing results from CSV if it exists
    if os.path.exists(CSV_FILE):
        st.session_state.results_history = pd.read_csv(CSV_FILE).to_dict('records')
    else:
        st.session_state.results_history = []

if 'result_counter' not in st.session_state:
    st.session_state.result_counter = len(st.session_state.results_history)


# Load model
@st.cache_resource
def load_model():
    model_path = 'fruit_ripeness_multitask_finetuned.keras'
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    return tf.keras.models.load_model(model_path)


model = load_model()
if model is None:
    st.stop()

RIPENESS_CLASSES = ['Unripe', 'Ripe', 'Overripe']
FRUIT_TYPES = ['Apple', 'Orange']


def preprocess_image(image_pil):
    """Convert PIL image to preprocessed tensor"""
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


def get_prediction(image_pil):
    """Get model prediction"""
    try:
        processed = preprocess_image(image_pil)
        if processed is None:
            return None
        
        predictions = model.predict(processed, verbose=0)
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
        st.error(f"❌ Prediction error: {e}")
        return None


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
        'Fruit_Type': result['fruit'],
        'Fruit_Confidence': round(result['fruit_conf'] * 100, 2),
        'Ripeness': result['ripeness'],
        'Ripeness_Confidence': round(result['ripeness_conf'] * 100, 2)
    }
    
    # Add to session state
    st.session_state.results_history.append(result_entry)
    
    # Save to CSV
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
                with st.spinner("🔍 Analyzing..."):
                    result = get_prediction(image)
                
                if result:
                    save_result_to_csv(result, "Camera")
                    st.success("✓ Analysis Complete & Saved!")
                    
                    st.metric("🍎 Ripeness", result['ripeness'], f"{result['ripeness_conf']:.1%}")
                    st.metric("🍊 Fruit Type", result['fruit'], f"{result['fruit_conf']:.1%}")
                    
                    with st.expander("📊 Detailed Scores"):
                        st.write("**Ripeness Breakdown:**")
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
                with st.spinner("🔍 Analyzing..."):
                    result = get_prediction(image)
                
                if result:
                    save_result_to_csv(result, "Upload")
                    st.success("✓ Analysis Complete & Saved!")
                    
                    st.metric("🍎 Ripeness", result['ripeness'], f"{result['ripeness_conf']:.1%}")
                    st.metric("🍊 Fruit Type", result['fruit'], f"{result['fruit_conf']:.1%}")
                    
                    st.info(f"💾 Result #{st.session_state.result_counter} saved!")


# Tab 3: Recent Results
with tab3:
    st.header("📋 Recent Results")
    
    if len(st.session_state.results_history) > 0:
        df = pd.DataFrame(st.session_state.results_history)
        
        # Show last 10 results
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
st.markdown("Made with ❤️ using TensorFlow & Streamlit")
