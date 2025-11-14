# app.py - Updated for 4-class model with "Not Fruit" detection
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
    if os.path.exists(CSV_FILE):
        st.session_state.results_history = pd.read_csv(CSV_FILE).to_dict('records')
    else:
        st.session_state.results_history = []

if 'result_counter' not in st.session_state:
    st.session_state.result_counter = len(st.session_state.results_history)

# Load model
@st.cache_resource
def load_model():
    """Load the trained ripeness classification model"""
    model_path = 'fruit_ripeness_with_person_rejection.keras'
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        return None
    
    model = tf.keras.models.load_model(model_path)
    return model

ripeness_model = load_model()

if ripeness_model is None:
    st.stop()

# Updated classes with "Not Fruit"
RIPENESS_CLASSES = ['Unripe', 'Ripe', 'Overripe', 'Not Fruit']
FRUIT_TYPES = ['Apple', 'Orange']

def preprocess_image(image_pil):
    """Convert PIL image to preprocessed tensor for model"""
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

def analyze_image(image_pil):
    """
    Analyze image with the 4-class model (includes "Not Fruit" detection)
    Returns: (success, result_dict, message)
    """
    try:
        processed = preprocess_image(image_pil)
        if processed is None:
            return False, None, "❌ Error preprocessing image."
        
        # Get predictions
        predictions = ripeness_model.predict(processed, verbose=0)
        ripeness_probs = predictions[0][0]  # 4 classes
        fruit_probs = predictions[1][0]     # 2 classes
        
        # Get top prediction
        ripeness_idx = int(np.argmax(ripeness_probs))
        fruit_idx = int(np.argmax(fruit_probs))
        
        ripeness_class = RIPENESS_CLASSES[ripeness_idx]
        ripeness_conf = float(ripeness_probs[ripeness_idx])
        
        # Check if it's "Not Fruit"
        if ripeness_idx == 3:  # "Not Fruit" class
            return True, {
                'ripeness': ripeness_class,
                'ripeness_conf': ripeness_conf,
                'fruit': 'N/A',
                'fruit_conf': 0.0,
                'ripeness_probs': ripeness_probs,
                'fruit_probs': fruit_probs,
                'is_fruit': False
            }, "⚠️ Not a fruit detected (person, object, or background)"
        
        # It's a fruit - return fruit type
        return True, {
            'ripeness': ripeness_class,
            'ripeness_conf': ripeness_conf,
            'fruit': FRUIT_TYPES[fruit_idx],
            'fruit_conf': float(fruit_probs[fruit_idx]),
            'ripeness_probs': ripeness_probs,
            'fruit_probs': fruit_probs,
            'is_fruit': True
        }, "✅ Fruit analyzed successfully"
    
    except Exception as e:
        st.error(f"❌ Analysis error: {e}")
        return False, None, f"❌ Error: {e}"

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
        'Is_Fruit': result.get('is_fruit', True),
        'Fruit_Type': result.get('fruit', 'N/A'),
        'Fruit_Confidence': round(result.get('fruit_conf', 0) * 100, 2),
        'Ripeness': result['ripeness'],
        'Ripeness_Confidence': round(result['ripeness_conf'] * 100, 2)
    }
    
    st.session_state.results_history.append(result_entry)
    
    df = pd.DataFrame(st.session_state.results_history)
    df.to_csv(CSV_FILE, index=False)

# Main tabs
tab1, tab2, tab3 = st.tabs(["📷 Camera", "📁 Upload", "📊 Dashboard"])

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
                with st.spinner("🔍 Analyzing image..."):
                    success, result, message = analyze_image(image)
                
                if success:
                    # Check if it's "Not Fruit"
                    if not result['is_fruit']:
                        st.warning("⚠️ **Not a Fruit Detected!**")
                        st.error(f"This appears to be: **{result['ripeness']}**")
                        st.metric("Detection Confidence", f"{result['ripeness_conf']:.1%}")
                        
                        st.info("💡 Please capture an image with an apple or orange.")
                    else:
                        # It's a fruit - show full results
                        save_result_to_csv(result, "Camera")
                        
                        st.success("✅ Analysis Complete & Saved!")
                        
                        # Display main results
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("🍎 Ripeness", result['ripeness'], 
                                     f"↑ {result['ripeness_conf']:.1%}")
                        with col_b:
                            st.metric("🍊 Fruit Type", result['fruit'], 
                                     f"↑ {result['fruit_conf']:.1%}")
                        
                        # Detailed scores
                        with st.expander("📊 Detailed Scores"):
                            st.write("**Ripeness Breakdown:**")
                            for i, cls in enumerate(RIPENESS_CLASSES):
                                st.progress(float(result['ripeness_probs'][i]), 
                                          text=f"{cls}: {result['ripeness_probs'][i]:.2%}")
                            
                            st.write("\n**Fruit Type Breakdown:**")
                            for i, cls in enumerate(FRUIT_TYPES):
                                st.progress(float(result['fruit_probs'][i]), 
                                          text=f"{cls}: {result['fruit_probs'][i]:.2%}")
                        
                        st.info(f"💾 Result #{st.session_state.result_counter} saved!")
                else:
                    st.error(message)

# Tab 2: Upload
with tab2:
    st.header("📁 Upload Image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            st.info("✓ Image uploaded! Click 'Analyze Image' to get results.")
    
    with col2:
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            if st.button("🔍 Analyze Image", key="analyze_upload", type="primary"):
                with st.spinner("🔍 Analyzing image..."):
                    success, result, message = analyze_image(image)
                
                if success:
                    if not result['is_fruit']:
                        st.warning("⚠️ **Not a Fruit Detected!**")
                        st.error(f"This appears to be: **{result['ripeness']}**")
                        st.metric("Detection Confidence", f"{result['ripeness_conf']:.1%}")
                        st.info("💡 Please upload an image with an apple or orange.")
                    else:
                        save_result_to_csv(result, "Upload")
                        st.success("✅ Analysis Complete & Saved!")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("🍎 Ripeness", result['ripeness'], 
                                     f"↑ {result['ripeness_conf']:.1%}")
                        with col_b:
                            st.metric("🍊 Fruit Type", result['fruit'], 
                                     f"↑ {result['fruit_conf']:.1%}")
                        
                        with st.expander("📊 Detailed Scores"):
                            st.write("**Ripeness Breakdown:**")
                            for i, cls in enumerate(RIPENESS_CLASSES):
                                st.progress(float(result['ripeness_probs'][i]), 
                                          text=f"{cls}: {result['ripeness_probs'][i]:.2%}")
                            
                            st.write("\n**Fruit Type Breakdown:**")
                            for i, cls in enumerate(FRUIT_TYPES):
                                st.progress(float(result['fruit_probs'][i]), 
                                          text=f"{cls}: {result['fruit_probs'][i]:.2%}")
                        
                        st.info(f"💾 Result #{st.session_state.result_counter} saved!")
                else:
                    st.error(message)

# Tab 3: Dashboard
with tab3:
    st.header("📊 Analysis Dashboard")
    
    if len(st.session_state.results_history) > 0:
        df = pd.DataFrame(st.session_state.results_history)
        
        # Filter to only fruits (exclude "Not Fruit" detections)
        df_fruits = df[df['Is_Fruit'] == True]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Scans", len(df))
        with col2:
            st.metric("Fruits Detected", len(df_fruits))
        with col3:
            st.metric("Non-Fruits Rejected", len(df) - len(df_fruits))
        with col4:
            if len(df_fruits) > 0:
                most_common = df_fruits['Fruit_Type'].mode()[0]
                st.metric("Most Common", most_common)
        
        # Charts
        if len(df_fruits) > 0:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Ripeness Distribution")
                ripeness_counts = df_fruits['Ripeness'].value_counts()
                st.bar_chart(ripeness_counts)
            
            with col_b:
                st.subheader("Fruit Type Distribution")
                fruit_counts = df_fruits['Fruit_Type'].value_counts()
                st.bar_chart(fruit_counts)
        
        # Recent results table
        st.subheader("Recent Results")
        st.dataframe(df.tail(10), use_container_width=True, hide_index=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Data (CSV)",
            data=csv,
            file_name=f"fruit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Clear data button
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
st.markdown("Made with ❤️ using MobileNetV2, TensorFlow & Streamlit")