import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

from model_loader import ResNet50CIFAR100Predictor
from cifar100_labels import CIFAR100_CLASSES, get_display_name, get_superclass_name

st.set_page_config(
    page_title="CIFAR-100 ResNet-50 Vision Studio",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ CIFAR-100 ResNet-50 Computer Vision Studio")
st.markdown("""
Welcome to the live inference portal for the fine-tuned **ResNet-50** PyTorch model trained on the **CIFAR-100** dataset (100 categories).
""")

@st.cache_resource
def load_predictor():
    return ResNet50CIFAR100Predictor(model_path='resnet50_cifar100_finetuned.pth')

with st.spinner("Loading ResNet-50 Model..."):
    predictor = load_predictor()

st.sidebar.header("⚙️ Model Configuration")
st.sidebar.write(f"**Execution Device:** `{predictor.device}`")
st.sidebar.write(f"**Categories:** {len(CIFAR100_CLASSES)}")

top_k = st.sidebar.slider("Top Predictions", min_value=1, max_value=10, value=5)

uploaded_file = st.file_uploader("Upload an Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("🖼️ Uploaded Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("📊 Top Predictions")
        rgb_np = np.array(image)
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Classifying image..."):
            res = predictor.predict(bgr_np, top_k=top_k)
        
        st.success(f"Inference Latency: **{res.get('latency_ms', 0):.2f} ms**")
        st.write(f"**Top Class:** `{res['top_class']}` ({res['top_confidence']*100:.2f}%)")
        st.write(f"**Superclass:** `{res['superclass']}`")
        
        st.markdown("---")
        for pred in res['predictions']:
            label = pred['display_name']
            conf = pred['confidence']
            st.write(f"**{label}** (`{pred['superclass']}`): {conf*100:.2f}%")
            st.progress(min(float(conf), 1.0))
else:
    st.info("👆 Please upload an image using the file uploader above to view real-time model predictions.")
