import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np
import os

st.set_page_config(
    page_title="Satellite Water Mapper",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: #00BFFF;'>🌊 Satellite Water Mapper</h1>
    <p style='text-align: center; color: gray;'>
        AI-based Water Detection from Satellite Images
    </p>
    <hr>
""", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    model_path = r"C:\Users\user\Videos\DATA ANALYSIS & SCIENCE PORTFOLIO\Water-Specific Segmentation\unet_model.pth"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
st.sidebar.header("⚙️ Controls")
threshold = st.sidebar.slider("Water Sensitivity", 0.0, 1.0, 0.5)


uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    input_image = transform(image).unsqueeze(0).to(device)

    with st.spinner("Detecting water regions..."):
        with torch.no_grad():
            output = model(input_image)
            pred_mask = torch.sigmoid(output)
            pred_mask = (pred_mask > threshold).float()

    mask = pred_mask.squeeze().cpu().numpy()

    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    colored_mask[:, :, 2] = mask  # blue channel in [0,1]

    with col2:
        st.subheader("Detected Water")
        st.image(colored_mask, use_container_width=True)

    st.markdown("### 🌍 Overlay Result")

    overlay = image.resize((256, 256))
    overlay = np.array(overlay).astype(np.float32) / 255.0  

    overlay[:, :, 2] = np.maximum(overlay[:, :, 2], mask)

    st.image(overlay, use_container_width=True)

    st.success("Water Detection Completed ✅")

st.markdown("""
    <hr>
    <p style='text-align: center; color: gray;'>
        🌍 Built by Khubaib | AI + Satellite Vision Project
    </p>
""", unsafe_allow_html=True)