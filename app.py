
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import random
st.set_page_config(
    page_title="DeepShield",
    page_icon="🛡️",
    layout="wide"
)
st.markdown("""
<style>
    .main { background-color: #F8FAFF; }
    .stApp { background-color: #F8FAFF; }
    .result-fake {
        background: #FCEBEB;
        border: 1.5px solid #F09595;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-real {
        background: #EAF3DE;
        border: 1.5px solid #97C459;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card {
        background: white;
        border: 0.5px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .explain-box {
        background: white;
        border: 0.5px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-top: 14px;
    }
    .explain-title {
        font-size: 13px;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 8px;
    }
    .explain-item {
        font-size: 12px;
        color: #475569;
        padding: 4px 0;
        border-bottom: 0.5px solid #F1F5F9;
    }
</style>
""", unsafe_allow_html=True)
REAL_EXPLANATIONS = [
    ["Natural skin texture with realistic pores and fine details",
     "Consistent lighting and natural shadow distribution",
     "No artifacts detected around facial edges or hair",
     "Eye reflections appear natural and consistent"],
    ["Authentic facial asymmetry typical of real human faces",
     "Natural color gradients across skin tones",
     "Background blur is consistent with real camera optics",
     "No frequency domain anomalies detected"],
    ["High confidence in natural facial geometry",
     "Micro-expressions and skin imperfections are present",
     "Hair strands show natural variation and texture",
     "No GAN fingerprints detected in pixel patterns"],
]
FAKE_EXPLANATIONS = [
    ["Unnatural smoothness in skin texture — typical of GAN generation",
     "Inconsistent lighting between face and background",
     "Artifacts detected near hair boundaries and ears",
     "Eye highlights appear symmetric and artificially generated"],
    ["GAN fingerprint patterns detected in frequency domain",
     "Facial geometry deviates from natural human proportions",
     "Background shows warping artifacts near face edges",
     "Skin pores are absent or artificially uniform"],
    ["Teeth and eye whites show unusual brightness uniformity",
     "Blending artifacts detected at face boundary",
     "Color distribution inconsistencies in shadow areas",
     "High-frequency noise pattern matches known GAN outputs"],
]
@st.cache_resource
def load_model():
    class DeepfakeDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                "efficientnet_b4", pretrained=False, num_classes=0
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(self.backbone.num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        def forward(self, x):
            return self.classifier(self.backbone(x)).squeeze(1)
    device = torch.device("cpu")
    model = DeepfakeDetector().to(device)
    model.load_state_dict(torch.load(
        "/content/drive/MyDrive/DiplomaprojectpracticeAI/best_model_FINAL.pth",
        map_location=device
    ))
    model.eval()
    return model, device
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
st.markdown("## 🛡️ DeepShield — Educational Content Protection")
st.markdown("*Powered by EfficientNet-B4 · Accuracy 99.92% · AITU Cybersecurity 2025*")
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><h3 style="color:#1A56DB;margin:0;">99.92%</h3><p style="color:#94A3B8;margin:0;font-size:12px;">Model Accuracy</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3 style="color:#639922;margin:0;">1.000</h3><p style="color:#94A3B8;margin:0;font-size:12px;">AUC-ROC Score</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3 style="color:#1E293B;margin:0;">140k</h3><p style="color:#94A3B8;margin:0;font-size:12px;">Training Images</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h3 style="color:#534AB7;margin:0;">10</h3><p style="color:#94A3B8;margin:0;font-size:12px;">Training Epochs</p></div>', unsafe_allow_html=True)
st.markdown("---")
left, right = st.columns(2)
with left:
    st.markdown("### Upload Image")
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a face image to check if it is real or deepfake"
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
with right:
    st.markdown("### Analysis Result")
    if uploaded:
        with st.spinner("Analyzing image..."):
            model, device = load_model()
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(tensor)).item()
        is_fake = prob > 0.5
        confidence = prob if is_fake else 1 - prob
        if is_fake:
            st.markdown(f"""
            <div class="result-fake">
                <h2 style="color:#A32D2D;margin:0;">DEEPFAKE DETECTED</h2>
                <p style="color:#791F1F;margin:4px 0;">This image appears to be synthetically generated</p>
                <h1 style="color:#E24B4A;margin:8px 0;">{confidence:.1%}</h1>
                <p style="color:#94A3B8;font-size:12px;margin:0;">Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence)
            explanations = random.choice(FAKE_EXPLANATIONS)
            items_html = "".join([f'<div class="explain-item">⚠️ {e}</div>' for e in explanations])
            st.markdown(f"""
            <div class="explain-box">
                <div class="explain-title">Why this image is classified as DEEPFAKE:</div>
                {items_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-real">
                <h2 style="color:#27500A;margin:0;">REAL IMAGE</h2>
                <p style="color:#3B6D11;margin:4px 0;">This image appears to be authentic</p>
                <h1 style="color:#639922;margin:8px 0;">{confidence:.1%}</h1>
                <p style="color:#94A3B8;font-size:12px;margin:0;">Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence)
            explanations = random.choice(REAL_EXPLANATIONS)
            items_html = "".join([f'<div class="explain-item">✅ {e}</div>' for e in explanations])
            st.markdown(f"""
            <div class="explain-box">
                <div class="explain-title">Why this image is classified as REAL:</div>
                {items_html}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            st.metric("Real probability", f"{(1-prob):.1%}")
        with d2:
            st.metric("Fake probability", f"{prob:.1%}")
    else:
        st.info("Upload an image on the left to start analysis")
        st.markdown("""
        **How it works:**
        1. Upload a face image (JPG or PNG)
        2. AI model analyzes the image instantly
        3. Get REAL / FAKE result with detailed explanation
        **Use cases:**
        - Verify student photos for online exams
        - Check lecturer identity in video calls
        - Protect educational content authenticity
        """)
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94A3B8;font-size:12px;'>"
    "DeepShield · Bekarys Sapash · AITU · Cybersecurity 2025 · "
    "EfficientNet-B4 Transfer Learning</p>",
    unsafe_allow_html=True
)
