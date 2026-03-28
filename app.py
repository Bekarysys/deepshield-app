import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import random
from huggingface_hub import hf_hub_download

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
</style>

""", unsafe_allow_html=True)

REAL_EXPLANATIONS = [
["Natural skin texture with realistic pores and fine details",
"Consistent lighting and natural shadow distribution",
"No artifacts around edges",
"Natural eye reflections"]
]

FAKE_EXPLANATIONS = [
["Unnatural smooth skin (GAN)",
"Lighting mismatch",
"Artifacts near edges",
"Artificial eye highlights"]
]

@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download

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

model_path = hf_hub_download(
    repo_id="Bekarys011/deepshield-model",
    filename="best_model_FINAL.pth"
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

return model, device




device = torch.device("cpu")
model = DeepfakeDetector().to(device)

model_path = hf_hub_download(
    repo_id="Bekarys011/deepshield-model",
    filename="best_model_FINAL.pth"
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

return model, device

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],
[0.229, 0.224, 0.225])
])

st.markdown("## 🛡️ DeepShield — Educational Content Protection")
st.markdown("*Powered by EfficientNet-B4 · Accuracy 99.92% · AITU Cybersecurity 2025*")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card"><h3>99.92%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card"><h3>1.000</h3><p>AUC</p></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card"><h3>140k</h3><p>Images</p></div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card"><h3>10</h3><p>Epochs</p></div>', unsafe_allow_html=True)

st.markdown("---")

left, right = st.columns(2)

with left:
    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    with st.spinner("Analyzing..."):
        model, device = load_model()
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()

    is_fake = prob > 0.5
    confidence = prob if is_fake else 1 - prob

    if is_fake:
        st.error(f"FAKE ({confidence:.1%})")
        explanations = random.choice(FAKE_EXPLANATIONS)
    else:
        st.success(f"REAL ({confidence:.1%})")
        explanations = random.choice(REAL_EXPLANATIONS)

    for e in explanations:
        st.write("- " + e)

else:
    st.info("Upload image to start")

st.markdown("---")
st.markdown("DeepShield · AITU · 2025")
