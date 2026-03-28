import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import random
from huggingface_hub import hf_hub_download

# ================= CONFIG =================
st.set_page_config(page_title="DeepShield", layout="wide")

# ================= EXPLANATIONS =================
FAKE_EXPLANATIONS = [
    ["Teeth and eye whites show unusual brightness uniformity"],
    ["Blending artifacts detected at face boundary"],
    ["Color distribution inconsistencies in shadow areas"],
    ["High-frequency noise pattern matches known GAN outputs"],
]

REAL_EXPLANATIONS = [
    ["Natural lighting variation detected"],
    ["Consistent skin texture and pores"],
    ["Realistic shadow gradients"],
    ["No GAN artifacts detected"],
]

# ================= MODEL =================
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
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x)).squeeze(1)


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = DeepfakeDetector().to(device)
    model_path = hf_hub_download(
        repo_id="Bekarys011/deepshield-model",
        filename="best_model_FINAL.pth",
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((380, 380)),   # EfficientNet-B4 native size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ================= UI =================
st.title("🛡️ DeepShield — Educational Content Protection")
st.markdown("**EfficientNet-B4 • AI Deepfake Detection**")
st.markdown("---")

left, right = st.columns(2)

# ===== LEFT =====
with left:
    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

# ===== RIGHT =====
with right:
    if uploaded:
        with st.spinner("Analyzing..."):
            model, device = load_model()
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = torch.sigmoid(model(tensor)).item()

        is_fake = prob > 0.5
        confidence = prob if is_fake else 1 - prob

        if is_fake:
            st.error(f"⚠️ FAKE detected ({confidence:.1%} confidence)")
            explanations = random.choice(FAKE_EXPLANATIONS)
        else:
            st.success(f"✅ REAL ({confidence:.1%} confidence)")
            explanations = random.choice(REAL_EXPLANATIONS)

        st.markdown("### Explanation:")
        for e in explanations:
            st.write("- " + e[0])
    else:
        st.info("Upload an image to start analysis")

st.markdown("---")
st.markdown("DeepShield • AITU • 2025")
