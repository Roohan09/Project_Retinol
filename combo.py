# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import cv2

# # =====================================================
# # 1️⃣ DEVICE SETUP
# # =====================================================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# st.set_page_config(
#     page_title="OCT Analyzer",
#     page_icon="👁️",
#     layout="wide",
# )
# st.markdown(
#     """
#     <h1 style='text-align: center; color: #1E90FF;'>👁️ Retinal OCT Analyzer</h1>
#     <h4 style='text-align: center; color: #808080;'>AI-powered tool for disease classification and lesion segmentation</h4>
#     <hr style='border:1px solid #ccc'>
#     """,
#     unsafe_allow_html=True
# )


# # =====================================================
# # 2️⃣ CLASSIFICATION MODEL (ResNet18)
# # =====================================================
# def build_classification_model():
#     model = models.resnet18(weights=None)
#     model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     model.fc = nn.Linear(model.fc.in_features, 4)
#     return model


# @st.cache_resource
# def load_classification_model():
#     model = build_classification_model()
#     try:
#         model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
#         model.to(DEVICE).eval()
#         return model
#     except Exception as e:
#         st.error(f"❌ Error loading classification model: {e}")
#         return None


# # =====================================================
# # 3️⃣ SEGMENTATION MODEL (Matches trained UNet)
# # =====================================================
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x): return self.double_conv(x)


# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )
#     def forward(self, x): return self.maxpool_conv(x)


# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         return self.conv(torch.cat([x2, x1], dim=1))


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#     def forward(self, x): return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self, n_channels=1, n_classes=1, bilinear=True):
#         super().__init__()
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         return self.outc(x)


# @st.cache_resource
# def load_segmentation_model():
#     model = UNet(n_channels=1, n_classes=1, bilinear=True)
#     try:
#         state_dict = torch.load("segmentation_best_unet_model.pth", map_location=DEVICE)
#         model.load_state_dict(state_dict, strict=True)
#         model.to(DEVICE).eval()
#         return model
#     except Exception as e:
#         st.error(f"❌ Error loading segmentation model: {e}")
#         return None


# # =====================================================
# # 4️⃣ PREPROCESSING & LABELS
# # =====================================================
# transform_clf = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
# DESCRIPTIONS = {
#     "CNV": "🩸 **Choroidal Neovascularization (CNV)** — abnormal blood vessels grow under the retina, leaking fluid or blood.",
#     "DME": "💧 **Diabetic Macular Edema (DME)** — fluid accumulates in the macula due to damaged blood vessels.",
#     "DRUSEN": "🟡 **Drusen** — yellow lipid deposits under the retina, early sign of age-related macular degeneration (AMD).",
#     "NORMAL": "✅ **Normal Retina** — healthy retinal structure with no visible fluid or vessel abnormalities."
# }


# # =====================================================
# # 5️⃣ PREDICTION FUNCTIONS
# # =====================================================
# def predict_class(model, image):
#     img = transform_clf(image.convert("L")).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         out = model(img)
#         probs = torch.softmax(out, dim=1)
#         conf, idx = torch.max(probs, dim=1)
#     return CLASSES[idx.item()], conf.item() * 100


# def preprocess_seg(image):
#     img_gray = image.convert("L").resize((512, 512))
#     img = np.array(img_gray, dtype=np.float32)
#     img = (img - img.mean()) / (img.std() + 1e-8)
#     tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
#     return tensor


# def predict_mask(model, image):
#     tensor = preprocess_seg(image)
#     with torch.no_grad():
#         output = torch.sigmoid(model(tensor))
#     return (output.squeeze().cpu().numpy() > 0.4).astype(np.uint8)


# # =====================================================
# # 6️⃣ APP LOGIC
# # =====================================================
# clf_model = load_classification_model()
# seg_model = load_segmentation_model()

# uploaded_file = st.file_uploader("📤 Upload an OCT image", type=["jpg", "jpeg", "png"])

# if uploaded_file and clf_model and seg_model:
#     image = Image.open(uploaded_file)

#     col1, col2 = st.columns(2)

#     # -----------------------------
#     # LEFT: CLASSIFICATION SECTION
#     # -----------------------------
#     with col1:
#         st.image(image, caption="Uploaded OCT Image", width=300)
#         with st.spinner("🔍 Classifying..."):
#             label, conf = predict_class(clf_model, image)
#         st.success(f"**Prediction:** {label}")
#         st.info(f"**Confidence:** {conf:.2f}%")
#         st.markdown(DESCRIPTIONS[label])

#     # -----------------------------
#     # RIGHT: SEGMENTATION (DME only)
#     # -----------------------------
#     with col2:
#         st.subheader("🩺 Disease Segmentation")

#         if label == "DME":
#             with st.spinner("🧩 Detecting diseased region..."):
#                 mask = predict_mask(seg_model, image)

#             # Convert and resize everything for perfect alignment
#             mask_uint8 = (mask * 255).astype(np.uint8)
#             image_resized = image.resize((mask.shape[1], mask.shape[0]))
#             base_rgb = np.array(image_resized.convert("RGB"))

#             # ✅ Morphological cleanup (fills gaps, smooth edges)
#             kernel = np.ones((3, 3), np.uint8)
#             mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
#             mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
#             mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)  # slight dilation for smoother coverage

#             # Calculate diseased area
#             disease_area = (mask_clean.sum() / mask_clean.size / 255) * 100

#             # ✅ Red overlay for diseased region
#             red_overlay = np.zeros_like(base_rgb)
#             red_overlay[mask_clean == 255] = [255, 0, 0]
#             blended = cv2.addWeighted(base_rgb, 0.7, red_overlay, 0.6, 0)

#             # ✅ Green filled contour for perfect lesion outline
#             contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             filtered = [c for c in contours if cv2.contourArea(c) > 80]
#             outlined = blended.copy()

#             for c in filtered:
#                 cv2.drawContours(outlined, [c], -1, (0, 255, 0), thickness=-1)

#             # ✅ Slight second blending for smooth natural color
#             final_overlay = cv2.addWeighted(base_rgb, 0.5, outlined, 0.5, 0)

#             # --- DISPLAY (Bigger, Cleaner) ---
#             st.markdown(f"### 🩸 **Detected Diseased Area:** {disease_area:.2f}% of image**")

#             # Two wide columns for better visualization
#             col3, col4 = st.columns([1, 1])
#             with col3:
#                 st.image(mask_clean, caption="🧠 Binary Mask", use_container_width=True)
#             with col4:
#                 st.image(final_overlay, caption="💉 Highlighted Disease Region", use_container_width=True)


#         else:
#             st.info("⚠️ Segmentation is available only for **Diabetic Macular Edema (DME)** cases.")
# else:
#     st.info("📁 Please upload an OCT image to begin.")















import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# =====================================================
# 1️⃣ DEVICE SETUP
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="OCT Analyzer", page_icon="👁️", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #1E90FF;'>👁️ Retinal OCT Analyzer</h1>
    <h4 style='text-align: center; color: #808080;'>
        AI-powered tool for disease classification and DME lesion segmentation
    </h4>
    <hr style='border:1px solid #ccc'>
""", unsafe_allow_html=True)


# =====================================================
# 2️⃣ CLASSIFICATION MODEL (ResNet18)
# =====================================================
def build_classification_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 4)
    return model


@st.cache_resource
def load_classification_model():
    model = build_classification_model()
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading classification model: {e}")
        return None


# =====================================================
# 3️⃣ SEGMENTATION MODEL (U-Net)
# =====================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


@st.cache_resource
def load_segmentation_model():
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    try:
        state_dict = torch.load("segmentation_best_unet_model.pth", map_location=DEVICE)
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading segmentation model: {e}")
        return None


# =====================================================
# 4️⃣ PREPROCESSING & LABELS
# =====================================================
transform_clf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
DESCRIPTIONS = {
    "CNV": "🩸 **Choroidal Neovascularization (CNV)** — abnormal blood vessels grow under the retina.",
    "DME": "💧 **Diabetic Macular Edema (DME)** — fluid accumulates in the macula.",
    "DRUSEN": "🟡 **Drusen** — yellow lipid deposits under the retina, early sign of AMD.",
    "NORMAL": "✅ **Normal Retina** — healthy retinal structure with no abnormalities."
}


# =====================================================
# 5️⃣ PREDICTION FUNCTIONS
# =====================================================
def predict_class(model, image):
    img = transform_clf(image.convert("L")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return CLASSES[idx.item()], conf.item() * 100


def preprocess_seg(image):
    img_gray = image.convert("L").resize((512, 512))
    img = np.array(img_gray, dtype=np.float32)
    img = (img - img.mean()) / (img.std() + 1e-8)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor


def predict_mask(model, image, threshold=0.25):
    tensor = preprocess_seg(image)
    with torch.no_grad():
        output = torch.sigmoid(model(tensor))
    mask = output.squeeze().cpu().numpy()

    # Smooth and threshold
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = (mask > threshold).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


# =====================================================
# 6️⃣ APP LOGIC
# =====================================================
clf_model = load_classification_model()
seg_model = load_segmentation_model()

uploaded_file = st.file_uploader("📤 Upload an OCT image", type=["jpg", "jpeg", "png"])

if uploaded_file and clf_model and seg_model:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    # LEFT COLUMN - Classification
    with col1:
        st.image(image, caption="Uploaded OCT Image", width=320)
        with st.spinner("🔍 Classifying..."):
            label, conf = predict_class(clf_model, image)
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {conf:.2f}%")
        st.markdown(DESCRIPTIONS[label])

    # RIGHT COLUMN - Segmentation (DME only)
    with col2:
        st.markdown("<h3 style='text-align:center; color:#00B050;'>🩺 DME Lesion Segmentation</h3>", unsafe_allow_html=True)

        if label == "DME":
            with st.spinner("🧩 Detecting diseased region..."):
                mask = predict_mask(seg_model, image)

            mask_uint8 = (mask * 255).astype(np.uint8)
            disease_area = (mask.sum() / mask.size) * 100

            base_rgb = np.array(image.convert("RGB").resize(mask.shape[::-1]))
            red_overlay = np.zeros_like(base_rgb)
            red_overlay[mask == 1] = [255, 0, 0]
            highlighted = cv2.addWeighted(base_rgb, 0.7, red_overlay, 0.5, 0)

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = [c for c in contours if cv2.contourArea(c) > 10]
            for c in filtered:
                cv2.drawContours(highlighted, [c], -1, (0, 255, 0), 2)

            st.markdown(f"<h4 style='text-align:center; color:#E74C3C;'>🩸 Detected DME Area: {disease_area:.2f}% of image</h4>", unsafe_allow_html=True)

            col3, col4 = st.columns(2)
            with col3:
                st.image(mask_uint8, caption="🧠 Binary Mask", width=360)
            with col4:
                st.image(highlighted, caption="🩺 Highlighted DME Region", width=360)

            if disease_area < 0.5:
                st.info("✅ Minimal or no lesion detected — likely normal or early-stage.")

        else:
            st.info("⚠️ Segmentation is available only for **Diabetic Macular Edema (DME)** cases.")
else:
    st.info("📁 Please upload an OCT image to begin.")
