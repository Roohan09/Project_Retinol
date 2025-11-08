import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io




# --- 1. Model Architecture (Copied from your notebook) ---

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-create the ResNet18 model structure
def build_model():
    model = models.resnet18(weights=None)  # Not using pretrained weights
    
    # Modify for 1-channel (grayscale) input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify for 4 output classes
    model.fc = nn.Linear(model.fc.in_features, 4)
    
    return model

# --- 2. Load the Trained Weights ---

@st.cache_resource
def load_model():
    model = build_model()
    model_path = "best_model.pth"  # Make sure this file is in the same folder
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please make sure 'best_model.pth' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# --- 3. Preprocessing and Prediction Functions ---

# Define the image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Define class names
CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# --- (NEW) Disease Descriptions ---
DISEASE_DESCRIPTIONS = {
    "CNV": "**Choroidal Neovascularization (CNV)** is the growth of new, abnormal blood vessels from the choroid layer into the retina. These new vessels are leaky and can let fluid or blood into the retina. This causes rapid and severe vision loss and is a key sign of 'wet' Age-related Macular Degeneration (AMD).",
    "DME": "**Diabetic Macular Edema (DME)** is a complication of diabetes. It's caused by damaged blood vessels leaking fluid into the macula (the center of the retina), causing it to swell. This swelling leads to blurry or wavy central vision. OCT is the standard method for diagnosing this swelling.",
    "DRUSEN": "**Drusen** are small, yellow deposits of proteins and lipids (fats) that build up under the retina. While a few small 'hard' drusen are a normal part of aging, the presence of many large, 'soft' drusen is a common early sign of 'dry' Age-related Macular Degeneration (AMD).",
    "NORMAL": "The OCT scan appears **Normal**. The retinal layers are distinct and there are no signs of fluid accumulation (edema), abnormal blood vessel growth (CNV), or significant drusen deposits."
}
# --- End of New Section ---

def predict(model, image):
    """Takes a PIL image, transforms it, and returns the predicted class and confidence."""
    img_gray = image.convert('L')
    img_tensor = transform(img_gray).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
    
    return CLASSES[pred_idx.item()], confidence.item() * 100

# --- 4. Streamlit Web App UI ---

st.set_page_config(page_title="OCT Classifier", layout="wide")
st.title("👁️ Retinal OCT Image Classifier")
st.write("Upload an OCT image to classify it as CNV, DME, DRUSEN, or NORMAL.")

# Load the model
model = load_model()

if model is not None:
    uploaded_file = st.file_uploader("Choose an OCT image...", type=["jpeg", "jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # --- UPDATED ---
            # Used use_container_width as you pointed out
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner('Classifying...'):
                label, confidence = predict(model, image)
            
            st.success(f"**Predicted Class:** {label}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            
            # --- NEW: Display the description ---
            st.write("---")
            st.subheader(f"About {label}")
            # Get the description from the dictionary
            description = DISEASE_DESCRIPTIONS.get(label, "No description available.")
            st.write(description)
            # --- End of New Section ---

else:
    st.warning("Model could not be loaded. Please check the `best_model.pth` file.")

# --- (NEW) Optional: Add a sidebar with all descriptions ---
st.sidebar.title("About the Classifications")
for disease, description in DISEASE_DESCRIPTIONS.items():
    with st.sidebar.expander(f"**{disease}**"):
        st.write(description)

# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# from segmentation_model import UNet  # 👈 Your U-Net definition file


# # --------------------------
# # 🔹 Device Setup
# # --------------------------
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# st.set_page_config(page_title="OCT Classifier + Segmenter", layout="wide")
# st.title("👁️ Retinal OCT Image Analysis")
# st.write("Upload an OCT scan to classify and visualize retinal abnormalities.")

# # --------------------------
# # 🔹 Model Builders
# # --------------------------
# def build_classification_model():
#     model = models.resnet18(weights=None)
#     model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     model.fc = nn.Linear(model.fc.in_features, 4)
#     return model


# @st.cache_resource
# def load_classification_model():
#     model = build_classification_model()
#     model_path = "best_model.pth"
#     try:
#         model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#         model.to(DEVICE).eval()
#         return model
#     except Exception as e:
#         st.error(f"❌ Error loading classification model: {e}")
#         return None


# @st.cache_resource
# def load_segmentation_model():
#     model = UNet(n_channels=1, n_classes=1)
#     seg_model_path = "segmentation_best_unet_model.pth"
#     try:
#         model.load_state_dict(torch.load(seg_model_path, map_location=DEVICE))
#         model.to(DEVICE).eval()
#         return model
#     except Exception as e:
#         st.error(f"❌ Error loading segmentation model: {e}")
#         return None


# # --------------------------
# # 🔹 Transforms
# # --------------------------
# transform_classification = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# transform_segmentation = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((256, 256)),  # Match your segmentation training size
#     transforms.ToTensor()
# ])

# # --------------------------
# # 🔹 Class Names + Info
# # --------------------------
# CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
# DISEASE_DESCRIPTIONS = {
#     "CNV": "**Choroidal Neovascularization (CNV)**: Abnormal new blood vessels grow under the retina, leading to fluid leakage and vision loss.",
#     "DME": "**Diabetic Macular Edema (DME)**: Fluid accumulation in the retina due to leaky blood vessels from diabetic damage.",
#     "DRUSEN": "**Drusen**: Yellow lipid deposits beneath the retina, common in early Age-related Macular Degeneration (AMD).",
#     "NORMAL": "Retinal structure appears **normal** — no fluid or abnormal vessel growth detected."
# }


# # --------------------------
# # 🔹 Prediction Functions
# # --------------------------
# def predict_classification(model, image):
#     img_gray = image.convert('L')
#     img_tensor = transform_classification(img_gray).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         outputs = model(img_tensor)
#         probs = torch.softmax(outputs, dim=1)
#         confidence, pred_idx = torch.max(probs, dim=1)

#     return CLASSES[pred_idx.item()], confidence.item() * 100


# def predict_segmentation(model, image):
#     img_gray = image.convert('L')
#     img_tensor = transform_segmentation(img_gray).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         mask = torch.sigmoid(model(img_tensor))
#         mask = mask.squeeze().cpu().numpy()
#         mask = (mask > 0.5).astype(np.uint8)
#     return mask


# # --------------------------
# # 🔹 Load Models
# # --------------------------
# clf_model = load_classification_model()
# seg_model = load_segmentation_model()


# # --------------------------
# # 🔹 Streamlit UI
# # --------------------------
# uploaded_file = st.file_uploader("📤 Upload an OCT image", type=["jpg", "jpeg", "png"])

# if uploaded_file and clf_model and seg_model:
#     image = Image.open(uploaded_file)

#     col1, col2 = st.columns(2)

#     with col1:
#         st.image(image, caption="Uploaded OCT Scan", use_container_width=True)

#     with col2:
#         with st.spinner("🧠 Analyzing image..."):
#             label, confidence = predict_classification(clf_model, image)
#             mask = predict_segmentation(seg_model, image)

#         st.success(f"**Prediction:** {label}")
#         st.info(f"**Confidence:** {confidence:.2f}%")

#         st.markdown("---")
#         st.subheader(f"🩺 Disease Information")
#         st.write(DISEASE_DESCRIPTIONS.get(label, "No description available."))

#     # --------------------------
#     # 🔹 Segmentation Visualization
#     # --------------------------
#     st.markdown("---")
#     st.subheader("🧩 Segmentation Result")

#     mask_col1, mask_col2 = st.columns(2)

#     with mask_col1:
#         st.image(mask * 255, caption="Segmentation Mask", use_container_width=True)

#     with mask_col2:
#         # Overlay mask on image (red areas)
#         overlay = np.array(image.convert("RGB")).copy()
#         overlay = np.array(Image.fromarray(overlay).resize(mask.shape[::-1]))
#         overlay[mask == 1] = [255, 0, 0]  # Highlight segmented regions in red
#         st.image(overlay, caption="Overlay (Lesion Highlighted)", use_container_width=True)

# else:
#     st.warning("⚠️ Please upload an OCT image to begin.")
