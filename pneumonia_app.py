import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# -----------------------------
# Load the Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 2)  # 2 classes: NORMAL, PNEUMONIA
    model.load_state_dict(torch.load("pneumonia_model_resnet50.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image Transformations
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probabilities).item()

    classes = ["NORMAL", "PNEUMONIA"]
    return classes[prediction], probabilities[prediction].item()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺", layout="wide")

st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.markdown("""
Upload a **Chest X-ray Image** and the model will classify it as:

- **NORMAL**
- **PNEUMONIA**

This model uses **ResNet50 Transfer Learning** trained on the Kaggle X-Ray dataset.
""")

uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    with col2:
        st.subheader("Prediction")

        label, confidence = predict(image)

        st.write(f"### 🧾 Result: **{label}**")
        st.write(f"### 🔢 Confidence: **{confidence*100:.2f}%**")

        if label == "PNEUMONIA":
            st.error("⚠ Pneumonia detected. Consult a medical professional.")
        else:
            st.success("✔ Normal lung X-ray detected.")

st.markdown("---")
st.caption("AI-powered Pneumonia Detector – built with PyTorch & Streamlit")
