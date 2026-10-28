"""
Streamlit App — Pneumonia Detection from Chest X-Rays
Users upload a DICOM or PNG/JPG image and get predicted class + probability.
"""

import streamlit as st
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.markdown("""
Upload a chest X-ray image (JPG, PNG, or DICOM .dcm) and the AI model will classify it as:
- ✅ **Normal** — No pneumonia detected
- ⚠️ **Lung Opacity** — Pneumonia likely present
- 🔶 **No Lung Opacity / Not Normal** — Abnormality present, but not pneumonia
""")

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best_model_final.h5')

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model not found: {e}")
    st.stop()

CLASS_NAMES = ['Normal', 'Lung Opacity (Pneumonia)', 'No Lung Opacity / Not Normal']
IMG_SIZE    = 224

# ── Image upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png", "dcm"]
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)

    # Decode image
    if uploaded_file.name.endswith('.dcm'):
        import pydicom, tempfile
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp:
            tmp.write(file_bytes.tobytes())
            tmp_path = tmp.name
        ds   = pydicom.dcmread(tmp_path)
        img  = ds.pixel_array.astype(np.float32)
        img  = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        img  = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    else:
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Display uploaded image
    st.image(img, caption='Uploaded X-Ray', use_column_width=True, clamp=True)

    # Preprocess
    img_norm = img.astype(np.float32) / 255.0
    img_rgb  = np.stack([img_norm, img_norm, img_norm], axis=-1)
    img_input = np.expand_dims(img_rgb, axis=0)

    # Predict
    if st.button("🔬 Analyse X-Ray", type="primary"):
        with st.spinner("Analysing..."):
            pred_probs = model.predict(img_input, verbose=0)[0]
            pred_class = np.argmax(pred_probs)
            pred_label = CLASS_NAMES[pred_class]
            confidence = pred_probs[pred_class] * 100

        # Display result
        if pred_class == 0:
            st.success(f"✅ **{pred_label}** — No pneumonia detected")
        elif pred_class == 1:
            st.error(f"⚠️ **{pred_label}** — Pneumonia likely. Consult a radiologist immediately.")
        else:
            st.warning(f"🔶 **{pred_label}** — Abnormality detected. Further evaluation recommended.")

        st.metric("Confidence", f"{confidence:.1f}%")

        # Probability breakdown
        st.subheader("Probability Distribution")
        for cls, prob in zip(CLASS_NAMES, pred_probs):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

        st.caption("⚠️ This tool is a decision-support aid only. Always consult a qualified radiologist for diagnosis.")

st.markdown("---")
st.caption("Powered by ResNet50 Transfer Learning | TensorFlow | Streamlit")
