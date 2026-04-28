"""
app.py — Streamlit UI for Image Feature-Based Prediction using Linear Regression
Authors: Vidyanand Yadav & Ayush Maurya | B.Tech CSE | CSJMU Kanpur
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Regression | CSJMU",
    page_icon="🖼️",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style='background-color:#1a237e;padding:14px 20px;border-radius:10px;margin-bottom:20px'>
        <h3 style='color:white;text-align:center;margin:0;font-size:20px'>
            🖼️ Image Feature-Based Prediction using Linear Regression
        </h3>
        <p style='color:#c5cae9;text-align:center;margin:5px 0 0 0;font-size:13px'>
            Vidyanand Yadav (CSJMA23001390066) &nbsp;|&nbsp;
            Ayush Maurya (CSJMA23001390059) &nbsp;|&nbsp;
            B.Tech CSE 3rd Year &nbsp;|&nbsp; CSJMU Kanpur
        </p>
    </div>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────
IMAGES_FOLDER = "images"
SUPPORTED = ('.jpg', '.jpeg', '.png', '.bmp')

def load_images(folder):
    images = []
    if not os.path.exists(folder):
        return images
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(SUPPORTED):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is not None:
                images.append((fname, img))
    return images

def extract_features(images):
    records = []
    for filename, img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        edges_img = cv2.Canny(gray, 100, 200)
        edge_count = np.sum(edges_img > 0)
        records.append({'filename': filename,
                        'brightness': round(brightness, 4),
                        'edges': edge_count})
    return pd.DataFrame(records)

def create_target(df):
    df['price'] = (df['brightness'] * 0.5) + (df['edges'] * 0.01)
    np.random.seed(42)
    noise = np.random.normal(0, 5, size=len(df))
    df['price'] = (df['price'] + noise).round(4)
    return df

def train_model(df):
    X = df[['brightness', 'edges']].values
    y = df['price'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg", width=100)
st.sidebar.markdown("## ⚙️ Project Info")
st.sidebar.markdown("""
- **Algorithm:** Linear Regression
- **Features:** Brightness, Edge Count
- **Target:** Simulated Price
- **Train/Test:** 80% / 20%
- **Images:** 30 synthetic images
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Formula:**")
st.sidebar.code("price = brightness×0.5 + edges×0.01 + noise")
st.sidebar.markdown("---")
st.sidebar.markdown("**Libraries Used:**")
st.sidebar.markdown("`OpenCV` `NumPy` `Pandas` `Scikit-learn` `Matplotlib`")

# ── Main content ──────────────────────────────────────────────────────────────
images = load_images(IMAGES_FOLDER)

if not images:
    st.error("❌ No images found in `images/` folder. Please run `generate_sample_images.py` first.")
    st.stop()

# ── Section 1: Sample Image Preview ──────────────────────────────────────────
st.markdown("### 📷 Sample Image — Color vs Grayscale")
fname, img = images[0]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image(img_rgb, caption="Original (Color)", use_container_width=True)
with col2:
    st.image(gray, caption="After Grayscale Conversion", use_container_width=True, clamp=True)
with col3:
    edges_img = cv2.Canny(gray, 100, 200)
    st.image(edges_img, caption="Canny Edge Detection", use_container_width=True, clamp=True)

st.markdown("---")

# ── Section 2: Dataset ────────────────────────────────────────────────────────
st.markdown("### 📊 Extracted Features Dataset")
df = extract_features(images)
df = create_target(df)
model, X_test, y_test, y_pred = train_model(df)
df['predicted_price'] = model.predict(df[['brightness', 'edges']].values).round(4)

st.dataframe(df.style.format({
    'brightness': '{:.2f}',
    'edges': '{:,}',
    'price': '{:.2f}',
    'predicted_price': '{:.2f}'
}), use_container_width=True)

st.markdown("---")

# ── Section 3: Metrics ────────────────────────────────────────────────────────
st.markdown("### 📈 Model Performance Metrics")
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("R² Score",      f"{r2:.4f}")
m2.metric("MSE",           f"{mse:.4f}")
m3.metric("MAE",           f"{mae:.4f}")
m4.metric("Training Size", f"{int(len(df)*0.8)} images")
m5.metric("Test Size",     f"{int(len(df)*0.2)} images")

st.markdown("---")

# ── Section 4: Plots ──────────────────────────────────────────────────────────
st.markdown("### 📉 Visualizations")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Image Feature-Based Prediction using Linear Regression", fontsize=13)

# Plot 1
ax1.scatter(y_test, y_pred, color='steelblue', edgecolors='black', s=80, label='Predictions')
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax1.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Ideal fit')
ax1.set_xlabel("Actual Price"); ax1.set_ylabel("Predicted Price")
ax1.set_title("Actual vs Predicted Price")
ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.5)

# Plot 2
ax2.scatter(df['brightness'], df['price'], color='darkorange',
            edgecolors='black', s=80, label='Actual Price')
ax2.scatter(df['brightness'], df['predicted_price'], color='green',
            marker='^', s=80, label='Predicted Price')
ax2.set_xlabel("Brightness (mean pixel intensity)"); ax2.set_ylabel("Price")
ax2.set_title("Brightness vs Price")
ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ── Section 5: Predict on uploaded image ─────────────────────────────────────
st.markdown("### 🔍 Try it — Upload Your Own Image")
uploaded = st.file_uploader("Upload a JPG/PNG image to predict its price",
                             type=['jpg', 'jpeg', 'png'])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_up = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray_up = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    brightness_up = np.mean(gray_up)
    edges_up = np.sum(cv2.Canny(gray_up, 100, 200) > 0)
    pred_price = model.predict([[brightness_up, edges_up]])[0]

    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(img_up, cv2.COLOR_BGR2RGB),
                 caption="Uploaded Image", use_container_width=True)
    with c2:
        st.markdown("#### Extracted Features")
        st.markdown(f"- **Brightness:** `{brightness_up:.2f}`")
        st.markdown(f"- **Edge Count:** `{edges_up}`")
        st.markdown(f"#### 💰 Predicted Price")
        st.success(f"**{pred_price:.2f}**")

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray'>B.Tech CSE | CSJMU Kanpur | April 2026</p>",
            unsafe_allow_html=True)
