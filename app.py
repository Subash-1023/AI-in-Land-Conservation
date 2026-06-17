import streamlit as st
import cv2
import numpy as np
import pickle

st.set_page_config(page_title="EcoAI Tracker Dashboard", layout="wide")

st.title("🌲 AI-Driven Deforestation & Landscape Tracker")
st.markdown("This enhanced version uses **Min-Max Feature Normalization** to accurately classify both synthetic templates and real-world satellite imagery.")

@st.cache_resource
def load_normalized_model():
    try:
        with open("deforestation_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

ai_model = load_normalized_model()

st.sidebar.header("Data Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])

if ai_model is None:
    st.error("🚨 Missing Model File. Please execute 'sample_data_generator.py' then 'train.py' within your environment setup.")
elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛰️ Uploaded Input View")
        st.image(img_rgb, use_column_width=True)
        
    with st.spinner("Executing Normalized Model Analysis..."):
        # Match image normalization precisely to train.py parameters
        img_float = img_rgb.astype(float)
        min_val = img_float.min()
        max_val = img_float.max()
        
        if max_val > min_val:
            img_normalized = (img_float - min_val) / (max_val - min_val)
        else:
            img_normalized = img_float / 255.0
            
        R = img_normalized[:, :, 0]
        G = img_normalized[:, :, 1]
        ndvi = (G - R) / (G + R + 1e-5)
        
        pixels_flat = img_normalized.reshape(-1, 3)
        ndvi_flat = ndvi.reshape(-1, 1)
        features = np.hstack((pixels_flat, ndvi_flat))
        
        # Execute Robust Inference
        predictions = ai_model.predict(features)
        prediction_mask = predictions.reshape(img_rgb.shape[0], img_rgb.shape[1])
        
        # Build Overlay
        analysis_map = img_rgb.copy()
        
        # CRITICAL REASSIGMENT FIX: Keep Forest natural, map soil to red, manmade/sky to blue
        analysis_map[prediction_mask == 0] = [220, 20, 60]  # Class 0 -> Crimson Red (Soil)
        analysis_map[prediction_mask == 2] = [0, 120, 255]  # Class 2 -> Electric Blue (Sky/Structures)
        
        # Calculate accurate statistics
        total_pixels = prediction_mask.size
        canopy_pct = (np.sum(prediction_mask == 1) / total_pixels) * 100
        deforested_pct = (np.sum(prediction_mask == 0) / total_pixels) * 100
        artificial_pct = (np.sum(prediction_mask == 2) / total_pixels) * 100

    with col2:
        st.subheader("📊 Multi-Class AI Overlay")
        st.image(analysis_map, use_column_width=True)
        
    st.markdown("---")
    st.subheader("📈 Environmental Metrics Breakdown")
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Protected Canopy Coverage", value=f"{canopy_pct:.1f}%")
    m2.metric(label="Canopy Loss / Ground Clearings", value=f"{deforested_pct:.1f}%")
    m3.metric(label="Isolated Features (Sky / Structures)", value=f"{artificial_pct:.1f}%")
    
    # === BOTANICAL RESTORATION LOGIC ===
    st.markdown("---")
    st.subheader("🌱 AI Soil Profiler & Native Afforestation Planner")
    
    if deforested_pct > 0:
        soil_pixels = img_rgb[prediction_mask == 0]
        mean_rgb = np.mean(soil_pixels, axis=0) if len(soil_pixels) > 0 else [0,0,0]
        
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        soil_only_gray = gray_img[prediction_mask == 0]
        soil_variance = np.var(soil_only_gray) if len(soil_only_gray) > 0 else 0
        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.write(f"**Soil Spectral Signatures:** R:{int(mean_rgb[0])}, G:{int(mean_rgb[1])}, B:{int(mean_rgb[2])}")
            # If red reflection dominates, it indicates dry/sandy/laterite traits
            if mean_rgb[0] > mean_rgb[1]:
                st.info("Ground Profile Identified: Arid / Sandy / Semi-Desert Sheet Terrain")
                st.markdown("🌵 **Recommended Species:** Neem (*Azadirachta indica*), Khejri (*Prosopis cineraria*), and Babool.")
            else:
                st.info("Ground Profile Identified: Moist Alluvial Plain / Silt / Active Humus Zone")
                st.markdown("🌳 **Recommended Species:** Peepal (*Ficus religiosa*), Arjun (*Terminalia arjuna*), and Bamboo clusters.")
        with sc2:
            st.write(f"**Topographical Structural Variance:** {soil_variance:.1f}")
            st.info("Surface Texture Context: High Local Erosion Risk" if soil_variance > 400 else "Surface Texture Context: Flat / Stable Ground Layer")
            
    else:
        st.success("Canopy cover is intact across the frame. No exposed soil requires afforestation interventions.")
        
    with st.expander("⚠️ Legal & Ecological Disclaimer"):
        st.markdown("This program serves as a computational decision-support tool using multi-spectral reflectance data. It cannot replace in-person laboratory chemical topsoil tests (pH or NPK analysis).")
else:
    st.info("💡 Control Panel: Upload any satellite or real aerial field view to execute model testing pipelines.")
