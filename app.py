import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

st.set_page_config(page_title="EcoAI Tracker Dashboard", layout="wide")

st.title("🌲 AI-Driven Deforestation & Fragmentation Tracker")
st.markdown("""
This model processes uploaded satellite fields and passes the imagery array to a multi-class pixel classification model. 
The AI explicitly isolates **natural forest canopy** from **deforested clearings** and **artificial green objects**.
""")


@st.cache_resource
def load_trained_ai_model():
    try:
        with open("deforestation_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


ai_model = load_trained_ai_model()

st.sidebar.header("Data Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload an Image Tile", type=["jpg", "png", "jpeg"])

if ai_model is None:
    st.error(
        "🚨 Critical Error: 'deforestation_model.pkl' file was not detected. Please run 'sample_data_generator.py' then 'train.py' first.")
elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛰️ Uploaded Input View")
        st.image(img_rgb, use_column_width=True)

    with st.spinner("AI Engine performing pixel structural analysis..."):
        R = img_rgb[:, :, 0].astype(float)
        G = img_rgb[:, :, 1].astype(float)
        ndvi = (G - R) / (G + R + 1e-5)

        pixels = img_rgb.reshape(-1, 3)
        ndvi_flat = ndvi.reshape(-1, 1)
        features = np.hstack((pixels, ndvi_flat))

        predictions = ai_model.predict(features)
        prediction_mask = predictions.reshape(img_rgb.shape[0], img_rgb.shape[1])

        analysis_map = img_rgb.copy()
        analysis_map[prediction_mask == 0] = [220, 20, 60]  # Class 0 -> Crimson Red (Soil / Clearings)
        analysis_map[prediction_mask == 2] = [0, 120, 255]  # Class 2 -> Electric Blue (Green Houses / Structures)

        total_pixels = prediction_mask.size
        canopy_pixels = np.sum(prediction_mask == 1)
        deforested_pixels = np.sum(prediction_mask == 0)
        artificial_pixels = np.sum(prediction_mask == 2)

        canopy_pct = (canopy_pixels / total_pixels) * 100
        deforested_pct = (deforested_pixels / total_pixels) * 100
        artificial_pct = (artificial_pixels / total_pixels) * 100

    with col2:
        st.subheader("📊 Multi-Class AI Overlay")
        st.image(analysis_map, use_column_width=True)

    st.markdown("---")
    st.subheader("📈 Environmental Metrics Breakdown")

    m1, m2, m3 = st.columns(3)
    m1.metric(label="Protected Canopy Coverage", value=f"{canopy_pct:.1f}%")
    m2.metric(label="Canopy Loss / Clearing", value=f"{deforested_pct:.1f}%",
              delta="Critical Fragmentation Risk" if deforested_pct > 15 else "Stable")
    m3.metric(label="Detected Green Man-Made Objects", value=f"{artificial_pixels} pixels",
              delta=f"{artificial_pct:.2f}% Surface Area")

    st.markdown("""
    **Map Visualization Key:** 🟥 **Crimson Red Zones:** Canopy Loss / Ground Cleared / Soil Roads  
    🟦 **Electric Blue Points:** AI-Isolated Green Man-made Structures (Greenhouses, roofs, artificial objects)  
    🟩 **Natural Colors:** Verified Intact Continuous Forest Canopy
    """)

    # === SOIL ANALYSIS & AFFORESTATION STRATEGY ===
    st.markdown("---")
    st.subheader("🌱 AI Soil Profiler & Native Afforestation Planner")

    if deforested_pixels > 0:
        soil_pixels = img_rgb[prediction_mask == 0]
        mean_rgb = np.mean(soil_pixels, axis=0)

        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        soil_only_gray = gray_img[prediction_mask == 0]
        soil_variance = np.var(soil_only_gray) if len(soil_only_gray) > 0 else 0

        sc1, sc2 = st.columns(2)
        with sc1:
            st.write(
                f"**Dominant Spectral Profile (Mean RGB):** R:{int(mean_rgb[0])}, G:{int(mean_rgb[1])}, B:{int(mean_rgb[2])}")
            if mean_rgb[0] > mean_rgb[1] * 1.2:
                soil_type = "Arid / Sandy / Laterite (Reddish/Low Moisture Content)"
                plant_recommendation = "🌵 **Recommended Indigenous Species:** Neem (*Azadirachta indica*), Khejri (*Prosopis cineraria*), Ber, and Babool. These are drought-resistant thorn and scrub species ideal for dry soil zones."
                method_tip = "🚜 **Restoration Method:** Implement contour trenching and rain-water harvesting structures to trap monsoon moisture before planting saplings."
            else:
                soil_type = "Alluvial / Clay Loam / Humus-Rich (Darker/Moderate-to-High Moisture Content)"
                plant_recommendation = "🌳 **Recommended Indigenous Species:** Peepal (*Ficus religiosa*), Arjun (*Terminalia arjuna*), Jamun, and Mahua. These broad-leaved trees support high native biodiversity."
                method_tip = "🌾 **Restoration Method:** Encourage Joint Forest Management (JFM) networks to monitor agroforestry buffers and establish structural wildlife corridors."

            st.info(f"**Identified Ground Profile:** {soil_type}")

        with sc2:
            texture_desc = "Rough / Patchy / High Erosion Vulnerability" if soil_variance > 500 else "Smooth / Homogeneous / Uniform Topography"
            st.write(f"**Calculated Texture Variance Value:** {soil_variance:.1f}")
            st.info(f"**Surface Texture Context:** {texture_desc}")

        st.markdown("### 📋 Targeted Ecosystem Interventions")
        st.markdown(plant_recommendation)
        st.markdown(method_tip)
    else:
        st.success(
            "Canopy cover is at 100%. No exposed or degraded soil patches were detected for afforestation planning.")

    # === ECOLOGICAL SURVEILLANCE DISCLAIMER ===
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚠️ LEGAL & ECOLOGICAL Surveillance Disclaimer"):
        st.markdown("""
        * **Algorithmic Limitations:** This module relies strictly on multi-spectral reflectance and pixel color/texture distribution data generated via satellite imagery. It serves as an exploratory decision-support utility and **cannot** replace comprehensive core laboratory physical/chemical soil testing (such as evaluating NPK nitrogen balances, exact pH measurements, or heavy metal contamination).
        * **Geographic Variation Alert:** The suggestions provided by this algorithm are tailored primarily around native Indian sub-continental biomes (Deciduous and Scrubland ecosystem guidelines). These results **may not be suitable or ecologically sound** for alpine montane zones, salt-heavy coastal delta mangroves, or highly localized hyper-arid desert systems.
        * **Biosecurity Notice:** Before conducting large-scale afforestation campaigns, users must consult localized field surveys conducted by respective state forest departments or agricultural extension centers to ensure non-invasive biodiversity management.
        """)
else:
    st.info("💡 Standby Mode: Use the sidebar dashboard to upload an input matrix image.")