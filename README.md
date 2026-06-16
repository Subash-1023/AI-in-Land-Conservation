# 🌲 EcoAI Tracker: Multi-Class Deforestation Detection & Soil-Driven Afforestation Planner

An interactive machine learning prototype that uses a multi-class Random Forest pixel classifier to detect deforestation from satellite imagery while explicitly mitigating data bias from green man-made structures (like greenhouses and roofs). It also includes an automated ecological engine that analyzes exposed soil color and texture to suggest native, indigenous plant species for targeted afforestation.

# 🚀 Key FeaturesMulti-Class Segmentation Architecture: Moves beyond simple binary masking to separate land covers into three distinct classes:

🟥 Crimson Red: Deforested Clearings / Bare Soil Roads (Class 0)

🟩 Natural Colors: Verified Intact Forest Canopy (Class 1)

🟦 Electric Blue: Isolated Green Man-Made Objects / Structures (Class 2)

**Feature-Engineered AI Backend**: Uses a customized, local $NDVI$ (Normalized Difference Vegetation Index) calculation combined with RGB spectral data.

**AI Soil Profiler**: Analyzes mean RGB color indices and spatial texture variance of cleared ground to identify soil erosion risk and dry vs. rich terrain.

**Native Afforestation Planner**: Automatically suggests indigenous tree species (e.g., Neem, Peepal, Khejri, Arjun) aligned with local conservation frameworks (like India's Joint Forest Management parameters).

**Responsive Web Dashboard**: Built entirely using Streamlit for fast, user-friendly remote sensing audits.
