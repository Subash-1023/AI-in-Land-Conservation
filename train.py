import cv2
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

IMAGE_DIR = "training_images"
MASK_DIR = "training_masks"

X_data = []
Y_data = []

print("📦 Loading arrays and running feature normalization...")

for file_name in os.listdir(IMAGE_DIR):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(IMAGE_DIR, file_name)
        mask_path = os.path.join(MASK_DIR, file_name)
        
        if not os.path.exists(mask_path):
            continue
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # CRITICAL FIX: Min-Max Normalization per image file
        img_float = img.astype(float)
        min_val = img_float.min()
        max_val = img_float.max()
        if max_val > min_val:
            img_normalized = (img_float - min_val) / (max_val - min_val)
        else:
            img_normalized = img_float / 255.0
            
        # Feature Engineering: Compute Robust Visible NDVI over standardized arrays
        R = img_normalized[:, :, 0]
        G = img_normalized[:, :, 1]
        ndvi = (G - R) / (G + R + 1e-5)
        
        # Flatten and stack features cleanly
        pixels_flat = img_normalized.reshape(-1, 3)
        ndvi_flat = ndvi.reshape(-1, 1)
        features = np.hstack((pixels_flat, ndvi_flat))
        labels = mask.reshape(-1)
        
        # Categorical Map Coding
        class_labels = np.zeros_like(labels)
        class_labels[labels > 200] = 1                      # Class 1: Forest Canopy
        class_labels[(labels >= 100) & (labels <= 200)] = 2 # Class 2: Man-made Structure
        # 0 remains default for Class 0 (Soil)
        
        X_data.append(features)
        Y_data.append(class_labels)

X = np.vstack(X_data)
Y = np.concatenate(Y_data)

# Downsample slightly to keep execution snappy on consumer hardware
sample_size = min(150000, X.shape[0])
subsample_idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
X = X[subsample_idx]
Y = Y[subsample_idx]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("🤖 Training normalized Multi-Class Random Forest Model...")
# Increased estimators and depth to better capture real geometric limits
model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100
print(f"✅ Balanced Core Training Complete. Test Accuracy: {accuracy:.2f}%")

with open("deforestation_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("💾 Normalization weights saved securely as 'deforestation_model.pkl'!")
