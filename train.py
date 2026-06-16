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

print("📦 Loading multi-class training data...")

for file_name in os.listdir(IMAGE_DIR):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(IMAGE_DIR, file_name)
        mask_path = os.path.join(MASK_DIR, file_name)

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Feature Engineering: Compute NDVI spectral layer
        R = img[:, :, 0].astype(float)
        G = img[:, :, 1].astype(float)
        ndvi = (G - R) / (G + R + 1e-5)

        pixels = img.reshape(-1, 3)
        ndvi_flat = ndvi.reshape(-1, 1)
        features = np.hstack((pixels, ndvi_flat))
        labels = mask.reshape(-1)

        # Split target categories
        class_labels = np.zeros_like(labels)
        class_labels[labels > 200] = 1  # Class 1: Real Forest Canopy
        class_labels[(labels >= 100) & (labels <= 200)] = 2  # Class 2: Artificial Green Objects

        X_data.append(features)
        Y_data.append(class_labels)

X = np.vstack(X_data)
Y = np.concatenate(Y_data)

# Protect system memory boundaries
sample_size = min(100000, X.shape[0])
subsample_idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
X = X[subsample_idx]
Y = Y[subsample_idx]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("🤖 Training Random Forest classification layers...")
model = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100
print(f"✅ Model Training Complete. Internal Validation Accuracy: {accuracy:.2f}%")

with open("deforestation_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("💾 Machine Learning Weights compiled as 'deforestation_model.pkl'!")