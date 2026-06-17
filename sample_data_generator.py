import cv2
import numpy as np
import os

def generate_robust_dataset():
    os.makedirs("training_images", exist_ok=True)
    os.makedirs("training_masks", exist_ok=True)
    
    print("🎨 Creating enhanced training dataset with variable color tones...")
    
    # We generate multiple variations (vibrant, dark, hazy) to teach the AI flexibility
    tonalities = [
        {"forest": [34, 139, 34], "soil": [139, 69, 19], "manmade": [124, 252, 0]}, # Vibrant
        {"forest": [45, 65, 45],   "soil": [105, 80, 60],  "manmade": [100, 130, 90]}, # Dark Olive Haze
        {"forest": [50, 85, 50],   "soil": [120, 90, 70],  "manmade": [110, 140, 100]}  # Moderate Shadow
    ]
    
    file_idx = 0
    for t in tonalities:
        for i in range(2):
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            mask = np.zeros((512, 512), dtype=np.uint8)
            
            # Class 1: Background Canopy
            img[:] = t["forest"]
            mask[:] = 255 
            
            # Class 0: Soil Clearings / Linear tracks
            cv2.line(img, (0, 80 + i*50), (512, 420 - i*30), t["soil"], 45)
            cv2.line(mask, (0, 80 + i*50), (512, 420 - i*30), 0, 45)
            
            # Class 2: Green Structures / Reflections
            cv2.rectangle(img, (150 + i*30, 150), (240 + i*30, 240), t["manmade"], -1)
            cv2.rectangle(mask, (150 + i*30, 150), (240 + i*30, 240), 128, -1)
            
            # Add heavy random texture noise to simulate organic surface variations
            noise = np.random.randint(-25, 25, img.shape, dtype=np.int16)
            img = cv2.add(img, noise, dtype=cv2.CV_8U)
            
            file_name = f"satellite_tile_{file_idx}.png"
            cv2.imwrite(os.path.join("training_images", file_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join("training_masks", file_name), mask)
            file_idx += 1
            
    print(f"✅ Generated {file_idx} highly diverse training pairs in data folders.")

if __name__ == "__main__":
    generate_robust_dataset()
