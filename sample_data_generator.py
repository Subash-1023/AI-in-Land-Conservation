import cv2
import numpy as np
import os


def generate_mock_dataset():
    os.makedirs("training_images", exist_ok=True)
    os.makedirs("training_masks", exist_ok=True)

    print("🎨 Creating synthetic dataset with forest canopies and green structures...")

    for i in range(5):
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)

        # Class 1: Fill with Natural Forest Green
        img[:] = [34, 139, 34]
        mask[:] = 255

        # Class 0: Non-green dirt road / deforested patch (Soil)
        cv2.line(img, (0, 100 + i * 40), (512, 400 - i * 20), (139, 69, 19), 40)
        cv2.line(mask, (0, 100 + i * 40), (512, 400 - i * 20), 0, 40)

        # Class 2: Synthetic "Green House" / Artificial Structure
        cv2.rectangle(img, (200, 200), (280, 280), (124, 252, 0), -1)
        cv2.rectangle(mask, (200, 200), (280, 280), 128, -1)

        # Add random texture noise to simulate satellite grain
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = cv2.add(img, noise, dtype=cv2.CV_8U)

        file_name = f"satellite_tile_{i}.png"
        cv2.imwrite(os.path.join("training_images", file_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join("training_masks", file_name), mask)

    print("✅ Mock dataset generated successfully inside 'training_images' and 'training_masks' folders!")


if __name__ == "__main__":
    generate_mock_dataset()