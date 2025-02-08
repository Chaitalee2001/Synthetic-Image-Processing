import cv2
import numpy as np
from PIL import Image
import glob

# Preprocessing function
def preprocess_image(image_path, target_size=(224, 224), grayscale=False):
    image = Image.open(image_path)
    if grayscale:
        image = image.convert("L")  # Convert to grayscale
    else:
        image = image.convert("RGB")

    image = image.resize(target_size)  # Resize to 224x224
    image_array = np.array(image) / 255.0  # Normalize pixel values

    return image_array

# Process all generated images
image_paths = glob.glob("generated_image_*.png")
preprocessed_images = [preprocess_image(img_path) for img_path in image_paths]

# Save or display preprocessed images for verification
for i, img in enumerate(preprocessed_images):
    cv2.imwrite(f"preprocessed_image_{i+1}.png", (img * 255).astype(np.uint8))  # Convert back for saving

print("Preprocessed images saved successfully!")
