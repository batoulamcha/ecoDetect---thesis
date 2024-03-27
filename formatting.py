import os
from PIL import Image

# Paths
original_data_dir = '/Users/dev_lap04/Desktop/data/images/train'
scaled_data_dir = '/Users/dev_lap04/Downloads/ecoDetect-thesis/scaled-images'

# Ensure the scaled images directory exists
if not os.path.exists(scaled_data_dir):
    os.makedirs(scaled_data_dir)

# Resize images to 128x128 pixels and save them
for image_name in os.listdir(original_data_dir):
    if image_name.lower().endswith((".jpg", ".jpeg")):  # Handle both ".jpg" and ".JPG" extensions
        original_image_path = os.path.join(original_data_dir, image_name)
        image = Image.open(original_image_path)
        image_resized = image.resize((128, 128))
        scaled_image_path = os.path.join(scaled_data_dir, image_name)
        image_resized.save(scaled_image_path)

print("Resizing and saving of images completed.")
