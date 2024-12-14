import os
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from tqdm import tqdm  # For progress bar

# Directory containing the images
folder_path = 'data/train_org/cardboard'
target_count = 2500  # Number of images needed in the folder

# Augmentation pipeline
aug = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # Randomly crop images
    iaa.LinearContrast((0.75, 1.5)),  # Adjust contrast
    iaa.Multiply((0.8, 1.2)),  # Randomly change brightness
    iaa.GaussianBlur(sigma=(0, 1.0))  # Blur images
])

def ensure_unique_filename(existing_files, base_name):
    """Generate a unique filename if the base_name already exists."""
    counter = 1
    new_name = base_name
    while new_name in existing_files:
        new_name = f"{base_name.rsplit('.', 1)[0]}_{counter}.{base_name.rsplit('.', 1)[1]}"
        counter += 1
    return new_name

def augment_images_in_folder(folder_path, target_count):
    images = []
    existing_files = set(os.listdir(folder_path))
    
    # Load all images from the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            images.append(np.array(img))

    if len(images) == 0:
        print(f"No images found in {folder_path}")
        return

    augmented_images = []
    while len(augmented_images) < target_count - len(images):
        augmented_batch = aug(images=images)
        augmented_images.extend(augmented_batch)
        if len(augmented_images) >= target_count - len(images):
            break

    # Save augmented images in the same folder
    for i, img_array in enumerate(augmented_images[:target_count - len(images)]):
        img = Image.fromarray(img_array)
        base_name = f'augmented_{i}.jpg'
        unique_name = ensure_unique_filename(existing_files, base_name)
        img.save(os.path.join(folder_path, unique_name))

# Run the augmentation
augment_images_in_folder(folder_path, target_count)
