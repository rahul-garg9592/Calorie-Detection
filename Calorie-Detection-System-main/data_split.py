import os
import random
import shutil

# Paths to your dataset
image_dir = 'new_images'
label_dir = 'new_labels'
train_image_dir = 'new_images/new_train'
val_image_dir = 'new_images/new_val'
train_label_dir = 'new_labels/new_train'
val_label_dir = 'new_labels/new_val'

# Make sure the train/val directories exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Split images into train and val
images = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
random.shuffle(images)

train_size = int(0.8 * len(images))  # 80-20 split
train_images = images[:train_size]
val_images = images[train_size:]

# Copy files to train/val folders
for image in train_images:
    shutil.copy(os.path.join(image_dir, image), train_image_dir)
    label_file = image.replace('.png', '.txt').replace('.jpg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), train_label_dir)

for image in val_images:
    shutil.copy(os.path.join(image_dir, image), val_image_dir)
    label_file = image.replace('.png', '.txt').replace('.jpg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), val_label_dir)
