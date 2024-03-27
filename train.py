import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
images_dir = '/Users/dev_lap04/Downloads/ecoDetect-thesis/scaled-images'  # Use the resized images for training
labels_dir = '/Users/dev_lap04/Desktop/data/labels/train'
model_path = '/Users/dev_lap04/Downloads/ecoDetect-thesis/models/eco_detect_model.h5'
num_classes = 6  # Update based on your number of classes

# Prepare your data arrays
image_data = []
label_data = []

# Read images and labels
for filename in os.listdir(images_dir):
    if filename.endswith((".jpg", ".jpeg")):
        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image_data.append(image)

        label_file = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                label_line = file.readline().strip()
                label_parts = label_line.split()
                if label_parts:
                    label = int(label_parts[0])
                    label_data.append(label)

image_data = np.array(image_data) / 255.0
label_data = to_categorical(label_data, num_classes=num_classes)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    image_data, label_data, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                    epochs=200, 
                    validation_data=(X_val, y_val))

# Save the trained model
model.save(model_path)
print(f"Model saved at {model_path}")

# Calculate and print confusion matrix
y_pred = model.predict(X_val)
y_true = np.argmax(y_val, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)
