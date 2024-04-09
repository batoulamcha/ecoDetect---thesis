import os
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow informational messages and set SSL context for downloading datasets
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the path for the dataset and where the model will be saved
dataset_path = '/Users/dev_lap04/Desktop/ecoDetect-thesis/garbage_classification'
model_save_path = '/Users/dev_lap04/Downloads/ecoDetect-thesis/models/custom-model/custom-model.keras'

# Image and training parameters
image_size = (224, 224)
batch_size = 32
epochs = 50

# Data preparation with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of the data for validation
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Custom CNN Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=image_size + (3,)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for training process optimization
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Load the best model from the checkpoint
model.load_weights(model_save_path)

# Evaluate the model with the validation data
evaluation = model.evaluate(validation_generator)
print(f'Loss: {evaluation[0]}, Accuracy: {evaluation[1]}')

# Predictions for the validation set
validation_generator.reset()  # Ensuring the generator is starting from the beginning
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# True labels
y_true = validation_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(cm)

# Calculate the accuracy
accuracy = np.trace(cm) / np.sum(cm)

print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the confusion matrix
np.save('/Users/dev_lap04/Downloads/ecoDetect-thesis/models/custom-model/confusion_matrix_custom.npy', cm)  # Change the file name appropriately for each model

# Class labels
class_labels = list(validation_generator.class_indices.keys())  # Extracting class labels from the generator

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.2)  # For label size
sns.heatmap(cm, annot=True, annot_kws={"size": 12}, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix, without normalization')
plt.show()