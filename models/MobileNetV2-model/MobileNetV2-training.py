import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Paths
dataset_path = '/Users/dev_lap04/Desktop/garbage_classification'
model_save_path = '/Users/dev_lap04/Desktop/ecoDetect-thesis/models/MobileNetV2-model/MobileNetV2-model.keras'

# Parameters
image_size = (224, 224)
batch_size = 32
epochs = 20

# Data preparation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load data without splitting
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=image_size + (3,))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Evaluate the model
model.load_weights(model_save_path)
evaluation = model.evaluate(validation_generator)
print(f'Loss: {evaluation[0]}, Accuracy: {evaluation[1]}')

# Use the same settings for the generator as during training but with shuffle turned off
validation_generator_for_cm = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False 
)

# Reset the generator to ensure the order is the same as when labels were extracted
validation_generator_for_cm.reset()
Y_pred = model.predict(validation_generator_for_cm)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator_for_cm.classes

# Now compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
accuracy = np.trace(cm) / np.sum(cm)
print(f'Accuracy from confusion matrix: {accuracy * 100:.2f}%')

accuracy = np.trace(cm) / np.sum(cm)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the confusion matrix
np.save('/Users/dev_lap04/Desktop/ecoDetect-thesis/models/MobileNetV2-model/confusion_matrix_MobileNetV2.npy', cm)

# Class labels and plotting
class_labels = list(validation_generator.class_indices.keys())
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, annot_kws={"size": 12}, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix, without normalization')
plt.show()
