import os
import cv2
import numpy as np
import tensorflow as tf

# Paths
scaled_dir = '/Users/dev_lap04/Desktop/scaled-images'
model_path = '/Users/dev_lap04/Downloads/ecoDetect-thesis/models/eco_detect_model.h5'
label_mapping = {0: 'metal', 1: 'organic', 2: 'paper', 3: 'plastic', 4: 'cardboard', 5: 'glass'}

# Ensure the scaled images directory exists
if not os.path.exists(scaled_dir):
    os.makedirs(scaled_dir)

# Load the trained model
print("Attempting to load model from:", model_path)
model = tf.keras.models.load_model(model_path)

# Open the camera for real-time detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 128x128 for the model
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0  # Normalize the frame
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Reshape for the model

    # Predict
    prediction = model.predict(input_frame)
    class_id = np.argmax(prediction)
    label = label_mapping[class_id]

    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
