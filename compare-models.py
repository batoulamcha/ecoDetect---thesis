import numpy as np
import subprocess

# Load confusion matrices
cm_custom = np.load('/Users/dev_lap04/Desktop/ecoDetect-thesis/models/custom-model/confusion_matrix_custom.npy')
cm_mobilenet = np.load('/Users/dev_lap04/Desktop/ecoDetect-thesis/models/MobileNetV2-model/confusion_matrix_MobileNetV2.npy')
cm_vgg16 = np.load('/Users/dev_lap04/Desktop/ecoDetect-thesis/models/VGG16-model/confusion_matrix_VGG16.npy')

def calculate_accuracy(cm):
    return np.trace(cm) / np.sum(cm)

accuracy_custom = calculate_accuracy(cm_custom)
accuracy_mobilenet = calculate_accuracy(cm_mobilenet)
accuracy_vgg16 = calculate_accuracy(cm_vgg16)

# Print the accuracies
print(f'Accuracy of CustomModel: {accuracy_custom:.4f}')
print(f'Accuracy of MobileNetV2: {accuracy_mobilenet:.4f}')
print(f'Accuracy of VGG16: {accuracy_vgg16:.4f}')
# Compare and print the accuracies
accuracies = {
    'CustomModel': accuracy_custom,
    'MobileNetV2': accuracy_mobilenet,
    'VGG16': accuracy_vgg16
}

# Compare and print the best model
best_model = max(accuracies, key=accuracies.get)
print(f'The best model is {best_model} with an accuracy of {accuracies[best_model]:.4f}')

# Automatically run classifier.py with the best model
classifier_script = 'classifier.py'
model_name = best_model

subprocess.run(['python', classifier_script, model_name])
