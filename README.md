# EcoDetect: Waste Classification Project

## Overview

EcoDetect is a machine learning project aimed at classifying various types of waste materials. Utilizing deep learning models, the project categorizes waste into multiple classes to aid in efficient waste management and recycling processes.

## Features

- Utilizes pre-trained models like MobileNetV2 and VGG16 for high-accuracy waste classification.
- Custom convolutional neural network (CNN) model for specific waste categorization tasks.
- Real-time waste detection using a webcam and pre-trained models.
- Analysis of model performance through confusion matrices.

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
   git clone https://github.com/batoulamcha/ecoDetect---thesis
   cd ecodetect

2) Install the required dependencies:
   pip install -r requirements.txt

## Usage

The project contains various scripts for training models, real-time detection, and performance evaluation:

### Training Models

To train a model, navigate to the model's directory and run the training script. For example, to train the MobileNetV2 model:
python MobileNetV2-model/MobileNetV2-training.py

### Real-Time Classification

For real-time classification, execute the classifier.py script with the desired model name as an argument. For example, to use MobileNetV2 for classification:
python classifier.py MobileNetV2

To quit the detection window, press `q`.

### Evaluating and Comparing Models

Run compare-models.py to evaluate and compare the performance of all trained models. This script automatically selects the best model based on its performance and then initiates real-time classification with it:
python compare-models.py

## Directory Structure

- `MobileNetV2-model/`: Contains scripts and files related to MobileNetV2 training and usage.
- `custom-model/`: Scripts and files for the custom CNN model.
- `VGG16-model/`: VGG16 related training scripts and model files.

## Requirements

- Python 3.8 or higher
- TensorFlow 2.0 or higher
- Other dependencies are listed in `requirements.txt`.

## Acknowledgments

- Dataset provided by Kaggle.
- Models and libraries used: TensorFlow, Keras, OpenCV, scikit-learn.
