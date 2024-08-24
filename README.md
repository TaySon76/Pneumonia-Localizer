# Pneumonia-Localizer
This project is a deep learning-based tool designed to detect pneumonia in chest X-ray images and localize potential regions of the infection within the images. This project leverages a pre-trained VGG16 model, fine-tuned to classify chest X-ray images as infected or uninfected, and generates heatmaps to identify the regions of infection (anomalies).

Project Overview
This project aims to automate the identification of pneumonia in X-ray images using a convolutional neural network (CNN). The pipeline includes:
  - Preprocessing the dataset.
  - Fine-tuning a pre-trained VGG16 model for binary classification (Pneumonia vs. Normal).
  - Generating heatmaps to localize anomalies within the X-ray images using the trained model.
  - Visualizing the performance of the model using confusion matrices and other evaluation metrics.
    
Project Structure:
Pneumonia-Localizer/
├── data/
│   ├── infected_images/          # Directory containing X-ray images with pneumonia
│   ├── uninfected_images/        # Directory containing normal X-ray images
├── models/
│   ├── vgg16_pneumonia.h5        # Saved model after training
├── notebooks/
│   ├── Pneumonia_Localizer.ipynb # Jupyter notebook for exploration and experimentation
├── results/
│   ├── confusion_matrix.png      # Confusion matrix visualization
│   ├── heatmaps/                 # Directory containing generated heatmaps
└── Pneumonia_Localizer.py        # Main Python script for model training and evaluation
└── README.md                     # Project README file
Prerequisites


This project requires the following dependencies:
1. Python 3.7+
2. TensorFlow 2.0+
3. Keras
4. NumPy
5. Matplotlib
6. OpenCV
7. SciPy
8. Seaborn
9. Scikit-Image
10. Scikit-Learn
11. Pillow

You can install these dependencies using pip

Dataset
infected_images/: Contains X-rays diagnosed with pneumonia.
uninfected_images/: Contains normal X-rays.
Ensure that the image files are in .png format.

Model Architecture
This project uses a pre-trained VGG16 model from Keras, fine-tuned for binary classification. 
The model architecture is as follows:
  - VGG16 Backbone: The base of the model, pre-trained on ImageNet, is used to extract image features.
  - Global Average Pooling: Reduces the spatial dimensions of the feature maps by averaging the values.
  - Dense Layer: A fully connected layer with a softmax activation to output the final classification (Pneumonia or Normal).


Training
The training process involves the following steps:
  - Data Preprocessing: Images are loaded, resized to 224x224 pixels, and normalized.
  - Model Training: The model is compiled with the SGD optimizer and trained for 30 epochs.
  - Evaluation: The trained model is evaluated on the test set, and metrics such as accuracy, loss, and confusion matrices are generated.
