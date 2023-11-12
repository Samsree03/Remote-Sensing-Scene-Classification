# Remote-Sensing-Scene-Classification

## Table of Contents:
1. ### Overview
2. ### Dataset
3. ### Methodology
4. ### Conclusion
5. ### Results

## Overview:
Remote Sensing Scene Classification (RSSC) is a crucial aspect of interpreting remote sensing images, allowing for the automated categorization of scenes such as forests, airports, and rivers. This project leverages state-of-the-art convolutional neural networks (CNNs) to classify scenes in the NWPU-RESISC45 dataset. The chosen pre-trained models include ResNet50, MobileNetV2, VGG16, and DenseNet121. The objective is to assess and compare the performance of these models in terms of accuracy and loss, ultimately identifying the most efficient model for remote sensing scene classification.

## Dataset:
The NWPU-RESISC45 dataset comprises 45 classes, each containing 700 RGB images. These images, sourced from Google Earth Imagery in 2017, are sized at 256x256 pixels. The dataset is split into training (90%) and testing (10%) sets to evaluate the models' performance.

## Methodology:
Four pre-trained CNN models, namely ResNet50, MobileNetV2, VGG16, and DenseNet121, are employed to train on the NWPU-RESISC45 dataset. The dataset is divided into training and testing sets with a ratio of 9:1. The program trains each model on the training data and evaluates its performance on the testing set. Accuracy and loss metrics are calculated for both training and testing images. The obtained results, including accuracy and loss for each neural network, are visualized through plots, available in the "Evaluation Graphs" folder. This allows for a comparative analysis to identify the most efficient model.

## Conclusion:
After rigorous testing and training, the ResNet50 model emerges as the most efficient among the four models, demonstrating higher accuracy and lower loss. This conclusion is drawn based on the evaluation of the NWPU-RESISC45 dataset. The ResNet50 model is selected for its superior performance in classifying remote sensing images.

## Results:
Detailed results, including accuracy and loss metrics, are available in the "Evaluation Graphs" folder. These graphs provide a visual representation of the performance of each model in the remote sensing scene classification task.
