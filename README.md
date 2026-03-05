# Pneumonia Detection from Chest X-Rays images using a Custom CNN

This project builds a deep learning pipeline from scratch to detect pneumonia in chest X-ray images. The model is a custom Convolutional Neural Network (CNN) trained on a labeled dataset of chest X-rays containing two classes — NORMAL and PNEUMONIA.
What this project covers:

* Exploratory data analysis and class distribution visualization
* Image preprocessing : resizing to 224×224px, normalization, and data augmentation (rotations, flips, zooms)
* Class imbalance handling using balanced class weights
* Custom CNN architecture : Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Sigmoid
* Model training with EarlyStopping and ReduceLROnPlateau callbacks

Full evaluation : Accuracy (82.3%), Precision (93.4%), Recall (76%), F1-Score (83.8%), and Confusion Matrix

Results:
The model correctly identifies the majority of pneumonia cases with a precision of 93.4%, while maintaining an overall accuracy of 82.3%. The main limitation is a Recall of 76%, meaning some pneumonia cases are missed — a known challenge in medical imaging with class imbalance.
