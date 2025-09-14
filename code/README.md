This repository implements a workflow for Facial Micro-Expression Recognition with a focus on improving predictions for minority classes (e.g., fear, disgust). The methodology consists of the following steps:

part1: Basic EDA
Before feature extraction, basic exploratory data analysis (EDA) was performed:
Verified dataset structure and image dimensions.
Visualized sample images to ensure correct preprocessing.
Analyzed class distribution, which revealed strong imbalance (e.g., fear and disgust underrepresented compared to happy and neutral).
This motivated the use of PCA for dimensionality reduction and SMOTE for minority class augmentation in later stages.

Part2: Test analysis
1. Feature Extraction
Local Binary Patterns (LBP): Capture local texture features from facial images.
CNN Features: Extract high-level deep representations using a convolutional neural network.

2. Dimensionality Reduction
Apply Principal Component Analysis (PCA) (e.g., 128 components) to reduce redundancy and computational cost.

3. Classification
Models include Support Vector Machine (SVM) with RBF kernel and Logistic Regression as a baseline.
Evaluation metrics: Accuracy, Macro-F1, Weighted-F1.

4. Data Augmentation
To address class imbalance, apply SMOTE (Synthetic Minority Oversampling Technique) to oversample minority emotion categories.
Results are compared between models trained with and without SMOTE to analyze trade-offs between overall accuracy and minority class performance.

This pipeline demonstrates how classical ML techniques (LBP + SVM) and deep learning features (CNN) can be combined with data augmentation to improve fairness and inclusivity in micro-expression recognition.
