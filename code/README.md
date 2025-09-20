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

5. System Configuration
This project was developed and tested in Google Colab. The following configuration ensures reproducibility of the experiments:
Python version: 3.10 (default Colab runtime)
Platform: Google Colab (GPU runtime optional, not required for baseline experiments)

6. Key Libraries
numpy 1.26.4 — numerical computation
pandas 2.2.2 — data manipulation and analysis
scikit-learn 1.5.2 — machine learning (PCA, SVM, Logistic Regression, evaluation metrics)
imblearn 0.12.3 — data balancing (SMOTE, Borderline-SMOTE)
matplotlib 3.9.2 — visualization
seaborn 0.13.2 — visualization (class distributions, confusion matrices)
opencv-python 4.10.0 — image preprocessing
torch 2.4.1 — deep learning framework (CNN feature extraction with ResNet18)
torchvision 0.19.1 — pretrained CNN models and transforms

7. Notes
Experiments were run in Colab’s standard environment, ensuring compatibility without requiring local installation.
GPU acceleration (CUDA) can optionally be enabled for faster CNN feature extraction, though CPU execution is sufficient for smaller datasets.
All required dependencies are available via pip install in Colab.
