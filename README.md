# Predicting Emotion Intensity and Addressing Class Imbalance in Micro-Expression Recognition

## Abstract
This project investigates two predictive challenges in micro-expression recognition using open-source Kaggle datasets: (1) extending traditional categorical classification to continuous emotion intensity prediction, and (2) improving recognition of minority emotion classes (e.g., fear, disgust) under imbalanced data conditions.  
By comparing regression- and classification-based approaches, and by applying imbalance-handling methods such as SMOTE, class weights, and Focal Loss, the project aims to improve both the granularity and the fairness of emotion recognition models.  
The study integrates the AI Triad (data, algorithms, computing power) and reflects critically on the role of GenAI tools (ChatGPT, STORM, Hugging Face) in accelerating reproducible, ethically responsible research.  

---

## System Configuration
- **Local setup**:  
  - Python 3.10+  
  - Jupyter Notebook  
  - PyTorch / TensorFlow (for deep learning experiments)  
  - scikit-learn, imbalanced-learn (for baseline models and imbalance handling)  
- **Cloud setup**:  
  - Google Colab (GPU runtime for model training)  
  - Google Drive (dataset hosting and integration)  
  - Hugging Face Hub (pretrained vision models for transfer learning)  

---

## Research Framing & AI Triad Connections
flowchart
    A[Research Question] --> B1[Emotion Intensity Prediction]
    A --> B2[Minority Class Prediction]

    B1 --> C1[Data: Kaggle datasets - soft labels 0-1]
    B1 --> D1[Algorithms: CNN + LSTM; Regression loss (MSE/MAE)]
    B1 --> E1[Compute: Colab GPU for training]

    B2 --> C2[Data: Imbalanced micro-expression samples]
    B2 --> D2[Algorithms: CNN + LSTM with imbalance handling (SMOTE, Focal Loss, Class weights)]
    B2 --> E2[Compute: Colab GPU + scikit-learn baselines]

    A --> F[Integration of GenAI Tools]
    F --> G1[ChatGPT: code prototyping]
    F --> G2[STORM: literature mapping]
    F --> G3[Hugging Face: pretrained models]

## FAIR & CARE Principles
FAIR:
All datasets used are open-access and properly cited.
Code, preprocessing scripts, and notebooks are shared in this repository with clear documentation.
Outputs are stored in interoperable formats (CSV, JSON, PNG).

CARE:
Collective benefit: Research aims to support education and healthcare applications.
Authority to control: Dataset licensing and participant consent are acknowledged.
Responsibility: Results emphasize transparency and report limitations, avoiding misuse in surveillance contexts.
Ethics: Bias and fairness are critically evaluated, particularly regarding minority emotion classes.
