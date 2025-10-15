# Predicting Emotion Intensity and Addressing Class Imbalance in Micro-Expression Recognition
## Authors and Roles
Yuxuan Huang: Designed research proposal; implemented data preprocessing, model training (SVM, Logistic Regression, AutoML), and visualization for comparative analysis; authored reflection on causal inference and optimization.
Prof. Zhang: Provided conceptual guidance, research framing, and feedback on proposal development, model interpretation, and methodological instructions.

## Abstract
This project investigates two predictive challenges in micro-expression recognition using open-source Kaggle datasets: (1) extending traditional categorical classification to continuous emotion intensity prediction, and (2) improving recognition of minority emotion classes (e.g., fear, disgust) under imbalanced data conditions.  
By comparing regression- and classification-based approaches, and by applying imbalance-handling methods such as SMOTE, class weights, and Focal Loss, the project aims to improve both the granularity and the fairness of emotion recognition models.  
The study integrates the AI Triad (data, algorithms, computing power) and reflects critically on the role of GenAI tools (ChatGPT, STORM, Hugging Face) in accelerating reproducible, ethically responsible research.  

---
## Navigation Instructions
1. Explanation Models
Location: /explanation/
Includes code for:
Topic modeling and clustering (BERTopic, LDA)
Sentiment and causal inference analysis
Network and feature importance visualizations
2. Prediction Models
Location: /prediction/
Contains:
SVM, Logistic Regression, and AutoML pipelines
Comparative evaluation (Accuracy, Macro-F1)
Performance plots showing pre- and post-training dynamics
3. Visualization
Location: Each subfolder's code
Includes generated figures for reflection and proposals:
Causal inference results
Optimization comparison figure
4. Datasets and Preprocessing
Location: /data/
Contains:
Original and cleaned datasets (train/test splits)
Data integration scripts combining 3 public datasets into training and 2 testing sets
5. System Configuration
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

## 📘 Notebooks
1. Prediction_of__Microexpression_basic_EDA_ipynb_.ipynb
Performs basic exploratory data analysis (EDA) on datasets.
Includes class distribution plots and imbalance checks.
Provides baseline evaluation metrics (Accuracy, Macro-F1, Weighted-F1).

2. Explanation_NLP_network_analysis.ipynb
Uses NLP (word cloud, keyword frequency) and network analysis (semantic co-occurrence networks) to map research themes.
Identifies central and peripheral terms in micro-expression literature.
Connects findings to broader research challenges (dataset bias, subtlety of low-frequency emotions).

## Results Summary
Baseline (ResNet18, no SMOTE): High accuracy but biased toward majority classes.
PCA + SMOTE: Lower accuracy, but better recall and Macro-F1 for minority classes (fear, disgust).
AutoML (FLAML): Best overall accuracy but less interpretable.
Logistic Regression: Competitive performance with highest interpretability.

## Acknowledgments and Revisions
This project benefited greatly from the constructive feedback of reviewers, whose comments guided substantial improvements in both methodology and presentation. Specifically:

Reviewer Feedback Integration
Expanded rationale for selecting ResNet18 compared with other CNN architectures (ResNet50, VGG, MobileNet), clarifying the trade-off between computational efficiency and feature richness.
Enhanced dataset transparency by explicitly listing the three datasets used, documenting Google Images sourcing protocols, and adding a summary table of dataset characteristics.
Improved terminology clarity with explicit definitions and citations for Macro-F1, Weighted-F1, PCA, SMOTE, and CNN features.
Strengthened ethical safeguards by specifying informed consent procedures, withdrawal rights, cultural adaptation of materials, and requirements for bias auditing in deployed systems.
Added a System Configuration section in the README for full reproducibility (Python version, Colab environment, dependencies).
Expanded figure captions to highlight logical relationships between technical methods, challenges, and ethical concerns.

Acknowledgments
I would like to thank Prof. Zhang and Jingting for their thoughtful and detailed reviews, which significantly improved the rigor, clarity, and ethical grounding of this work. Their feedback shaped both the updated manuscript and this repository’s documentation.

These revisions ensure that the project is methodologically transparent, ethically responsible, and aligned with both FAIR and CARE principles.

Also thanks for AIGC Tools: OpenAI GPT-5 and AutoML (FLAML) for iterative code generation and optimization testing.
And thanks for Open-source Communities: Hugging Face, Scikit-learn, and Matplotlib contributors for accessible libraries and documentation.

## Disclaimer
This repository supports the final research proposal submitted to STATS 201: Machine Learning for Social Science, instructed by Prof. Luyao Zhang at Duke Kunshan University in Autumn 2025

## Statement of Growth

Through this project, I transitioned from applying statistical models mechanically to understanding their explanatory power in social science contexts.
I learned:
To interpret model transparency and causal inference, rather than focusing solely on accuracy.
The balance between explainability and performance across algorithms (e.g., SVM vs. LogReg vs. AutoML).
To connect data-driven results with broader social interpretation, which is a shift from coding to conceptual reasoning.
This process strengthened both my technical confidence and my academic independence.

## Table of Contents
Dataset: https://drive.google.com/drive/folders/1G6J9fBjIkc0Fs30SHYVkeK1lKWp9fc8Z
EDA-data:https://github.com/YuxuanHuang455/PS-Microexpression/blob/main/code/Microexpression_basic_EDA.ipynb
Test-data: https://github.com/YuxuanHuang455/PS-Microexpression/blob/main/code/Test_LBP%2BCNN%2BSVM_Classification_copy_of_%E2%80%9CMicroexpression_basic_EDA_ipynb%E2%80%9D.ipynb
Prediction: https://github.com/YuxuanHuang455/PS-Microexpression/tree/main/Prediction
Explanation: https://github.com/YuxuanHuang455/PS-Microexpression/tree/main/Explanation
Causal Inference: https://github.com/YuxuanHuang455/PS-Microexpression/tree/main/Causal-Inference
Vidro: https://github.com/YuxuanHuang455/PS-Microexpression/tree/main/Video

## Embedded Media
Poster link: https://www.canva.com/design/DAGz2tkl0FA/4uT1plaiBaCdTY36e1qH6Q/edit?utm_content=DAGz2tkl0FA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 

Video: 
https://github.com/user-attachments/assets/7c0b1689-0816-4182-8f81-37bade976eda
https://github.com/user-attachments/assets/9afe8b78-16aa-415a-bcd1-e628a74a89b8



