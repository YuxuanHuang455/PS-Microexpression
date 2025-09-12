# PS1-Microexpression
# Facial Micro-Expression Dataset

This dataset, named **data**, contains material for the Facial Micro-Expression Recognition project. It is divided into two subsets: **train** (for model training) and **test** (for evaluation). The dataset comprises images and annotations of subtle facial expressions corresponding to six emotional states: *Anger, Disgust, Fear, Happy, Neutral,* and *Surprised*. Each category is stored in a dedicated folder, and annotations provide emotion labels for supervised learning tasks. The images are suitable for computer vision pipelines requiring high temporal and spatial resolution. The dataset enables the development and benchmarking of machine learning models for automatic micro-expression recognition.  

**Sources:**  
- [Ziya07 Facial Micro-Expression Recognition](https://www.kaggle.com/datasets/ziya07/facial-micro-expression-recognition)  
- [Kmirfan Micro-Expressions](https://www.kaggle.com/datasets/kmirfan/micro-expressions/data)  
- [Muhammad Saad Khan Kori Microexpression](https://www.kaggle.com/datasets/muhammadsaadkhankori/microexpression)  

---

## Data Dictionary

| Component   | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `train/`    | Training set organized into subfolders by emotion class.                    |
| `test/`     | Testing set organized into subfolders by emotion class.                     |
| `Anger/`    | Images of micro-expressions labeled as anger.                               |
| `Disgust/`  | Images of micro-expressions labeled as disgust.                             |
| `Fear/`     | Images of micro-expressions labeled as fear.                                |
| `Happy/`    | Images of micro-expressions labeled as happiness.                           |
| `Neutral/`  | Images of micro-expressions labeled as neutral.                             |
| `Surprised/`| Images of micro-expressions labeled as surprise.                            |
| `annotations.csv` | File linking image names with emotion labels (if included).           |

---

## Replicability and Reuse

This dataset builds upon openly available Kaggle resources and is structured for ease of replication in machine learning workflows. Its clear directory organization ensures reproducibility for emotion recognition tasks, while alignment with standardized emotion categories supports cross-study comparability. Researchers can extend the dataset by integrating additional samples or applying preprocessing methods such as temporal segmentation or data augmentation. Reuse potential is high in fields like affective computing, human–computer interaction, and psychological research, provided that ethical guidelines around emotion recognition and human data are respected. 

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

```mermaid
flowchart TD
    A[Research Question] --> B1[Emotion Intensity Prediction]
    A --> B2[Minority Class Prediction]

    B1 --> C1[Data: Kaggle micro-expression datasets<br/>Soft labels 0-1]
    B1 --> D1[Algorithms: CNN + LSTM<br/>Regression loss (MSE/MAE)]
    B1 --> E1[Compute: Colab GPU for training]

    B2 --> C2[Data: Imbalanced micro-expression samples]
    B2 --> D2[Algorithms: CNN + LSTM + imbalance handling<br/>SMOTE / Focal Loss / Class weights]
    B2 --> E2[Compute: Colab GPU + scikit-learn baselines]

    A --> F[Integration of GenAI Tools]
    F --> G1[ChatGPT for code prototyping]
    F --> G2[STORM for literature mapping]
    F --> G3[Hugging Face pretrained models]

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
