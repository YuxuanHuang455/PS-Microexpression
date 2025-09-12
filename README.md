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
