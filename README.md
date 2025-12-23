#  AgriTech: Intelligent Plant Disease Detection
### An End-to-End Computer Vision Pipeline for Agricultural Robotics

##  Project Overview
In modern precision agriculture, early detection of plant diseases is critical for yield protection. This project implements a **Convolutional Neural Network (CNN)** pipeline to classify 38 distinct plant diseases from leaf images. 

Unlike standard tutorials, this project focuses heavily on **Data Engineering** and **Automated Auditing** to ensure model reliability in real-world conditions.

##  Tech Stack
* **Core:** Python 3.10+
* **Computer Vision:** OpenCV, TensorFlow/Keras (MobileNetV2)
* **Data Analysis:** NumPy, Pandas, Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter

##  Phase 1: Data Auditing & Quality Control
Before modeling, I engineered a rigorous auditing pipeline to clean the **New Plant Diseases Dataset**:
1.  **Corruption Checks:** Automatically scanned 80,000+ files to remove corrupted headers and non-image files.
2.  **Blur Detection (Laplacian Variance):** Implemented an algorithm to calculate image sharpness.
    * *Result:* Found the dataset was high quality (Avg Score: >4000), but identified specific outliers (<10 score) that were removed to prevent noise.
3.  **Class Balance:** Visualized distribution to ensure no class imbalance required synthetic oversampling.

##  Phase 2: Model Architecture
I utilized **Transfer Learning** to achieve high accuracy with minimal computational cost.
* **Base Model:** **MobileNetV2** (Pre-trained on ImageNet).
    * *Reasoning:* Lightweight architecture suitable for deployment on mobile devices or robotic embedded systems (e.g., Raspberry Pi).
* **Custom Head:**
    * Global Average Pooling
    * Dropout (0.2) for regularization
    * Dense Output Layer (38 Classes, Softmax)
* **Training Strategy:**
    * **Optimizer:** Adam (LR=0.001)
    * **Callbacks:** Early Stopping (Patience=3) and Model Checkpointing.

##  Results
* **Validation Accuracy:** ~92% (achieved within 2 epochs).
* **Inference:** The system successfully processes raw image inputs, resizes them to 224x224, and outputs a diagnosis with a confidence score.

##  Future Step
* **Deployment:** Wrap the model in a Streamlit web app or FastAPI backend.
* **Edge Computing:** Convert the model to **TensorFlow Lite** for deployment on an agricultural robot.

---
*Author: [Your Name]*
