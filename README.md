# 🔥 Wildfire Image Classification Using CNNs & ResNet50

This project tackles wildfire detection from satellite images using two deep learning models: a custom Convolutional Neural Network (CNN) and a pre-trained **ResNet50** architecture. The goal is to support early wildfire detection systems by accurately classifying images as `wildfire` or `nowildfire`.

---
🔗 [Wildfire Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset/)


## 📌 Project Highlights

- Built and trained a **Custom CNN** from scratch.
- Fine-tuned **ResNet50** using transfer learning for binary classification.
- Visualized performance through training curves, confusion matrices, and ROC-AUC plots.
- Achieved **~98.5% test accuracy** with ResNet50 and **~96% with Custom CNN**.
- Analyzed misclassifications to understand model behavior and performance limits.

---

## 🧠 Model Architectures

### ✅ Custom CNN
- Multiple Conv2D layers with ReLU & MaxPooling
- Dropout for regularization
- Dense layer with sigmoid activation for binary output

### ✅ ResNet50
- Pre-trained on ImageNet
- Feature extraction with `include_top=False`
- GlobalAveragePooling + Dense layers added on top
- Fine-tuned only the classification head

---

## 📊 Evaluation Metrics

| Model       | Accuracy | Precision | Recall | F1-Score | AUC  |
|-------------|----------|-----------|--------|----------|------|
| Custom CNN  | ~96.3%   | 0.96      | 0.96   | 0.96     | 0.99 |
| ResNet50    | ~98.5%   | 0.98+     | 0.98+  | 0.98+    | 1.00 |

- 📈 **ResNet50 shows superior performance** across all metrics with fewer false positives and negatives.
- 📉 **Custom CNN performs well** but misclassifies more samples, indicating room for improvement.

---

## 📁 Dataset

- Binary classification: `wildfire` vs `nowildfire`
- Satellite images resized to 224x224
- Loaded via `ImageDataGenerator` with `flow_from_directory()`

---

## 🖼️ Visualizations

- 📌 **Training & Validation Accuracy/Loss**
- 📌 **Confusion Matrix for both models**
- 📌 **ROC Curve with AUC comparison**

---

## 🛠️ Tech Stack

- Python 🐍
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## 📦 Installation

```bash
git clone https://github.com/your-username/wildfire-classifier.git
cd wildfire-classifier
pip install -r requirements.txt
