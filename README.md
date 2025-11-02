# 🏭 Industrial Anomaly Detection with Compact Convolutional Transformer (CCT)

## 📌 Overview

This project implements a **Compact Convolutional Transformer (CCT)** for visual anomaly detection in industrial objects using the **MVTec AD** dataset.

The goal is to distinguish between normal and defective (anomalous) images across different object classes such as `wood`, `metal_nut`, etc.

---

## ✨ Key Features

- 🔍 **Transformer-based vision model**:

  - Convolutional tokenization (patch extraction)
  - Optional positional embeddings
  - Stochastic Depth for regularization

- ⚙️ **End-to-end deep learning pipeline**:
  - 📥 Image loading, resizing, and normalization
  - ♻️ Data balancing through augmentation
  - 🧠 Model training, prediction, and evaluation
  - 📊 Performance visualization (F1-score, AUC, accuracy, etc.)

---

## 📚 Project Structure

### 1. **Data Preparation**

- Load selected MVTec AD categories (e.g., `wood`, `metal_nut`)
- Resize all images to **224×224**
- Normalize pixel values to [0, 1]
- Perform an **85/15 train/test split**
- Balance classes using data augmentation

---

### 2. **Model Architecture**

A simplified version of the CCT model in Keras:

```python
def create_cct_model():
    inputs = layers.Input((224, 224, 3))

    # Convolutional tokenizer
    tokens = CCTTokenizer()(inputs)

    # Transformer blocks
    for _ in range(2):
        x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(tokens, tokens)
        x = StochasticDepth(0.4)(x)
        tokens = layers.Add()([x, tokens])

    # Classification head
    x = layers.GlobalAveragePooling1D()(tokens)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Saabeelaa/cct-anomaly-detection.git
cd cct-anomaly-detection

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset

Download the **MVTec AD dataset** from the official site:  
🔗 https://www.mvtec.com/company/research/datasets/mvtec-ad

Expected folder structure:

```
dataset/
├── wood/
│   ├── train/
│   └── test/
├── metal_nut/
│   ├── train/
│   └── test/
...
```

---

## 📈 Results

| Class       | AUROC     | Accuracy  | Precision | F1 Score  | Recall    | FPR      |
| ----------- | --------- | --------- | --------- | --------- | --------- | -------- |
| Wood        | 99.74     | 97.5      | 97.77     | 97.77     | 97.77     | 2.86     |
| Metal Nut   | 95.88     | 89.04     | 97.14     | 89.47     | 82.92     | 3.12     |
| Capsule     | 91.59     | 82.19     | 100.0     | 77.96     | 63.88     | 0.00     |
| Hazelnut    | 99.73     | 98.46     | 98.46     | 98.46     | 98.46     | 1.54     |
| Carpet      | 92.54     | 92.47     | 97.47     | 92.30     | 87.50     | 2.22     |
| Pill        | 96.73     | 89.77     | 94.44     | 88.31     | 82.92     | 4.26     |
| Grid        | 92.89     | 90.69     | 92.30     | 90.00     | 87.80     | 6.67     |
| Zipper      | 97.66     | 90.24     | 100.0     | 90.00     | 81.81     | 0.00     |
| Transistor  | 99.94     | 97.56     | 100.0     | 97.56     | 95.23     | 0.00     |
| Tile        | 99.42     | 91.13     | 86.66     | 91.76     | 97.50     | 15.38    |
| Leather     | 99.37     | 94.04     | 90.69     | 93.97     | 97.50     | 9.09     |
| Toothbrush  | 97.50     | 86.36     | 81.81     | 85.71     | 90.00     | 16.67    |
| Bottle      | 100.0     | 100.0     | 100.0     | 100.0     | 100.0     | 0.00     |
| Cable       | 91.35     | 90.58     | 94.59     | 89.74     | 85.36     | 4.55     |
| Screw       | 88.74     | 79.81     | 83.72     | 76.59     | 70.58     | 12.07    |
| **Average** | **96.33** | **91.19** | **94.45** | **90.81** | **88.87** | **5.18** |

> Note: Results may vary depending on hyperparameters, augmentations, and number of epochs.

---

## 📖 References

- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [CCT Paper – arXiv:2104.05704](https://arxiv.org/abs/2104.05704)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
