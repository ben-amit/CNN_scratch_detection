# 🧠 Wafer Die Scratch Detection using CNN

This project implements a **Convolutional Neural Network (CNN)** to identify **scratched dies** on a semiconductor wafer based on input maps showing good and bad dies.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/65/Siliconwafer_2.jpg" width="400" alt="Wafer Example">
</p>

---

## 🧾 Description

Scratches on wafers can lead to multiple defective dies, which must be automatically identified. The goal of this project is to predict whether each die on a wafer is on a scratch, using a CNN trained on labeled wafer data.

The model classifies each pixel (die) based on a binary label: **scratched** or **not scratched**.

---

## 📌 Features

- 🧠 Custom **U-Net-like CNN architecture**
- 🔁 **Dropout** regularization
- 📊 **F1-score** based early stopping
- 🧪 **Hyperparameter tuning** via grid search
- 📉 Training & validation loss tracking
- ✅ Binary segmentation output

---

## 🧱 Model Architecture

```python
Input (2 channels: good/bad dies)
 ↓
Conv2D → ReLU → Conv2D → ReLU → MaxPool
 ↓
Conv2D → ReLU → Conv2D → ReLU
 ↓
TransposeConv (upsample)
 ↓
Concatenate (skip connection)
 ↓
Conv2D → ReLU → Conv2D → ReLU
 ↓
Conv2D (1 output channel: scratch prediction)
