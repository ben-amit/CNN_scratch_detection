# Wafer Die Scratch Detection using CNN

This project implements a Convolutional Neural Network (CNN) to identify scratched dies on a semiconductor wafer.

---

## Overview

Scratches on wafers can result in defective dies which must be detected automatically. This project trains a CNN to predict for each die on a wafer, whether it's on a scratch or not.

Solving this problem I have implemented different techniques learned in my ML and Deep Learning courses, while also learning new things on my own, the main one of them being the score given on each wafer using the F1-metric.

---

## Key Features

- Custom CNN architecture
- Dropout regularization
- F1-score-based early stopping
- Hyperparameter tuning via grid search
- Training and validation loss tracking
- Binary segmentation output (per-pixel classification)

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
