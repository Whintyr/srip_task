# YOLOv8 Object Detection Pipeline

## Overview
This repository contains the complete pipeline for object detection using YOLOv8, including dataset preparation, exploratory data analysis, model training, and performance evaluation. The best-trained model can be found at:

```
runs/detect/train/weights/best.pt
```

## Repository Structure

- `runs/detect/train/weights/best.pt` - Best trained model.
- `dataset_generation.ipynb` - Code for dataset creation with augmentations and train/val/test split.
- `EDA.ipynb` - Exploratory Data Analysis (EDA) notebook visualizing dataset labels.
- `AP_calculations.ipynb` - Implementation of Average Precision (AP) calculations with a demo on randomly generated images.
- `srip-yolo-training.ipynb` - YOLOv8 training script along with results and metric curves.
- `runs/detect/train/` - Contains training logs, results, and performance metrics.

---

## Dataset Generation

In the dataset generation phase, images were augmented using **rotation and flipping** in all possible ways. The dataset was then split into:

- **80%** - Training set
- **10%** - Validation set
- **10%** - Test set

Future plans involve adding more augmentations such as:
- Random darkening/whitening
- High/low contrast
- Hue shifting
- Haze effect

These enhancements will improve the model's robustness to varying weather conditions.

---

## Exploratory Data Analysis (EDA)

The EDA notebook (`EDA.ipynb`) contains visualizations that analyze the distribution of labels within the dataset, providing insights into class imbalances and data diversity.

---

## Model Training & Performance

A **YOLOv8** model was trained on the generated dataset using the `srip-yolo-training.ipynb` script. Training was performed on **Kaggle GPU**, and the model achieved an impressive **mAP50 of 0.98803**.

Key results and performance curves are available in:
```
runs/detect/train/
```

Due to limited compute resources, only rotation-based augmentations were applied. Future work includes implementing additional augmentations for enhanced robustness.

---

## Future Work

- Implementing additional augmentations (contrast shifts, hue variations, haze effects, etc.) to improve model generalization across different conditions.
- Optimizing training time by leveraging more efficient GPU resources.
- Fine-tuning the model further to achieve higher robustness in real-world scenarios.


