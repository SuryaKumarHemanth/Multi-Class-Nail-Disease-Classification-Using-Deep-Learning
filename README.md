# Multi-Class Nail Disease Classification Using Deep Learning

A deep learning system that classifies nail images into four disease categories, with a Flask web app for real-time prediction through image upload.

## Overview

Nail changes can be early signs of underlying health conditions. This project trains and compares three deep learning approaches to classify nail images into four categories, and deploys the best-performing model as a web application where a user can upload an image and receive a prediction.

> **Disclaimer:** This is an academic project for learning purposes. It is not a medical device and must not be used for diagnosis. Consult a qualified doctor for any health concern.

## Key Results

| Item | Detail |
|---|---|
| Task | Multi-class image classification (4 categories) |
| Dataset size | 2,844+ medical images |
| Best accuracy | **90.20%** (transfer learning) |
| Models compared | CNN, Hybrid Capsule CNN, MobileNetV2 |
| Deployment | Flask web app (image upload and prediction) |

## Tech Stack

- **Language:** Python
- **Deep learning:** TensorFlow, Keras
- **Image processing:** OpenCV, NumPy
- **Evaluation and plots:** Scikit-learn, Matplotlib
- **Web app:** Flask

## Project Workflow

1. **Data collection and preprocessing:** curated 2,844+ images, resized and normalised them for training.
2. **Data augmentation:** applied augmentation to improve generalisation and reduce overfitting.
3. **Model training:** built and trained three models:
   - A baseline CNN
   - A Hybrid Capsule CNN
   - A MobileNetV2 model using transfer learning
4. **Evaluation:** compared models using accuracy and other metrics from Scikit-learn.
5. **Deployment:** served the trained model through a Flask app with an image upload interface.

## Disease Categories

The model classifies images into four categories:

1.Healthy – Normal, healthy nails.
2.Acral Lentiginous Melanoma – A rare type of melanoma that can affect the nail unit.
3.Clubbing – Abnormal enlargement and curvature of the fingertips and nails.
4.Pitting – Small dents or depressions on the nail surface.
## Project Structure

```
.
├── [notebook_name].ipynb      # Training and evaluation
├── app.py                     # Flask application
├── model/                     # Saved trained model
├── templates/                 # HTML templates
├── static/                    # CSS and uploaded images
├── requirements.txt           # Dependencies
└── README.md
```



