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

1. [Class 1 name]
2. [Class 2 name]
3. [Class 3 name]
4. [Class 4 name]

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

Edit this to match the files actually in your repository.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SuryaKumarHemanth/[repository-name].git
cd [repository-name]
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the web app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser, upload a nail image and view the predicted category.

## Screenshots

Add screenshots here, for example:

- Upload page
- Prediction result
- Accuracy / confusion matrix plot

```markdown
![Upload page](screenshots/upload.png)
![Prediction result](screenshots/result.png)
```

## What I Learned

- Building and comparing CNN, Hybrid Capsule CNN and transfer learning models
- Handling a small medical image dataset with augmentation
- Deploying a trained Keras model as a Flask web application

## Future Improvements

- Train on a larger and more varied dataset
- Add model explainability (for example Grad-CAM) to show which part of the image drove the prediction
- Containerise the app with Docker for easier deployment

## Author

**Surya Hemanth Kumar K**
B.Tech Computer Science and Engineering, SRM Institute of Science and Technology
[LinkedIn](https://www.linkedin.com/in/surya-hemanth-kumar-798722375/) | [GitHub](https://github.com/SuryaKumarHemanth)
