# Disease Classifier and Segmentation

A deep learning web app that detects Pneumonia from chest X-rays, then visually segments both the lungs and the infected regions — with model explainability via Grad-CAM.

## Overview

This project combines two computer vision models into one Flask web application for chest X-ray analysis:
1. A **VGG19-based classifier** that predicts whether an X-ray shows *Pneumonia* or is *Normal*.
2. A **U-Net segmentation model** that highlights the lung regions in the image.

On top of classification, it uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize *which parts of the X-ray* the model focused on to make its infection prediction — outlining the likely infected regions directly on the image, so the result isn't just a label but a visual explanation.

## Tech Stack

- **Backend:** Flask (Python)
- **Deep Learning:** TensorFlow / Keras
- **Models:** VGG19 (transfer learning, classification) + custom U-Net (segmentation)
- **Explainability:** Grad-CAM
- **Image Processing:** OpenCV, PIL, NumPy
- **Frontend:** HTML, CSS, Bootstrap, jQuery
- **Data Prep/Training:** Jupyter Notebook (preprocessing + augmentation), standalone training script for U-Net

## Features

- **Pneumonia classification** — upload a chest X-ray, get a Normal/Pneumonia prediction from a fine-tuned VGG19 model
- **Lung segmentation** — U-Net model outlines lung boundaries directly on the X-ray (blue contours)
- **Infection segmentation (Grad-CAM)** — highlights the specific regions the classifier attended to when detecting pneumonia (green contours), giving a visual "where is the disease" explanation rather than a black-box label
- **Three dedicated pages** — separate flows for Classification, Lung Segmentation, and Infection Segmentation
- **Data augmentation pipeline** — training notebook includes rotation, shift, shear, and flip augmentation to improve model generalization on a relatively small medical imaging dataset
- **Custom-trained U-Net** — trained from scratch (encoder-decoder with skip connections) on lung/infection masks

## Architecture

```
                     ┌─────────────────────────┐
   X-ray Upload ───► │      Flask app.py       │
                     └───────────┬─────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
     /predict_class       /predict_lung        /predict_infection
   (VGG19 classifier)     (U-Net model)      (VGG19 + Grad-CAM)
             │                   │                   │
       Normal/Pneumonia    Lung contour mask   Infection heatmap →
         label (JSON)      drawn on image      thresholded contours
                                                 drawn on image
```

**Model pipeline (training side):**
```
Chest X-Ray Dataset
      │
      ▼
preprocessing.ipynb — resize, normalize, augment (rotate/shift/shear/flip)
      │
      ├──► VGG19 (transfer learning) ──► Classification head (Normal/Pneumonia)
      │
      └──► train_unet.py ──► Custom U-Net (encoder-decoder) ──► Lung/infection masks
                                                                        │
                                                                        ▼
                                                          model_weights/*.h5
```

## Setup & Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/Shriya-2701/Disease-Classifier-and-Segmentation.git
   cd Disease-Classifier-and-Segmentation
   ```
2. Install dependencies:
   ```bash
   pip install flask tensorflow opencv-python pillow numpy scikit-learn
   ```
3. Place trained model weights in a `model_weights/` folder at the project root:
   ```
   model_weights/
   ├── vgg19_model_03.h5      # Classification model
   └── unet_pneumonia.h5      # Segmentation model
   ```
   (Train these yourself using `preprocessing.ipynb` for the classifier and `train_unet.py` for the U-Net, or plug in your own weights of the same architecture.)
4. Run the Flask app:
   ```bash
   cd "Flask Application"
   python app.py
   ```
5. Open `http://127.0.0.1:5000` in your browser, upload a chest X-ray, and try the Classification, Lung Segmentation, and Infection Segmentation pages.

## Folder Structure

```
Disease-Classifier-and-Segmentation-main/
├── preprocessing.ipynb          # Data loading, augmentation, VGG19 training/prep
├── train_unet.py                # U-Net training script for lung/infection masks
└── Flask Application/
    ├── app.py                   # Main Flask app — routes + inference endpoints
    ├── gradcam.py                # Grad-CAM heatmap generation logic
    ├── templates/                # HTML pages (home, classify, lung/infection segmentation)
    ├── static/                   # CSS/JS assets (Bootstrap-based UI)
    └── uploads/                  # Sample/uploaded X-ray images
```

## Key Learnings / Challenges

- Combining a **classification model** and a **segmentation model** in one pipeline to go beyond a simple label — showing both *what* the diagnosis is and *where* the evidence lies in the image.
- Implementing **Grad-CAM from scratch** with `tf.GradientTape` to visualize which convolutional features drove the classifier's decision, then converting the heatmap into thresholded contours for a clean visual overlay.
- Building a **U-Net encoder-decoder** with skip connections and training it on a limited medical imaging dataset within realistic compute constraints (small image size, few epochs).
- Debugging a class-label mismatch — the classifier initially had `Normal`/`Pneumonia` label indices reversed, which was traced and corrected in `get_className()`.
- Preprocessing pipeline differences between models: the classifier expects RGB input, while the U-Net expects grayscale — requiring separate preprocessing functions (`preprocess_rgb`, `preprocess_gray`) feeding the same Flask app.

## Disclaimer

This project is for educational/demonstration purposes only and is **not a certified diagnostic tool**. Predictions should not be used for real medical decision-making.
