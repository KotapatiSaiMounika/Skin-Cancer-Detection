# Skin Cancer Detection using CNN

## Overview
A deep learning project that classifies skin lesion images as benign or malignant using a Convolutional Neural Network built with TensorFlow/Keras.

## Dataset
- Source: Skin Cancer MNIST (Kaggle)
- 3,297 dermoscopy images total
- Training: 2,637 images (1,440 benign, 1,197 malignant)
- Testing: 660 images (360 benign, 300 malignant)

## Approach
- Images resized to 100x100 pixels, normalized to [0,1]
- Data augmentation (rotation, horizontal flip, zoom) to prevent overfitting on training data
- CNN architecture: 3 Conv2D + MaxPooling blocks (32→64→128 filters) followed by Dense layers with Dropout

## Results
- Test Accuracy: 83.3%
- Malignant Recall: 88% (catches 88% of actual cancer cases)
- Benign Precision: 89%

## Key Insight
The model prioritizes high recall on malignant cases — minimizing missed cancer diagnoses (false negatives), which is the most critical error type in medical screening.

## Tech Stack
Python, TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn

## How to Run
1. Open `skin_cancer_detection.ipynb` in Google Colab
2. Download your own `kaggle.json` from Kaggle (Account Settings → API → Create New Token) and upload it when prompted in the notebook
3. The notebook will automatically download the dataset using the Kaggle API
4. Run all cells sequentially

## Future Improvements
- Transfer learning with pre-trained models (ResNet, VGG16)
- Larger dataset for better generalization
- Grad-CAM visualization to show which image regions influence predictions
