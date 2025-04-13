# Dry Eye Patch Detection using Mask R-CNN

## Project Overview
This repository contains a deep learning solution for automated detection of dry patches in eyes using **Mask R-CNN** for instance segmentation. The project applies advanced computer vision techniques to address a medically significant problem, demonstrating expertise in medical image analysis.

## Problem Statement
Dry eye syndrome affects millions worldwide. Accurate detection of dry patches is crucial for diagnosis and treatment planning. Manual detection is time-consuming and prone to errors. This project automates the detection process using state-of-the-art deep learning techniques.

## Models Implemented
- **Mask R-CNN**: A state-of-the-art instance segmentation model using ResNet-50 FPN backbone.
- The model was trained for 100 epochs with careful monitoring of validation metrics.

### Best Model
The best-performing model was saved at **Epoch 10**, achieving the highest F1-score during validation.

## Results
| Metric     | Value   |
|------------|---------|
| Precision  | 0.0136  |
| Recall     | 0.2000  |
| F1-Score   | 0.0254  |
| Accuracy   | 0.0129  |

![Sample Detection](image.png)

## Repository Structure
"""dry-eye-detection-maskrcnn/
├── dataset/ # Training and validation datasets
│ ├── train/ # Training dataset
│ │ ├── annotations/ # JSON annotation files for training
│ │ └── images/ # Images for training
│ ├── val/ # Validation dataset
│ │ ├── annotations/ # JSON annotation files for validation
│ │ └── images/ # Images for validation
│ └── visualizations/ # Visualized results (e.g., segmented images)
├── datset_full_raw/ # Full raw dataset (training and validation)
│ ├── train/
│ │ ├── annotaions/ # Augmented annotations for training
│ │ └── images/ # Augmented images for training
│ ├── val/
│ │ ├── annotations/ # Augmented annotations for validation
│ │ └── images/ # Augmented images for validation
├── output/ # Output directory for processed data
│ ├── train/
│ │ └── images/
│ ├── val/
│ │ └── images/
├── inference.html # HTML visualization of inference results
├── inference.ipynb # Jupyter notebook for inference
├── main_final_with_visualiztion.ipynb # Final notebook with visualizations
├── README.md # Project documentation"""


## Installation

### Clone the Repository


### Set Up Environment (Optional)
Install dependencies from requirements.txt:
pip install -r requirements.txt


## Dependencies

The following libraries are required to run the project:
- Python 3.6+
- TensorFlow 2.x or PyTorch 1.7+
- OpenCV (for image processing)
- NumPy, Matplotlib, Pillow (for data handling and visualization)
- scikit-learn (for evaluation metrics)

## Dataset

The dataset consists of ocular surface images with manually annotated dry patches. Images were collected from clinical settings and preprocessed to normalize lighting conditions and enhance visibility of dry regions.

- **Training Set**: XX images with annotated masks.
- **Validation Set**: XX images with annotated masks.
- **Test Set**: XX images with annotated masks.

## Training

To train the model from scratch, use the following command:
python src/train.py --dataset dataset/ --epochs 100 --batch-size 2 --learning-rate 0.001 --backbone resnet50_fpn


### Training Parameters:
- **Backbone**: ResNet-50-FPN.
- **Learning Rate**: Starts at `0.001`, reduced by `0.1` at epochs `40` and `80`.
- **Optimizer**: Adam optimizer.
- **Loss Functions**: Combined loss for classification, bounding box regression, and mask segmentation.
- **Augmentation**: Random horizontal flips, rotations, brightness/contrast adjustments.

## Evaluation

To evaluate the model performance on the test set:
python src/evaluate.py --dataset dataset/test --model models/mask_rcnn_epoch_10.pth --batch-size 2


The evaluation script will output metrics such as precision, recall, F1-score, and accuracy.

## Inference

To detect dry patches in new images:
python src/inference.py --image path/to/image.jpg --model models/mask_rcnn_epoch_10.pth --output path/to/output.jpg


This will generate a segmented image highlighting detected dry patches.

## Acknowledgments

This implementation is based on:
- The [Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN) framework.
- The paper *"Mask R-CNN"* by Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick.
  

