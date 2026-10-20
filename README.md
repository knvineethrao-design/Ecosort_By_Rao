# Ecosort_By_Rao
1. Project Overview

The Compact Recycling CNN is a lightweight deep learning project designed to classify waste images into six categories: Cardboard, Glass, Metal, Paper, Plastic, and Trash.The project uses a compact Convolutional Neural Network (CNN) with depthwise separable convolutions to reduce computational complexity while maintaining image classification capabilities.The trained model can be exported to ONNX format for efficient inference and deployment.

Key Features
*Lightweight CNN architecture.
*Six-class waste image classification.
*Image augmentation and normalization.
*Training with validation and early stopping.
*Model checkpoint saving.
*ONNX model export and verification.
*CPU-compatible inference using ONNX Runtime.
*Model parameter limit of fewer than 350,000 parameters.

2. Waste Categories

The model classifies images into the following categories:

Class Index  Waste Category

0              Cardboard

1                 Glass

2                 Metal

3                 Paper

4                Plastic

5                 Trash

The class order is explicitly defined in the training script to maintain consistent classification labels.

3. Project Architecture

The project follows this workflow:

Dataset → Image Preprocessing → CNN Training → Model Evaluation → ONNX Export → Inference Verification

CNN Architecture
The model consists of:

Initial Convolution Layer: Extracts basic image features.

Depthwise Separable Convolutions: Extract features efficiently while reducing computational cost.

Batch Normalization: Stabilizes training.

ReLU6 Activation: Introduces non-linearity.

Adaptive Average Pooling: Reduces spatial feature dimensions.

Fully Connected Layer: Produces six class scores.

The model accepts images with an input shape of 3 × 96 × 96 and produces six output scores.

4. Technologies and Libraries

Library          Purpose

PyTorch (torch) -Tensor operations, CNN execution, training, and model saving.

PyTorch NN (torch.nn)-Implements convolutional layers, activation functions, normalization, and classification.

PyTorch Optimizers (torch.optim)-Updates model parameters using AdamW and adjusts the learning rate.

Torchvision (torchvision)-Loads image datasets and applies preprocessing and augmentation.

NumPy (numpy)-Handles numerical arrays and validates inference outputs.

ONNX (onnx)-Loads and validates the exported model.

ONNX Runtime (onnxruntime)-Executes predictions using the ONNX model.

Argparse (argparse)-Accepts training, export, and verification parameters from the command line.

Pathlib (pathlib)-Manages dataset paths, model files, and output directories.

Random (random)-Supports reproducible random operations during training.

5. Project Structure

Compact-Recycling-CNN/
│
├── data/
│   ├── train/
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   │
│   └── validation/
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
│
├── model.py
├── train.py
├── export.py
├── verify_onnx.py
├── inspect_data.py
│
├── outputs/
│   ├── best_model.pth
│   └── waste_classifier.onnx
│
└── README.md

Note: The outputs/ directory and model files are generated during training and export. They may not exist before running the project.

6. Dataset Preparation

Dataset is organised into training and validation directories.
Each directory contains six subdirectories corresponding to the waste categories.

Supported image extensions include:

JPG / JPEG,PNG,BMP,WEBP,TIF / TIFF

The validation dataset contains images separate from the training dataset to provide a more meaningful evaluation of model performance.

7. Inspect the Dataset

Before training, checks whether the dataset directory contains images.

python inspect_data.py

This script reports image counts in the dataset directories and identifies ZIP archives found under data/.
Ensures that the image files are extracted into the expected class directories before training.

8. Train the CNN Model

Run the training script:

python train.py

The default configuration includes:

Parameter  Default Value

Number of epochs-25

Batch size-32

Learning rate-0.001

Validation split-20% when a validation folder is unavailable

Random seed-42

Early stopping patience-6 epochs

Input image size-96 × 96

Number of classes-6

During training, the script displays:

Training loss.
Training accuracy.
Validation loss.
Validation accuracy.
Current learning rate.
The model with the best validation accuracy is saved automatically.
Output:
outputs/best_model.pth

9. Export the Model to ONNX

After training, export the best checkpoint:

python export.py

The script loads the saved PyTorch model and exports it to ONNX format.

Output:

outputs/waste_classifier.onnx

The exported model uses a dynamic batch dimension and an input image size of 96 × 96.The export script also reports the model's parameter count and file size.
The configured limits are:
Fewer than 350,000 parameters.
ONNX file size of no more than 1.50 MB.
These limits are checked by the script; whether the exported model meets them depends on the resulting model.

10. Verify the ONNX Model

Run:

python verify_onnx.py

The verification script checks:
Whether the ONNX model loads successfully.
Whether the ONNX graph passes validation.
Whether inference executes through ONNX Runtime.
Whether the output has the expected shape (1, 6).
Whether the output contains only finite numerical values.
Whether predictions remain consistent for the tested input scales.
A successful verification indicates that the exported model can execute inference under the tested conditions.

Optional: Evaluate ONNX Accuracy

You can also evaluate classification accuracy on a dataset containing six class subdirectories:

python verify_onnx.py --data data/validation

The script reports the number of correctly classified images and the resulting accuracy.

Uses a separate validation dataset to evaluate performance on images not used for training.

11. Model Evaluation

Model performance can be assessed using:
Training Accuracy: Measures classification accuracy on training images.
Validation Accuracy: Measures classification accuracy on validation images.
Training Loss: Measures classification error on training data.
Validation Loss: Measures classification error on validation data.
Model Size: Measures the storage required by the exported model.
Parameter Count: Measures the number of trainable and non-trainable model parameters.
Actual accuracy and model size should be recorded from the training and export outputs rather than assumed in advance.

12. Implementation Files

File

Responsibility

model.py

Defines the compact CNN architecture and depthwise separable convolution blocks.

train.py

Prepares datasets, trains the CNN, evaluates performance, and saves the best checkpoint.

export.py

Converts the trained PyTorch model into ONNX format.

verify_onnx.py

Validates the ONNX graph and tests inference using ONNX Runtime.

inspect_data.py

Inspects the dataset directory and reports image counts.

13. Conclusion

The Compact Recycling CNN demonstrates how a lightweight convolutional neural network can be developed for six-class waste image classification.
By combining depthwise separable convolutions, image augmentation, model evaluation, and ONNX deployment, the project provides a foundation for building computationally efficient waste classification systems.

Technologies: Python · PyTorch · Torchvision · NumPy · Pillow · ONNX · ONNX Runtime

Project Goal: Develop a compact and deployable de
