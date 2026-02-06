# Plant Disease Classification using EfficientNet-B0

This project focuses on identifying 38 different classes of plant diseases using deep learning.

## Key Features
* **Architecture:** EfficientNet-B0 (Transfer Learning)
* **Dataset:** PlantVillage (54,303 images)
* **Accuracy:** 99.98% on Validation Set
* **Optimization:** Mixed Precision Training (FP16) for faster GPU performance.

## Tech Stack
* **Framework:** PyTorch
* **Model:** torchvision.models.efficientnet_b0
* **Optimizer:** Adam (LR: 0.0005)
* **Augmentation:** Random Rotation, Vertical/Horizontal Flips, ColorJitter.

## How to run
1. Train the model: `python trainn.py`
2. Test on a custom image: `python test_image.py --image path/to/image.jpg`
