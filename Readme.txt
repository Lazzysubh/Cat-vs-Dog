# Convolutional Neural Network for Image Classification

This repository contains code for a Convolutional Neural Network (CNN) designed for image classification tasks. The codebase includes modules for defining the CNN architecture, loading and preprocessing image data, training the model, and evaluating its performance.

## Code Structure
- `model.py`: Defines the CNN architecture using PyTorch.
- `data/load.py`: Contains functions for loading and transforming image data.
- `train.py`: Implements the training loop for the CNN model.

## Usage
To train the CNN model, run the `train.py` script with the required arguments:
- `logdir`: Directory to store training logs and model checkpoints.
- `-b`, `--batch_size`: Batch size for training.
- `--no_normalization`: Flag to disable data normalization.
- `-n`, `--n_epochs`: Number of training epochs.
- `-o`, `--optimizer`: Optimizer for training (default is Adam).

Example usage:
python train.py --logdir logs/ --batch_size 64 --no_normalization -n 10 -o Adam

bash
python train.py logs/ -b 128 --no_normalization -n 10 -o 'optim.Adam(parameters)'


Ensure you have PyTorch and torchvision installed to run the code successfully.

For more details on the code structure and usage, refer to the specific files in the repository.

