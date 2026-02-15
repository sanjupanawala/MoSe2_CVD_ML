# Deep Learning for Synthesis Optimization of CVD-Grown MoSe2

This repository contains the code and links to the data for the above project. We provide a trained U-Net model and a high-throughput analysis pipeline for the semantic segmentation of optical microscopy images of MoSe2.

## Dataset and Pretrained Models

The complete, annotated dataset of optical images and masks, as well as our final trained model weights, are permanently archived on Zenodo.

*   **Dataset:**  https://doi.org/10.5281/zenodo.18646700
*   **Pretrained Models:**  https://doi.org/10.5281/zenodo.18646700



## Usage

### Training
To retrain the model on the provided dataset, update the paths in `code/train.py` 

### Prediction
To run a prediction on a new batch of images , update which model to use and the image paths in `code/predictions.py` 


