
---

# Project Title

## CSE 515 Multimedia and Web Databases - Phase #3

This project continues the exploration of multimedia and web databases using the Caltec101 dataset. It involves tasks related to clustering, indexing, and classification/relevance feedback.


The code is designed to work with the Caltech-101 dataset but can be adapted for other image datasets.

## Prerequisites

Before using this code, ensure you have the following prerequisites installed:

- Python 3
- PyTorch
- torchvision
- NumPy
- SciPy
- Matplotlib
- PIL (Python Imaging Library)
- MongoDB (for storing and retrieving feature descriptors)

### Tasks Overview
1. **Task 0a:** Computing and printing the inherent dimensionality associated with the even-numbered Caltec101 images involves understanding the underlying structure and complexity of these images.
2. **Task 0b:** Understanding the inherent dimensionality associated with each unique label of the even-numbered Caltec101 images provides insights into label-specific characteristics.
3. **Task 1:** Latent semantics computation and label prediction for odd-numbered images involve leveraging latent representations to make accurate label predictions.
4. **Task 2:** Significant cluster computation and label prediction for odd-numbered images aim to identify meaningful clusters within the dataset for effective label prediction.
5. **Task 3:** Implementing classifiers (m-NN, decision tree, PPR-based) for label prediction using even-numbered images requires building robust models for accurate classification.
6. **Task 4a:** Implementing Locality Sensitive Hashing (LSH) and a similar image search algorithm aims at efficient and effective image retrieval based on a given query image.
7. **Task 4b:** Employing an image search algorithm using LSH index and a visual model facilitates retrieving similar images from a vast dataset based on visual similarity.
8. **Task 5:** Implementing SVM and probabilistic relevance feedback systems enable users to refine queries based on feedback, enhancing retrieval accuracy.


## Usage

## Task 1

### Description
Task 1 evaluates the precision of predicted labels of odd images in the Caltech dataset based on the similarity score with the even numbered images in the Caltech dataset.

### Usage
python3 task1.py

## Features
  - The code provides the following features:
  - Label specific latent semantic: the task computes latent semantics for Caltech101 images. The extracted semantics are specific to a label.
  - Odd-Numbered Image Prediction: Predicts likely labels for the odd images based on the distances/similarities computed under label-specific latent semantics.
  - Evaluation Metrics: Computes essential evaluation metrics such as per-label precision, recall, F1-score, and overall accuracy to assess prediction performance.

## Usage

## Task 2

### Description

### Usage

## Features

## Task 3

### Description
Task 3 encompasses the implementation of various classifiers for image classification using the Caltec101 dataset. This task involves the creation and utilization of m-NN (m-Nearest Neighbors), decision tree, and PPR (Personalized PageRank) classifiers to predict labels for odd-numbered images.

### Usage

python3 task3.py

## Features

This code supports the extraction and analysis of the following image feature descriptors:
  - Classifier Implementation: The program provides the implementation of multiple classifiers tailored for image classification tasks.
  - Prediction for Odd-Numbered Images: Predicts the most likely labels for odd-numbered Caltec101 images using the trained classifiers.
  - Feature Space Selection: Allows for the selection and utilization of specific feature spaces for image representation and subsequent classification.
  - Evaluation Metrics: Computes essential evaluation metrics such as per-label precision, recall, F1-score, and overall accuracy to assess classifier performance.
  - Command-Line Interface (CLI) Usage: The program features a user-friendly interface accessible through command-line arguments.

## Usage

## Task 4

### Description

### Usage

## Features

## Usage

## Task 5

### Description

### Usage

## Features

---
