
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
```bash
python3 task3.py
````
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
This task implements a Locality-Sensitive Hashing (LSH) based Approximate Nearest Neighbor Search system for image retrieval. The system is designed to create an LSH index and efficiently search for the t nearest neighbors in a large dataset.

## Task 4a

### Usage
To create the LSH index, run the following command in your terminal:

```bash
python3 task4a.py
```
You will be prompted to enter  parameters l(number of layers) & h(number of hashes per layer). The system will create a LSH index based on the aforementioned params and will persist it on the disk for future usages.

## Features
1. **Customizable LSH Parameters:**
Users can specify the number of layers (L) for the LSH index.
Users can set the number of hashes per layer (h) to control the granularity of the hash buckets.

2. **LSH Index Training:**
The system automatically trains the LSH index using the specified parameters.
The training is performed on the Caltech101 dataset, utilizing ResNet-50 features.

3. **Index Saving and Loading:**
The LSH index details, including layers, hashes, and the index itself, are saved to a file (lsh_index_details.pkl).
The saved index can be loaded for later use, reducing the need for repeated training.

4. **Output Visualization:**
Users can choose to visualize the hash tables for debugging purposes.
The system provides insights into the distribution of images across hash buckets.

## Task 4b

### Usage
To search for the t nearest neighbors using the LSH index created in Task 4a, run the following command :

```bash
python3 task4b.py
```
You will be prompted to enter either an image id or image path and the parameter t. The system will then retrieve and display the t nearest neighbors using the LSH index.
## Features
1. **Interactive User Input:**
Users are prompted to input an image id or image path for the query.
Users specify the parameter t to determine the number of nearest neighbors to retrieve.

2. **Query Execution:**
The system uses the trained LSH index to efficiently search for the t nearest neighbors.
Euclidean distances are computed to measure the similarity between the query image and the retrieved neighbors.

3. **Extended Query for Insufficient Results:**
If the number of unique images in the initial result set is less than t, an extended query is automatically performed.
The extended query broadens the search space to ensure a sufficient number of results are obtained.

4. **Result Visualization:**
The system generates visualizations of the t nearest neighbors, including the input image and their Euclidean distances.
Results are plotted and saved to an output directory for further analysis.

5. **CSV Output:**
Euclidean distances and image IDs of the t nearest neighbors are saved to a CSV file (task4.csv) for external use or analysis.

6. **User Feedback:**
The system provides information on the number of unique images considered during the process.
Overall statistics on the number of images considered during the process are also displayed.

## Task 5

### Description

The task described involves implementing two types of relevance feedback systems: an SVM (Support Vector Machine) based system and a probabilistic relevance feedback system. These systems are designed to enhance the image results by updating the order from user input feedback.

### Usage
```bash
python3 task5.py
```
## Features

The code for the task have the following features:
-Data Retrieval: The code reads image data from a CSV file using pandas.
-User Interaction: Allows the user to input tags for certain images, providing feedback on the relevance of those images.
-SVM-based Relevance Feedback System: Trains a Support Vector Machine (SVM) using user-provided tags for images. The SVM model is trained to predict the relevance of images based on the Euclidean Distance feature.
-Probabilistic Relevance Feedback System: Estimates the probability of relevance for each document using Logistic Regression. Combines the relevance probabilities, user tags, and Euclidean distance to calculate a score for each result. Ranks the results based on the calculated score.
-Result Ranking: Ranks the results based on the SVM model's predictions or the probabilistic model's relevance scores.
-Command-Line Interface (CLI) Usage: Provides a simple CLI for the user to choose between the SVM-based and probabilistic relevance feedback systems.

---
