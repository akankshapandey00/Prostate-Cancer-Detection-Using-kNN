# Prostate Cancer Detection Using kNN


**Author:** Akanksha Pandey  
**Date:** 2023-09-30

---

## Overview

This project utilizes the k-Nearest Neighbors (kNN) algorithm to detect prostate cancer from a dataset. The tasks involve data preparation, model training, evaluation, and comparison of different preprocessing techniques and train-test splits.

## Installation

Install the necessary R packages:
- `class`
- `gmodels`
- `caret`
- `tidyverse`

## Data Preparation

### Task 1: Loading Data

Load the dataset from the specified file path and check its structure to ensure it has 569 examples and 32 features.

### Task 2: Data Preprocessing

1. **Drop the ID column** to prevent overfitting.
2. **Recode the diagnosis result** to factors "Benign" and "Malignant".
3. **Normalize the data** using a custom normalization function.

### Creating Training and Test Sets

Split the normalized data into training and test sets, and create corresponding labels.

### Model Training and Evaluation

1. **Train the kNN model** with the training data and make predictions on the test data.
2. **Evaluate the model performance** using a cross-table to compare actual vs. predicted results.

### Improving Model Performance

1. **Scale the data** and split it into new training and test sets.
2. **Train the kNN model** with the new training data and evaluate its performance again.

### Task 3: Using kNN from the caret Package

1. **Load the dataset** directly from the specified file path.
2. **Partition the data** into training and testing sets using the `createDataPartition` function from the caret package.
3. **Preprocess the data** by centering and scaling using the `preProcess` function.
4. **Train the kNN model** using `trainControl` and `train` functions, specifying repeated cross-validation.
5. **Evaluate the model** by making predictions on the test set and using a confusion matrix to determine accuracy.

## Results

- **Algorithm 1**:
  - When k=10 and the train:test split is 65:35, the accuracy is 60%.
  - When k=9 and the train:test split is 80:20, the accuracy is 75%.
- **Algorithm 2**:
  - The accuracy is 75%.

## Conclusion

The kNN algorithm, especially when optimized and preprocessed, can be an effective method for detecting prostate cancer. The accuracy varies significantly with different train-test splits and preprocessing methods, highlighting the importance of data preparation and model tuning.
