# MLOPS


# Project Overview:

**This project implements a Random Forest model for predicting customer revenue intentions using the Azure Machine Learning pipeline. The pipeline consists of data preprocessing, model training, and evaluation. The model is trained on the online_shoppers_intention.csv dataset and registered in the Azure ML model registry.**

# Files:

**data_prep.py: Script for data preprocessing, including encoding categorical features and splitting data into training and testing sets.**

**train_model.py: Script for training the Random Forest model and registering it in the Azure ML model registry.**

**evaluation.py: Script for evaluating the model on the test dataset, calculating accuracy, F1 score, precision, and recall.**

**pipeline.py: Pipeline orchestration script that connects the above steps into a single Azure ML pipeline.**

**Colab.ipynb: Google Colab notebook demonstrating the initial model implementation.**

**evaluation_metrics.md: Summary of evaluation metrics.**

# Requirements

Ensure the following dependencies are installed:

**pandas**
**scikit-learn**
**tensorflow**
**tensorflow_decision_forests**
**azureml-sdk**
**joblib**
**setuptools**


# How to Run

**Set Up Azure ML: Make sure your Azure Machine Learning workspace is configured and connected.**

**Upload Files: Place all four script files in the same directory (or modify the source_directory paths accordingly).**

# Submit the Pipeline:

**Run the pipeline.py script to submit the pipeline to Azure ML. This will trigger data preparation, model training, and evaluation.**

# Results: 

**The evaluation metrics will be logged in Azure ML, and the model will be available in the model registry.**

#Results

**The model achieved the following metrics on the test data:**

- **Accuracy:** 90.54%
- **F1 Score:** 89.88%
- **Precision:** 89.82%
- **Recall:** 90.54%

