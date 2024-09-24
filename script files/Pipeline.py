from azureml.core import Workspace, Experiment
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
import os

# Initialize workspace
ws = Workspace.from_config()

# Define compute target
compute_target = ws.compute_targets['joshiyeluri71']  # Replace with your actual compute target

# Run configuration and environment setup
aml_config = RunConfiguration()
aml_config.target = compute_target
aml_config.environment.python.user_managed_dependencies = False

# Define dependencies (including setuptools to resolve missing dependency)
aml_config.environment.python.conda_dependencies = CondaDependencies.create(
    conda_packages=['pandas', 'scikit-learn', 'tensorflow', 'setuptools'],  # Added 'setuptools'
    pip_packages=['azureml-sdk', 'azureml-dataset-runtime[fuse,pandas]', 'tensorflow_decision_forests', 'joblib']
)

# Check current working directory and file listings for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Files in the working directory: {os.listdir(os.getcwd())}")

# Adjust the source_directory if scripts are in the home directory
source_directory = "."

# Define pipeline steps
prep_step = PythonScriptStep(
    script_name="data_prep.py",  # This script is in the home directory
    source_directory=source_directory,
    compute_target=compute_target,
    arguments=['--input-data', "online_shoppers_intention.csv"],  # Adjust this as needed
    runconfig=aml_config,
    allow_reuse=False
)

train_step = PythonScriptStep(
    script_name="train_model.py",  # This script is in the home directory
    source_directory=source_directory,
    compute_target=compute_target,
    arguments=['--train', "train_data.csv"],  # Passing training data path
    runconfig=aml_config,
    allow_reuse=False
)

eval_step = PythonScriptStep(
    script_name="evaluation.py",  # This script is in the home directory
    source_directory=source_directory,  # Ensure the script path is correct
    compute_target=compute_target,
    arguments=['--test', "test_data.csv", '--model-name', "random_forest_model"],  # Pass model name as argument
    runconfig=aml_config,
    allow_reuse=False
)

# Build the pipeline
pipeline_steps = [prep_step, train_step, eval_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

# Submit the pipeline to an experiment
experiment = Experiment(ws, "random_forest_pipeline")
pipeline_run = experiment.submit(pipeline)

# Output the pipeline run details
print(f"Pipeline submitted successfully. Run ID: {pipeline_run.id}")
