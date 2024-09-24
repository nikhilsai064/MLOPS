from azureml.core import Workspace, Dataset, Datastore, Run
import pandas as pd
import tensorflow_decision_forests as tfdf
import argparse
from azureml.core.model import Model
from azureml.core.authentication import InteractiveLoginAuthentication
import joblib

# Auth
interactive_auth = InteractiveLoginAuthentication(tenant_id='51bdbbb4-12ab-426e-b041-8f1bb3405d8f', force=True)
ws = Workspace(subscription_id='1f0c4c69-865a-4779-bd83-d2a98937f6b0', resource_group='MLOPS', workspace_name='ML-Resource-Group', auth=interactive_auth)

# Data Import
datastore = Datastore.get(ws, 'workspaceblobstore')

# Argparser
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, help="Path to the training data in the datastore")
args = parser.parse_args()

# Run Context
run = Run.get_context()

# Load training data
train_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, args.train)]).to_pandas_dataframe()

# Prepare dataset for training
train_tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_dataset, label="Revenue")

# Train the model
model = tfdf.keras.RandomForestModel()
model.fit(train_tf_dataset)

# Save the model
model_name = "random_forest_model.pkl"
joblib.dump(model, model_name)

# Register the model in Azure ML
model = Model.register(
    workspace=ws,
    model_path=model_name,  # Path to the model
    model_name="random_forest_model"  # Name of the model in Azure ML
)

run.complete()
print(f"Model {model.name} registered successfully!")
