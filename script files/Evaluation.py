from azureml.core import Workspace, Run
from azureml.core.model import Model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import argparse
import joblib
from azureml.core.authentication import InteractiveLoginAuthentication

# Auth
interactive_auth = InteractiveLoginAuthentication(tenant_id='51bdbbb4-12ab-426e-b041-8f1bb3405d8f', force=True)
ws = Workspace(subscription_id='1f0c4c69-865a-4779-bd83-d2a98937f6b0', resource_group='MLOPS', workspace_name='ML-Resource-Group', auth=interactive_auth)

# Argparser
parser = argparse.ArgumentParser()
parser.add_argument("--test", type=str, help="Path to the testing data in the datastore")
parser.add_argument("--model-name", type=str, help="Name of the model to load from the registry")
args = parser.parse_args()

# Run Context
run = Run.get_context()

# Load test data
datastore = Datastore.get(ws, 'workspaceblobstore')
test_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, args.test)]).to_pandas_dataframe()
X_test = test_dataset.drop(columns=['Revenue'])
y_test = test_dataset['Revenue']

# Load the model from Azure ML Model Registry
model = Model(ws, name=args.model_name)
model_path = model.download(exist_ok=True)
loaded_model = joblib.load(model_path)

# Predictions
predicted_classes = loaded_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, predicted_classes)
f1 = f1_score(y_test, predicted_classes, average='weighted')
precision = precision_score(y_test, predicted_classes, average='weighted')
recall = recall_score(y_test, predicted_classes, average='weighted')

# Log metrics
run.log("Accuracy", accuracy)
run.log("F1 Score", f1)
run.log("Precision", precision)
run.log("Recall", recall)

run.complete()
print("Model evaluation completed!")
