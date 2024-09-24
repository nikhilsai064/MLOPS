from azureml.core import Workspace, Dataset, Datastore, Run
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import argparse
from azureml.core.authentication import InteractiveLoginAuthentication

# Auth
interactive_auth = InteractiveLoginAuthentication(tenant_id='51bdbbb4-12ab-426e-b041-8f1bb3405d8f', force=True)
ws = Workspace(subscription_id='1f0c4c69-865a-4779-bd83-d2a98937f6b0', resource_group='MLOPS', workspace_name='ML-Resource-Group', auth=interactive_auth)

# Data Import
datastore = Datastore.get(ws, 'workspaceblobstore')

# Argparser
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, help="Relative path of the file in the datastore")
args = parser.parse_args()

# Run Context
run = Run.get_context()

# Load data from the datastore
df = Dataset.Tabular.from_delimited_files(path=[(datastore, args.input_data)]).to_pandas_dataframe()
print(f"Data shape before preprocessing: {df.shape}")

# Drop duplicates
df = df.drop_duplicates()

# Encoding categorical variables
label_encoder = LabelEncoder()
df['Month'] = label_encoder.fit_transform(df['Month'])
df['VisitorType'] = label_encoder.fit_transform(df['VisitorType'])
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)

# Split into features and labels
X = df.drop(columns=['Revenue'])  # Features
y = df['Revenue']  # Target label

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and labels back into a single DataFrame for storage
train_dataset = pd.concat([X_train, y_train], axis=1)
test_dataset = pd.concat([X_test, y_test], axis=1)

# Save the processed train and test datasets locally as CSV files
train_dataset.to_csv('train_data.csv', index=False)
test_dataset.to_csv('test_data.csv', index=False)

run.complete()
print("Data preprocessing and saving completed!")
