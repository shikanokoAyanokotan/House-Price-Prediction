import pandas as pd
from scipy import stats
import torch
from linearRegression import train_linear_regression
from dataVisualization import DataVisualization


# Load the data
url = ("https://drive.google.com/uc?export=download&id=14aM0y_Q8lHAnsixprClRA-0an9ZmCwzu")
df = pd.read_csv(url)


# Visualize the data
DataVisualization(df)


# Handle missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df = df.drop(missing_data[missing_data['Total'] > 1].index, axis=1, errors='ignore')
df = df.drop(df.loc[df['Electrical'].isnull()].index, errors='ignore')


# Handle Outliers
z_scores = stats.zscore(df['GrLivArea'])
outliers = df[(z_scores > 3) | (z_scores < -3)]
outliers = outliers.sort_values(by='GrLivArea', ascending=False)
df = df.drop(df[(df['Id'] == 1299) | (df['Id'] == 524)].index, errors='ignore')
df = df.drop('Id', axis=1)


# Dimensionality Reduction with PCA
X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']

# Convert all object-type features into numerical format
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
object_columns = X.select_dtypes(include=['object']).columns
for column in object_columns:
    X[column] = label_encoder.fit_transform(X[column])

# Scale the data
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X_scaled = standard_scaler.fit_transform(X)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)


# Data splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


model_choice = input("Enter 0 (Linear Regression) or 1 (Decision Tree): ")
if model_choice == '0':
    train_linear_regression(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
else:
    print("Invalid input. Please enter 0 or 1.")
