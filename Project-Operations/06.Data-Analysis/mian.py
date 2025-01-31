# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data from both sheets
df_employee = pd.read_excel('Data.xlsx', sheet_name='Employee')
df_performance = pd.read_excel('Data.xlsx', sheet_name='PerformanceRating')

# Merge the two dataframes on EmployeeID
df = pd.merge(df_employee, df_performance, on='EmployeeID', how='inner')

print(df.head())
# Check for missing values in the dataset
print(df.isnull().sum())
# Display basic information about the dataset (column types, memory usage, etc.)
print(df.info())

# Preprocessing categorical columns for analysis
# Convert 'OverTime', 'Attrition', and 'BusinessTravel' to numeric
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
label_enc = LabelEncoder()
df['BusinessTravel'] = label_enc.fit_transform(df['BusinessTravel'])

