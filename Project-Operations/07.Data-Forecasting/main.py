# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Stage 1: Data Loading and Preprocessing
def load_and_prepare_data(file_path):
    """Load and preprocess employee data from Excel file."""
    try:
        xls = pd.ExcelFile(file_path)
        employees = pd.read_excel(xls, sheet_name="Employee")
        performance = pd.read_excel(xls, sheet_name="PerformanceRating")
        df = pd.merge(employees, performance, on="EmployeeID", how="left").dropna()
        df = pd.get_dummies(df, drop_first=True)  # One-hot encode categorical variables
        print("📌 Available columns:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print("❌ File not found! Please check the file path.")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Stage 2: Model Training Functions
def train_classification_model(df, features, target, class_weight=None):
    """Train and evaluate a classification model."""
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"✅ {target} Model Accuracy: {accuracy:.2f}")
    print(f"📊 {target} Classification Report:\n", report)
    return model

def train_regression_model(df, features, target):
    """Train and evaluate a regression model with non-negative predictions."""
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative predictions
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"🔍 {target} Mean Absolute Error: {mae:.2f}")
    print(f"🔍 {target} R² Score: {r2:.2f}")
    return model

# Stage 3: Input Validation and Preprocessing
def get_new_employee_data(df, numerical_features, categorical_features, feature_ranges):
    """Collect, validate, and process input data for a new employee."""
    print("\n🔹 Enter new employee details:")
    employee_data = {}
    
    # Collect and validate numerical features
    for feature in numerical_features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                if feature in feature_ranges:
                    min_val, max_val = feature_ranges[feature]
                    if min_val <= value <= max_val:
                        employee_data[feature] = value
                        break
                    else:
                        print(f"❌ {feature} must be between {min_val} and {max_val}.")
                else:
                    employee_data[feature] = value
                    break
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
    
    # Collect categorical features
    for feature in categorical_features:
        value = input(f"{feature}: ")
        employee_data[feature] = value
    
    # Create DataFrame and apply one-hot encoding
    employee_df = pd.DataFrame([employee_data])
    employee_df_encoded = pd.get_dummies(employee_df, drop_first=True)
    
    # Align with training data columns
    employee_df_encoded = employee_df_encoded.reindex(columns=df.columns, fill_value=0)
    return employee_df_encoded

# Main Execution
file_path = "HrData.xlsx"  # Adjust path as needed
df = load_and_prepare_data(file_path)
if df is None:
    exit()

# Define realistic ranges for input validation
feature_ranges = {
    "Age": (18, 65),
    "Education": (1, 5),
    "YearsAtCompany": (0, 40),
    "YearsInMostRecentRole": (0, 40),
    "YearsWithCurrManager": (0, 40),
    "EnvironmentSatisfaction": (1, 5),
    "JobSatisfaction": (1, 5),
    "RelationshipSatisfaction": (1, 5),
    "WorkLifeBalance": (1, 5),
    "SelfRating": (1, 5),
    "ManagerRating": (1, 5),
    "Salary": (0, 1e7)  # Adjust max salary as needed
}

# Define input features
numerical_input_features = [
    "Age", "Education", "YearsAtCompany", "YearsInMostRecentRole",
    "YearsWithCurrManager", "EnvironmentSatisfaction", "JobSatisfaction",
    "RelationshipSatisfaction", "WorkLifeBalance", "SelfRating", "ManagerRating", "Salary"
]
categorical_input_features = ["OverTime", "Department", "JobRole"]

# Define common features (excluding targets)
common_features = [
    "Age", "Education", "YearsInMostRecentRole", "YearsWithCurrManager",
    "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
    "SelfRating", "ManagerRating"
]

# Train models with corrected feature sets
if "Attrition_Yes" in df.columns:
    turnover_features = common_features + ["Salary", "OverTime_Yes"]
    turnover_model = train_classification_model(df, turnover_features, "Attrition_Yes", class_weight="balanced")

if "YearsAtCompany" in df.columns:
    tenure_features = [f for f in common_features if f != "YearsAtCompany"]
    tenure_model = train_regression_model(df, tenure_features, "YearsAtCompany")

if "JobSatisfaction" in df.columns:
    satisfaction_features = [f for f in common_features if f != "JobSatisfaction"]
    satisfaction_model = train_classification_model(df, satisfaction_features, "JobSatisfaction")

promotion_column = [col for col in df.columns if "Promotion" in col]
if not promotion_column and "YearsSinceLastPromotion" in df.columns:
    df["Promotion"] = (df["YearsSinceLastPromotion"] == 0).astype(int)
    promotion_column = ["Promotion"]
if promotion_column:
    promotion_features = common_features + ["YearsAtCompany", "SelfRating", "ManagerRating"]
    promotion_model = train_classification_model(df, promotion_features, promotion_column[0], class_weight="balanced")

# Collect and validate new employee data
new_employee_data = get_new_employee_data(df, numerical_input_features, categorical_input_features, feature_ranges)

# Make predictions
if "Attrition_Yes" in df.columns:
    turnover_pred = turnover_model.predict(new_employee_data[turnover_features])[0]
    print("\n🚨 Attrition Likelihood:", "Yes" if turnover_pred == 1 else "No")

if "YearsAtCompany" in df.columns:
    tenure_pred = max(tenure_model.predict(new_employee_data[tenure_features])[0], 0)
    print("\n🚀 Predicted Tenure (Years):", round(tenure_pred, 2))

if "JobSatisfaction" in df.columns:
    satisfaction_pred = satisfaction_model.predict(new_employee_data[satisfaction_features])[0]
    print("\n😊 Predicted Job Satisfaction (1-5):", satisfaction_pred)

if promotion_column:
    promotion_pred = promotion_model.predict(new_employee_data[promotion_features])[0]
    print("\n🚀 Promotion Likelihood:", "Yes" if promotion_pred == 1 else "No")

# Optional visualization
if "OverTime_Yes" in df.columns and "Attrition_Yes" in df.columns:
    sns.boxplot(x="OverTime_Yes", y="Attrition_Yes", data=df)
    plt.title("Impact of Overtime on Attrition")
    plt.show()