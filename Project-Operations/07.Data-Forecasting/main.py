# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Stage 1: Data Loading and Preprocessing
def load_and_prepare_data(file_path):
    """Load and preprocess employee data from Excel file."""
    try:
        xls = pd.ExcelFile(file_path)
        employees = pd.read_excel(xls, sheet_name="Employee")
        performance = pd.read_excel(xls, sheet_name="PerformanceRating")
        df = pd.merge(employees, performance, on="EmployeeID", how="left").dropna()
        df = pd.get_dummies(df, drop_first=True)  # One-Hot Encoding for categorical variables
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
    print(f"✅ Model Accuracy: {accuracy:.2f}")
    print("📊 Classification Report:\n", report)
    return model

def train_regression_model(df, features, target):
    """Train and evaluate a regression model."""
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"🔍 Mean Absolute Error: {mae:.2f}")
    print(f"🔍 Mean Squared Error: {mse:.2f}")
    print(f"🔍 R² Score: {r2:.2f}")
    return model

# Stage 3: Prediction Function for New Employees
def predict_for_new_employee(model, features, target_type="classification"):
    """Predict outcomes for a new employee based on input data."""
    print("\n🔹 Enter employee details:")
    employee_data = {}
    for feature in features:
        value = float(input(f"{feature}: "))
        employee_data[feature] = value
    employee_df = pd.DataFrame([employee_data])
    if target_type == "classification":
        prediction = model.predict(employee_df)[0]
        return "Yes" if prediction == 1 else "No"
    else:
        return model.predict(employee_df)[0]

# Main Execution
file_path = "HrData.xlsx"
df = load_and_prepare_data(file_path)
if df is None:
    exit()

# Define common features
common_features = [
    "Age", "Education", "YearsAtCompany", "YearsInMostRecentRole",
    "YearsWithCurrManager", "EnvironmentSatisfaction", "JobSatisfaction",
    "RelationshipSatisfaction", "TrainingOpportunitiesWithinYear",
    "TrainingOpportunitiesTaken", "WorkLifeBalance", "SelfRating", "ManagerRating"
]

# Stage 4: Answer Each Question

## Question 1: Which employees are likely to leave?
if "Attrition_Yes" in df.columns:
    turnover_features = common_features + ["Salary", "OverTime_Yes"]
    turnover_model = train_classification_model(df, turnover_features, "Attrition_Yes", class_weight="balanced")
else:
    print("⚠️ 'Attrition_Yes' column not found!")

## Question 2: How long is an employee expected to stay?
if "YearsAtCompany" in df.columns:
    tenure_model = train_regression_model(df, common_features, "YearsAtCompany")
    print("\n🚀 Predicted Tenure for a New Employee:", predict_for_new_employee(tenure_model, common_features, "regression"))
else:
    print("⚠️ 'YearsAtCompany' column not found!")

## Question 3: Likelihood of leaving based on data?
# Uses the same turnover_model from Question 1
if "Attrition_Yes" in df.columns:
    print("\n🚨 Attrition Likelihood for a New Employee:", predict_for_new_employee(turnover_model, turnover_features))

## Question 4: Key factors for performance ratings?
performance_column = [col for col in df.columns if "PerformanceRating" in col]
if performance_column:
    performance_model = train_classification_model(df, common_features, performance_column[0])
    importances = performance_model.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": common_features, "Importance": importances}).sort_values(by="Importance", ascending=False)
    print("\n🔑 Key Factors for Performance Ratings:\n", feature_importance_df)
else:
    print("⚠️ 'PerformanceRating' column not found!")

## Question 5: Level of job satisfaction?
if "JobSatisfaction" in df.columns:
    satisfaction_model = train_classification_model(df, common_features, "JobSatisfaction")
    print("\n😊 Job Satisfaction for a New Employee:", predict_for_new_employee(satisfaction_model, common_features))
else:
    print("⚠️ 'JobSatisfaction' column not found!")

## Question 6: Which employees are likely to be promoted?
promotion_column = [col for col in df.columns if "Promotion" in col]
if not promotion_column and "YearsSinceLastPromotion" in df.columns:
    df["Promotion"] = (df["YearsSinceLastPromotion"] == 0).astype(int)
    promotion_column = ["Promotion"]
if promotion_column:
    promotion_model = train_classification_model(df, common_features, promotion_column[0], class_weight="balanced")
    print("\n🚀 Promotion Likelihood for a New Employee:", predict_for_new_employee(promotion_model, common_features))
else:
    print("⚠️ No 'Promotion' or 'YearsSinceLastPromotion' column found!")

## Question 7: High-risk groups and strategies?
if "Attrition_Yes" in df.columns:
    high_risk_employees = df[df["Attrition_Yes"] == 1]
    avg_job_satisfaction = high_risk_employees["JobSatisfaction"].mean() if "JobSatisfaction" in df.columns else "N/A"
    print(f"\n🚨 Average Job Satisfaction for High-Risk Employees: {avg_job_satisfaction}")
    print("Strategies to Reduce Attrition:")
    print("- Enhance job satisfaction through better work conditions.")
    print("- Reduce overtime and improve work-life balance.")
    print("- Offer competitive salaries and career growth opportunities.")
else:
    print("⚠️ 'Attrition_Yes' column not found for high-risk analysis!")

## Question 8: Does overtime increase attrition likelihood?
if "OverTime_Yes" in df.columns and "Attrition_Yes" in df.columns:
    sns.boxplot(x="OverTime_Yes", y="Attrition_Yes", data=df)
    plt.title("Impact of Overtime on Attrition")
    plt.show()
else:
    print("⚠️ Required columns for overtime analysis not found!")

## Question 9: Expected salary based on department, experience, and job role?
salary_features = [col for col in df.columns if "Department" in col or "JobRole" in col] + ["YearsAtCompany", "Education"]
if "Salary" in df.columns and salary_features:
    salary_model = train_regression_model(df, salary_features, "Salary")
    print("\n💰 Expected Salary for a New Employee:", predict_for_new_employee(salary_model, salary_features, "regression"))
else:
    print("⚠️ Required columns for salary prediction not found!")