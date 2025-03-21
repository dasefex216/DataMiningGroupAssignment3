# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
import pickle

# Load the data
df = pd.read_csv(r"D:\UCD\Spring\Data management and mining\Group Assignment\student_dataset.csv")

# Print basic info
df.info()


# ---------------------- Data Preprocessing ----------------------
# Data Preprocessing Function
def preprocess_data(df_raw):
    """Preprocess the financial transaction data."""
    df = df_raw.copy()

    # Convert 'Transaction.Date' to datetime and extract features
    df["Transaction.Date"] = pd.to_datetime(df["Transaction.Date"], errors="coerce")
    df["Transaction.Day"] = df["Transaction.Date"].dt.day.fillna(df["Transaction.Date"].dt.day.median())
    df["Transaction.Month"] = df["Transaction.Date"].dt.month.fillna(df["Transaction.Date"].dt.month.median())
    df["Transaction.Year"] = df["Transaction.Date"].dt.year.fillna(df["Transaction.Date"].dt.year.median())
    df["Transaction.Weekday"] = df["Transaction.Date"].dt.weekday.fillna(
        df["Transaction.Date"].dt.weekday.median()
    )
    df["Is_Weekend"] = (df["Transaction.Date"].dt.weekday >= 5).astype(int)
    df["Is_Night_Transaction"] = (
            (df["Transaction.Hour"] >= 22) | (df["Transaction.Hour"] <= 6)
    ).astype(int)
    df.drop(columns=["Transaction.Date"], inplace=True)

    # Handle missing values in numeric columns
    numeric_columns = [
        "Transaction.Amount",
        "Customer.Age",
        "Account.Age.Days",
        "Transaction.Hour",
        "Quantity",
    ]
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Feature engineering: Additional features
    df["Transaction_per_AccountAge"] = df["Transaction.Amount"] / (df["Account.Age.Days"] + 1)
    df["Transaction_per_Quantity"] = df["Transaction.Amount"] / (df["Quantity"] + 1)
    df["Total_Spending"] = df["Transaction.Amount"] * df["Quantity"]

    # Feature engineering: Binning and Label Encoding for categorical features
    df["Amount_Binned"] = pd.qcut(df["Transaction.Amount"], q=3, labels=False, duplicates="drop")
    df["Hour_Category"] = pd.cut(
        df["Transaction.Hour"], bins=[0, 6, 12, 18, 24], labels=False, include_lowest=True
    )
    df["Age_Group"] = pd.cut(df["Customer.Age"], bins=[0, 25, 40, 100], labels=False, include_lowest=True)

    # Encode categorical variables
    categorical_columns = [
        "source",
        "browser",
        "Payment.Method",
        "Product.Category",
        "Device.Used",
        "Amount_Binned",
        "Hour_Category",
        "Age_Group",
    ]
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = df[col].fillna(df[col].mode()[0])  # Fill missing with mode
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Storing encoders for evaluation/pre-processing

    df_processed = df.copy()

    # Ensure 'Is.Fraudulent' is properly handled
    if "Is.Fraudulent" in df.columns:
        df_processed["Is.Fraudulent"] = df["Is.Fraudulent"].astype(int)

    return df_processed, label_encoders


# Preprocess the data
df_processed, label_encoders = preprocess_data(df)

print("\nProcessed Data Preview:")
print(df_processed.head())

# ---------------------- Feature & Target Selection ----------------------
# Define target variable and feature set
target_column = "Is.Fraudulent"
if target_column in df_processed.columns:
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
else:
    X = df_processed
    y = None

# ---------------------- Data Resampling (SMOTE) & Scaling ----------------------
# Apply SMOTE to address class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# ---------------------- Model Definition ----------------------
# Define models for evaluation
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}

# ---------------------- Hyperparameter Tuning (Random Forest) ----------------------
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled_scaled, y_resampled, test_size=0.2, random_state=42
)

# Random search parameters
param_dist = {
    "n_estimators": np.arange(50, 300, 50),
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=20, cv=3, scoring="f1", n_jobs=-1, verbose=2, random_state=42
)
random_search.fit(X_train, y_train)

# Train the best Random Forest model
best_model = random_search.best_estimator_

# ---------------------- Model Evaluation ----------------------
# Evaluate model
y_pred = best_model.predict(X_test)

print("\nBest Hyperparameters:", random_search.best_params_)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy on Test Set: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Visualize Feature Importance
feature_importances = best_model.feature_importances_
features = X.columns
sorted_idx = np.argsort(feature_importances)
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance Plot")
plt.show()

# ---------------------- Save the Model ----------------------
filename = "optimized_rf_model.pkl"
with open(filename, "wb") as f:
    pickle.dump(best_model, f)

print(f"\nModel saved as {filename}")
