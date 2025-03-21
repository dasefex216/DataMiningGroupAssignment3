# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load dataset
df = pd.read_csv(r"D:\UCD\Spring\Data management and mining\Group Assignment\student_dataset.csv")


# -------------------------------- Preprocessing Without Feature Engineering --------------------------------

def preprocess_no_feature_engineering(df_raw):
    """Preprocess the data without additional feature engineering."""
    df = df_raw.copy()

    # Convert Transaction.Date to datetime
    df["Transaction.Date"] = pd.to_datetime(df["Transaction.Date"], errors="coerce")
    df["Transaction_Day"] = df["Transaction.Date"].dt.day.fillna(df["Transaction.Date"].dt.day.median())
    df["Transaction_Month"] = df["Transaction.Date"].dt.month.fillna(df["Transaction.Date"].dt.month.median())
    df["Transaction_Year"] = df["Transaction.Date"].dt.year.fillna(df["Transaction.Date"].dt.year.median())
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

    # Encode Categorical Variables
    categorical_columns = [
        "source",
        "browser",
        "Payment.Method",
        "Product.Category",
        "Device.Used",
    ]
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    if "Is.Fraudulent" in df.columns:
        df["Is.Fraudulent"] = df["Is.Fraudulent"].astype(int)

    return df


# -------------------------------- Preprocessing With Feature Engineering --------------------------------

def preprocess_with_feature_engineering(df_raw):
    """Preprocess the data with advanced feature engineering."""
    df = preprocess_no_feature_engineering(df_raw)  # Start with basic preprocessing

    # Add Feature Engineering
    df["Transaction_per_AccountAge"] = df["Transaction.Amount"] / (df["Account.Age.Days"] + 1)
    df["Transaction_per_Quantity"] = df["Transaction.Amount"] / (df["Quantity"] + 1)
    df["Total_Spending"] = df["Transaction.Amount"] * df["Quantity"]

    # Spending Speed
    df["Spending_Speed"] = df["Total_Spending"] / (df["Account.Age.Days"] + 1)

    # High Amount Transaction Flag
    threshold = np.percentile(df["Transaction.Amount"], 75)
    df["High_Amount_Transaction"] = (df["Transaction.Amount"] > threshold).astype(int)

    return df


# -------------------------------- Model Training & Evaluation Function --------------------------------

def train_and_evaluate(X, y, description="Model"):
    """Train and evaluate a Random Forest classifier."""
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    # Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Output
    print(f"\n--- {description} ---")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {description}")
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {description}")
    plt.legend(loc="lower right")
    plt.show()


# -------------------------------- Pre-Feature Engineering Evaluation --------------------------------

# Preprocessing without Feature Engineering
df_no_fe = preprocess_no_feature_engineering(df)

# Split features and target
X_no_fe = df_no_fe.drop(columns=["Is.Fraudulent"])
y_no_fe = df_no_fe["Is.Fraudulent"]

# Train and evaluate without feature engineering
train_and_evaluate(X_no_fe, y_no_fe, description="Pre-Feature Engineering Model")

# -------------------------------- Post-Feature Engineering Evaluation --------------------------------

# Preprocessing with Feature Engineering
df_with_fe = preprocess_with_feature_engineering(df)

# Split features and target
X_with_fe = df_with_fe.drop(columns=["Is.Fraudulent"])
y_with_fe = df_with_fe["Is.Fraudulent"]

# Train and evaluate with feature engineering
train_and_evaluate(X_with_fe, y_with_fe, description="Post-Feature Engineering Model")
