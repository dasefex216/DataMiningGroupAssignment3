# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the data
df = pd.read_csv(r'D:\UCD\Spring\Data management and mining\Group Assignment\student_dataset.csv')
df.info()


# Data Preprocessing Function
def preprocess_data(df_raw):
    """Preprocess the financial transaction data."""
    df = df_raw.copy()

    # Convert Transaction.Date to datetime with mixed format handling
    df['Transaction.Date'] = pd.to_datetime(df['Transaction.Date'], format='mixed', errors='coerce')
    df['Transaction_Day'] = df['Transaction.Date'].dt.day
    df['Transaction_Month'] = df['Transaction.Date'].dt.month
    df['Transaction_Year'] = df['Transaction.Date'].dt.year

    # Handle missing values in date-derived columns
    df['Transaction_Day'] = df['Transaction_Day'].fillna(df['Transaction_Day'].median())
    df['Transaction_Month'] = df['Transaction_Month'].fillna(df['Transaction_Month'].median())
    df['Transaction_Year'] = df['Transaction_Year'].fillna(df['Transaction_Year'].median())

    # Handle missing values in numeric columns
    numeric_columns = ['Transaction.Amount', 'Customer.Age', 'Account.Age.Days', 'Transaction.Hour', 'Quantity']
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical variables
    categorical_columns = ['source', 'browser', 'Payment.Method', 'Product.Category', 'Device.Used']
    df_encoded = pd.get_dummies(df[categorical_columns], drop_first=True)

    # Combine features
    numeric_features = df[numeric_columns + ['Transaction_Day', 'Transaction_Month', 'Transaction_Year']]
    df_processed = pd.concat([numeric_features, df_encoded], axis=1)
    df_processed['Is.Fraudulent'] = df['Is.Fraudulent']

    # Drop or impute missing values in the target column
    df_processed = df_processed.dropna(subset=['Is.Fraudulent'])
    df_processed['Is.Fraudulent'] = df_processed['Is.Fraudulent'].fillna(df_processed['Is.Fraudulent'].mode()[0])

    return df_processed


# Preprocess the data
df_processed = preprocess_data(df)
print("\nProcessed Data Shape:", df_processed.shape)
print("\nProcessed Data Columns:", df_processed.columns.tolist())

# Split features and target
X = df_processed.drop(columns=['Is.Fraudulent'])
y = df_processed['Is.Fraudulent']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC Curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Save the trained model to disk
filename = 'rf_model.pkl'
with open(filename, "wb") as f:
    pickle.dump(rf_model, f)

# Load evaluation data
df_eval_raw = pd.read_csv('evaluation_dataset.csv')
df_eval = preprocess_data(df_eval_raw)

# Load the model and evaluate it on the evaluation data
with open(filename, "rb") as f:
    eval_model = pickle.load(f)

# Test the model on the evaluation data
y_eval = eval_model.predict(df_eval.drop('Is.Fraudulent', axis=1))

# Calculate the F1 score
f1_eval = f1_score(df_eval['Is.Fraudulent'], y_eval)

# Print the F1 evaluation score
print(f'\nEvaluation F1 Score: {f1_eval:.3f}')

# Create a comparison table
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Probability': y_pred_proba})
print(comparison_df.head())
