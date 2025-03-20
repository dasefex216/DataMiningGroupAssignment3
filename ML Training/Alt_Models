# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_curve

# Load dataset (replace this with your actual file path)
df = pd.read_csv("student_dataset.csv")

# Print basic info
df.info()

# Data Preprocessing Function
def preprocess_data(df_raw):
    """Preprocess the financial transaction data."""
    df = df_raw.copy()

    # Convert Transaction.Date to datetime
    df['Transaction.Date'] = pd.to_datetime(df['Transaction.Date'], errors='coerce')
    df['Transaction_Day'] = df['Transaction.Date'].dt.day.fillna(df['Transaction.Date'].dt.day.median())
    df['Transaction_Month'] = df['Transaction.Date'].dt.month.fillna(df['Transaction.Date'].dt.month.median())
    df['Transaction_Year'] = df['Transaction.Date'].dt.year.fillna(df['Transaction.Date'].dt.year.median())

    # Handle missing values in numeric columns
    numeric_columns = ['Transaction.Amount', 'Customer.Age', 'Account.Age.Days', 
                      'Transaction.Hour', 'Quantity']
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Encode categorical variables using one-hot encoding
    categorical_columns = ['source', 'browser', 'Payment.Method', 
                           'Product.Category', 'Device.Used']
    df_encoded = pd.get_dummies(df[categorical_columns], drop_first=True)

    # Create new time-based features
    df['Is_Weekend'] = (df['Transaction.Date'].dt.weekday >= 5).astype(int)
    df['Is_Night_Transaction'] = ((df['Transaction.Hour'] >= 22) | (df['Transaction.Hour'] <= 6)).astype(int)

    # Create new ratio-based features
    df['Transaction_per_AccountAge'] = df['Transaction.Amount'] / (df['Account.Age.Days'] + 1)
    df['Transaction_per_Quantity'] = df['Transaction.Amount'] / (df['Quantity'] + 1)

    # Combine all features
    numeric_features = df[numeric_columns + ['Transaction_Day', 'Transaction_Month', 'Transaction_Year', 
                                             'Is_Weekend', 'Is_Night_Transaction', 
                                             'Transaction_per_AccountAge', 'Transaction_per_Quantity']]
    df_processed = pd.concat([numeric_features, df_encoded], axis=1)
    df_processed['Is.Fraudulent'] = df['Is.Fraudulent']

    return df_processed

# Preprocess the data
df_processed = preprocess_data(df)

print("\nProcessed Data Shape:", df_processed.shape)
print("\nProcessed Data Columns:", df_processed.columns.tolist())

# Define features and target variable
X = df_processed.drop(columns=['Is.Fraudulent'])
y = df_processed['Is.Fraudulent']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize numerical features
scaler = StandardScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
}

# Add Regression Models (thresholding to work for classification)
regression_models = {
    "Ordinary Least Squares (OLS)": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.01)
}

# Train and evaluate classification models
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_resampled_scaled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({'Model': name, 'Accuracy': acc, 'F1 Score': f1})
    
    print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Train and evaluate regression models (convert predictions to classification)
for name, model in regression_models.items():
    print(f"Training {name}...")
    model.fit(X_train_resampled_scaled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)  # Continuous output
    y_pred_class = (y_pred >= 0.5).astype(int)  # Convert to binary classification

    acc = accuracy_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)

    results.append({'Model': name, 'Accuracy': acc, 'F1 Score': f1})

    print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred_class))
    print("-" * 50)

# Convert results into DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)

# Plot results
plt.figure(figsize=(12, 6))
sns.barplot(x="Model", y="F1 Score", data=results_df)
plt.title("Model Comparison - F1 Score")
plt.xticks(rotation=45)
plt.show()
