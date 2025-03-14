import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from imblearn.over_sampling import SMOTE

# load the data
df = pd.read_csv(r'D:\UCD\Spring\Data management and mining\Group Assignment\student_dataset.csv')
df.head(100)

# Checking for missing values
print("\nChecking for Missing Values:")
print(df.isnull().sum())

 # Checking the data types of each column
print("\nDataset Information:")
df.info()

# Converting 'Transaction.Date' to datetime format
# This will helps in extracting useful time-based features
df['Transaction.Date'] = pd.to_datetime(df['Transaction.Date'])

# Extracting useful date-related features
# We extract the day, month, and weekday from the date column
df['Transaction.Day'] = df['Transaction.Date'].dt.day
df['Transaction.Month'] = df['Transaction.Date'].dt.month
df['Transaction.Weekday'] = df['Transaction.Date'].dt.weekday

# Drop the original 'Transaction.Date' column since we have extracted its useful parts
df.drop(columns=['Transaction.Date'], inplace=True)

from sklearn.preprocessing import LabelEncoder, StandardScaler
#  Encode categorical features using Label Encoding
# Label Encoding converts categorical variables into numerical values
categorical_cols = ['source', 'browser', 'Payment.Method', 'Product.Category', 'Device.Used']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical values to numerical
    label_encoders[col] = le  # Store encoders for future use

df['Transaction.Amount'] = df['Transaction.Amount'].round().astype(int)
#  Display the first few rows of the processed dataset
print("\nProcessed Data Preview:")
print(df.head())

# Checking class distribution before resampling
fraud_counts = df['Is.Fraudulent'].value_counts()
print("\nClass Distribution Before Resampling:")
print(fraud_counts)

# Class imbalance is there we will be using SMOTE to encounter it and get better results.

# Define target variable and feature set
target_column = 'Is.Fraudulent'
df[target_column] = df[target_column].astype(int)
X = df.drop(columns=[target_column])
y = df[target_column]

# Addressing class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#  Spliting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#  Performing Hyperparameter Tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(50, 200, 50),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

#  Train the best model from random search
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

#  Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy on Test Set: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#  Visualizing Model Performance
## Feature Importance Plot
feature_importances = best_model.feature_importances_
features = X.columns
sorted_idx = np.argsort(feature_importances)
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance Plot")
plt.show()

#  Ensuring the model generalizes well to new data
# Simulating new unseen data (assuming X_test is unseen data)
new_data_predictions = best_model.predict(X_test)

# Print predictions for unseen data
print("\nPredictions on New Unseen Data:")
print(new_data_predictions)

