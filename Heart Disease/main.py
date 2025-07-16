import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import joblib
import os

# Load the dataset
data = pd.read_csv("Heart Disease\dataset.csv")
print(data.head())
print(data["target"].value_counts())

#nonzero_mean = data.loc[data["cholesterol"] > 0, "cholesterol"].mean()
#data.loc[data["cholesterol"] == 0, "cholesterol"] = nonzero_mean

X = data.drop("target", axis=1)
y = data["target"]

# Data Preprocessing
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluating the rf model
print("Random Forest Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance using Random Forest
importance = model.feature_importances_
features = X.columns

sorted_indices = np.argsort(importance)[::-1]

# Plotting the feature importance
#plt.figure(figsize=(10, 6))
#sns.barplot(x=importance[sorted_indices], y=features[sorted_indices])
#plt.title("Feature Importance")
#plt.xlabel("Importance")
#plt.ylabel("Features")
#plt.tight_layout()
#plt.show()

# Save the model and scaler
output_dir = "Heart Disease"
os.makedirs(output_dir, exist_ok=True)
model_path = "Heart Disease/heart_disease_model.pkl"
scaler_path = "Heart Disease/heart_disease_scaler.pkl"
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
# Save the feature importance and coefficients
importance_df = pd.DataFrame({"Features": X.columns, "Importance": importance})
importance_path = "Heart Disease/importance.csv"
importance_df.to_csv(importance_path, index=False)