import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv("Heart Disease\dataset.csv")
print(data.head())
print(data["target"].value_counts())

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

# Evaluating the model
print("Random Forest Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance using Random Forest
importance = model.feature_importances_
features = X.columns

sorted_indices = np.argsort(importance)[::-1]

# Plotting the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importance[sorted_indices], y=features[sorted_indices])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Logistic Regression Model to analyze coefficients
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
coefficients = log_model.coef_[0]
coef_4f = pd.DataFrame({"Features": X.columns, "Coefficients": coefficients}).sort_values(by="Coefficients", key=abs, ascending=False)

# Plotting the coefficients of the logistic regression model
plt.figure(figsize=(10, 6))
sns.barplot(x="Coefficients", y="Features", data=coef_4f)
plt.title("Logistic Regression Coefficients")
plt.xlabel("Effects on Probability of Heart Disease")
plt.tight_layout()
plt.show()

# Plotting the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()