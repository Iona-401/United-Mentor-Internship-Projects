import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
file_path = "Lung Cancer\dataset_med.csv"
df = pd.read_csv(file_path)
print(f"Original dataset shape: {df.shape}")

# Drop unnecessary columns
df_clean = df.drop(columns=["id", "country", "diagnosis_date", "end_treatment_date"])

# Map categorical variables
df_clean["gender"] = df_clean["gender"].map({"Male": 0, "Female": 1})  
df_clean["cancer_stage"] = df_clean["cancer_stage"].map({
    "Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3
})  
df_clean["smoking_status"] = df_clean["smoking_status"].map({
    "Never Smoked": 0, "Former Smoker": 1, "Current Smoker": 2
})    
df_clean["treatment_type"] = df_clean["treatment_type"].map({
    "Surgery": 0, "Chemotherapy": 1, "Radiation": 2, "Immunotherapy": 3
})
df_clean["family_history"] = df_clean["family_history"].map({"Yes": 1, "No": 0})

print("Data preprocessing completed.")

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 24))

# 1. Class Distribution (Survival)
plt.subplot(6, 3, 1)
survival_counts = df_clean['survived'].value_counts()
plt.pie(survival_counts.values, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=90)
plt.title('Survival Distribution (Target Variable)', fontsize=14, fontweight='bold')

# 2. Survival count bar plot
plt.subplot(6, 3, 2)
sns.countplot(data=df_clean, x='survived', palette=['red', 'green'])
plt.title('Survival Count Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Survived (0=No, 1=Yes)')
for i, v in enumerate(survival_counts.values):
    plt.text(i, v + 10, str(v), ha='center', fontweight='bold')

# 3. Age Distribution by Survival
plt.subplot(6, 3, 3)
sns.boxplot(data=df_clean, x='survived', y='age', palette=['red', 'green'])
plt.title('Age Distribution by Survival Status', fontsize=14, fontweight='bold')
plt.xlabel('Survived (0=No, 1=Yes)')

# 4. Cancer Stage Distribution
plt.subplot(6, 3, 4)
stage_counts = df_clean['cancer_stage'].value_counts().sort_index()
stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
plt.bar(range(len(stage_counts)), stage_counts.values, color=['lightgreen', 'yellow', 'orange', 'red'])
plt.title('Cancer Stage Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Cancer Stage')
plt.xticks(range(len(stage_labels)), stage_labels)
for i, v in enumerate(stage_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')

# 5. Survival by Cancer Stage
plt.subplot(6, 3, 5)
survival_by_stage = pd.crosstab(df_clean['cancer_stage'], df_clean['survived'], normalize='index') * 100
survival_by_stage.plot(kind='bar', stacked=True, color=['red', 'green'], ax=plt.gca())
plt.title('Survival Rate by Cancer Stage', fontsize=14, fontweight='bold')
plt.xlabel('Cancer Stage')
plt.ylabel('Percentage')
plt.xticks(range(4), stage_labels, rotation=45)
plt.legend(['Not Survived', 'Survived'])

# 6. Smoking Status Distribution
plt.subplot(6, 3, 6)
smoking_counts = df_clean['smoking_status'].value_counts().sort_index()
smoking_labels = ['Never Smoked', 'Former Smoker', 'Current Smoker']
plt.bar(range(len(smoking_counts)), smoking_counts.values, color=['lightblue', 'orange', 'darkred'])
plt.title('Smoking Status Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Smoking Status')
plt.xticks(range(len(smoking_labels)), smoking_labels, rotation=45)
for i, v in enumerate(smoking_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')

# 7. Survival by Smoking Status
plt.subplot(6, 3, 7)
survival_by_smoking = pd.crosstab(df_clean['smoking_status'], df_clean['survived'], normalize='index') * 100
survival_by_smoking.plot(kind='bar', stacked=True, color=['red', 'green'], ax=plt.gca())
plt.title('Survival Rate by Smoking Status', fontsize=14, fontweight='bold')
plt.xlabel('Smoking Status')
plt.ylabel('Percentage')
plt.xticks(range(3), smoking_labels, rotation=45)
plt.legend(['Not Survived', 'Survived'])

# 8. Treatment Type Distribution
plt.subplot(6, 3, 8)
treatment_counts = df_clean['treatment_type'].value_counts().sort_index()
treatment_labels = ['Surgery', 'Chemotherapy', 'Radiation', 'Immunotherapy']
plt.bar(range(len(treatment_counts)), treatment_counts.values, color=['blue', 'purple', 'orange', 'green'])
plt.title('Treatment Type Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Treatment Type')
plt.xticks(range(len(treatment_labels)), treatment_labels, rotation=45)
for i, v in enumerate(treatment_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')

# 9. Survival by Treatment Type
plt.subplot(6, 3, 9)
survival_by_treatment = pd.crosstab(df_clean['treatment_type'], df_clean['survived'], normalize='index') * 100
survival_by_treatment.plot(kind='bar', stacked=True, color=['red', 'green'], ax=plt.gca())
plt.title('Survival Rate by Treatment Type', fontsize=14, fontweight='bold')
plt.xlabel('Treatment Type')
plt.ylabel('Percentage')
plt.xticks(range(4), treatment_labels, rotation=45)
plt.legend(['Not Survived', 'Survived'])

# 10. BMI Distribution by Survival
plt.subplot(6, 3, 10)
sns.boxplot(data=df_clean, x='survived', y='bmi', palette=['red', 'green'])
plt.title('BMI Distribution by Survival Status', fontsize=14, fontweight='bold')
plt.xlabel('Survived (0=No, 1=Yes)')

# 11. Gender Distribution
plt.subplot(6, 3, 11)
gender_counts = df_clean['gender'].value_counts()
plt.pie(gender_counts.values, labels=['Male', 'Female'], autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution', fontsize=14, fontweight='bold')

# 12. Survival by Gender
plt.subplot(6, 3, 12)
survival_by_gender = pd.crosstab(df_clean['gender'], df_clean['survived'], normalize='index') * 100
survival_by_gender.plot(kind='bar', stacked=True, color=['red', 'green'], ax=plt.gca())
plt.title('Survival Rate by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.xticks([0, 1], ['Male', 'Female'], rotation=0)
plt.legend(['Not Survived', 'Survived'])

# 13. Cholesterol Distribution by Survival
plt.subplot(6, 3, 13)
sns.boxplot(data=df_clean, x='survived', y='cholesterol_level', palette=['red', 'green'])
plt.title('Cholesterol Level by Survival Status', fontsize=14, fontweight='bold')
plt.xlabel('Survived (0=No, 1=Yes)')

# 14. Family History Distribution
plt.subplot(6, 3, 14)
family_counts = df_clean['family_history'].value_counts()
plt.pie(family_counts.values, labels=['No Family History', 'Family History'], autopct='%1.1f%%', startangle=90)
plt.title('Family History Distribution', fontsize=14, fontweight='bold')

# 15. Survival by Family History
plt.subplot(6, 3, 15)
survival_by_family = pd.crosstab(df_clean['family_history'], df_clean['survived'], normalize='index') * 100
survival_by_family.plot(kind='bar', stacked=True, color=['red', 'green'], ax=plt.gca())
plt.title('Survival Rate by Family History', fontsize=14, fontweight='bold')
plt.xlabel('Family History')
plt.ylabel('Percentage')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.legend(['Not Survived', 'Survived'])

# 16. Hypertension Distribution
plt.subplot(6, 3, 16)
hypertension_counts = df_clean['hypertension'].value_counts()
plt.pie(hypertension_counts.values, labels=['No Hypertension', 'Hypertension'], autopct='%1.1f%%', startangle=90)
plt.title('Hypertension Distribution', fontsize=14, fontweight='bold')

# 17. Survival by Hypertension
plt.subplot(6, 3, 17)
survival_by_hypertension = pd.crosstab(df_clean['hypertension'], df_clean['survived'], normalize='index') * 100
survival_by_hypertension.plot(kind='bar', stacked=True, color=['red', 'green'], ax=plt.gca())
plt.title('Survival Rate by Hypertension', fontsize=14, fontweight='bold')
plt.xlabel('Hypertension')
plt.ylabel('Percentage')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.legend(['Not Survived', 'Survived'])

# 18. Correlation Heatmap
plt.subplot(6, 3, 18)
# Select only numeric columns for correlation
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
correlation_matrix = df_clean[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


# Print detailed statistics
print("\n" + "="*50)
print("DETAILED DATA ANALYSIS")
print("="*50)

print(f"\n1. Dataset Overview:")
print(f"   - Total samples: {len(df_clean)}")
print(f"   - Features: {len(df_clean.columns) - 1}")
print(f"   - Missing values: {df_clean.isnull().sum().sum()}")

print(f"\n2. Target Variable (Survival) Distribution:")
survival_dist = df_clean['survived'].value_counts(normalize=True) * 100
print(f"   - Not Survived (0): {survival_dist[0]:.1f}% ({df_clean['survived'].value_counts()[0]} samples)")
print(f"   - Survived (1): {survival_dist[1]:.1f}% ({df_clean['survived'].value_counts()[1]} samples)")
print(f"   - Class Imbalance Ratio: {survival_dist[0]/survival_dist[1]:.2f}:1")

print(f"\n3. Cancer Stage Analysis:")
for stage in range(4):
    stage_data = df_clean[df_clean['cancer_stage'] == stage]
    if len(stage_data) > 0:
        survival_rate = stage_data['survived'].mean() * 100
        print(f"   - Stage {stage+1}: {len(stage_data)} patients, {survival_rate:.1f}% survival rate")

print(f"\n4. Smoking Status Analysis:")
smoking_labels = ['Never Smoked', 'Former Smoker', 'Current Smoker']
for i, label in enumerate(smoking_labels):
    smoking_data = df_clean[df_clean['smoking_status'] == i]
    if len(smoking_data) > 0:
        survival_rate = smoking_data['survived'].mean() * 100
        print(f"   - {label}: {len(smoking_data)} patients, {survival_rate:.1f}% survival rate")

print(f"\n5. Treatment Type Analysis:")
treatment_labels = ['Surgery', 'Chemotherapy', 'Radiation', 'Immunotherapy']
for i, label in enumerate(treatment_labels):
    treatment_data = df_clean[df_clean['treatment_type'] == i]
    if len(treatment_data) > 0:
        survival_rate = treatment_data['survived'].mean() * 100
        print(f"   - {label}: {len(treatment_data)} patients, {survival_rate:.1f}% survival rate")

print(f"\n6. Numerical Features Summary:")
numeric_features = ['age', 'bmi', 'cholesterol_level']
for feature in numeric_features:
    feature_data = df_clean[feature].dropna()
    print(f"   - {feature.capitalize()}: Mean={feature_data.mean():.2f}, Std={feature_data.std():.2f}, Range=[{feature_data.min():.1f}, {feature_data.max():.1f}]")

# Identify potential data quality issues
print(f"\n7. Data Quality Issues:")
print(f"   - Missing values per column:")
missing_counts = df_clean.isnull().sum()
for col, count in missing_counts.items():
    if count > 0:
        print(f"     * {col}: {count} missing ({count/len(df_clean)*100:.1f}%)")

print(f"\n8. Feature Importance Insights:")
print("   - Features most correlated with survival:")
correlations = df_clean.corr()['survived'].abs().sort_values(ascending=False)
for feature, corr in correlations.items():
    if feature != 'survived' and corr > 0.1:
        print(f"     * {feature}: {corr:.3f}")