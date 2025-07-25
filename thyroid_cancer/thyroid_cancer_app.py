import pandas as pd

# Load your dataset
df = pd.read_csv("thyroid_cancer/dataset.csv")

# Print unique values for each column
for col in df.columns:
    print(f"Column: {col}")
    print(df[col].unique())
    print("-" * 40)