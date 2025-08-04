

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- Load Dataset ---
df = pd.read_csv(r"C:\Users\Betraaj\Downloads\Desktop\internship\elevate labs\Titanic-Dataset.csv")

print("📦 Dataset loaded successfully.")
print(f"🔍 Initial shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n🧮 Missing Values (Before Cleaning):")
print(df.isnull().sum()[df.isnull().sum() > 0])

# --- Handle Missing Values ---
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns='Cabin', inplace=True)

# --- Encode Categorical Variables ---
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# --- Normalize Numerical Features ---
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# --- Remove Outliers using IQR method (Fare) ---
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

# --- Save Boxplot for Visual Reference ---
plt.figure(figsize=(8, 4))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot after Cleaning")
plt.tight_layout()
plt.savefig("boxplot_outliers.png")
plt.close()

# --- Save Cleaned Dataset ---
df.to_csv("cleaned_titanic.csv", index=False)

# --- Final Output Summary ---
print("\n✅ Data cleaning complete.")
print(f"📏 Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("🔍 Missing Values (After Cleaning):")
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() > 0 else "No missing values remaining.")
print("💾 Cleaned data saved as: cleaned_titanic.csv")
print("📊 Boxplot saved as: boxplot_outliers.png")
