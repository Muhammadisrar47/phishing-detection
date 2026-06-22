import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("phishing_Dataset/dataset_phishing.csv")
# =========================
# 1. Remove Missing Values
# =========================
# Display missing values
print("Missing Values Before:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

# =========================
# 2. Remove Duplicate Records
# =========================

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# =========================
# 3. Convert Categorical Data
# =========================

label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# =========================
# 4. Normalize Numerical Data
# =========================

scaler = StandardScaler()

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# =========================
# Final Dataset Information
# =========================

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved as 'cleaned_dataset.csv'")

# =========================
# 5. Train a phishing detection model
# =========================

X = df.drop(columns=["url", "status"])
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save trained model
import joblib
joblib.dump(model, "phishing_model.joblib")
print("\nTrained model saved as 'phishing_model.joblib'")