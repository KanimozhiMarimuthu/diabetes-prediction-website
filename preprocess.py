import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "data/diabetes.csv"
PROCESSED_DIR = "processed"
MODEL_DIR = "models"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGET = "Outcome"
X = df.drop(columns=[TARGET])
y = df[TARGET]

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save(f"{PROCESSED_DIR}/X_train.npy", X_train)
np.save(f"{PROCESSED_DIR}/X_test.npy", X_test)
np.save(f"{PROCESSED_DIR}/y_train.npy", y_train)
np.save(f"{PROCESSED_DIR}/y_test.npy", y_test)

joblib.dump(imputer, f"{MODEL_DIR}/imputer.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

print("Preprocessing completed successfully!")