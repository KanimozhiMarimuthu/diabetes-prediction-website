import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

X_train = np.load("processed/X_train.npy")
X_test = np.load("processed/X_test.npy")
y_train = np.load("processed/y_train.npy")
y_test = np.load("processed/y_test.npy")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy     : {accuracy:.2f}")
print(f"Precision    : {precision:.2f}")
print(f"Recall       : {recall:.2f}")
print(f"F1 Score     : {f1:.2f}")
print("Confusion Matrix:")
print(cm)

model.save_model("models/xgb_model.json")

print("Training completed and model saved!")