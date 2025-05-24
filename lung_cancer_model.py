# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('dataset_med.csv')
df.drop(columns=["id", "diagnosis_date", "end_treatment_date"], inplace=True)
df.ffill(inplace=True)
label_cols = ['gender', 'country', 'cancer_stage', 'family_history',
              'smoking_status', 'hypertension', 'asthma',
              'cirrhosis', 'other_cancer', 'treatment_type', 'survived']

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])
scaler = StandardScaler()
num_cols = ['age', 'bmi', 'cholesterol_level']
df[num_cols] = scaler.fit_transform(df[num_cols])
X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
 importances = model.feature_importances_
feat_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feat_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show() 
import joblib
joblib.dump(model, 'lung_cancer_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
