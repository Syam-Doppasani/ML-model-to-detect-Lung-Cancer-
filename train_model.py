import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and clean dataset
df = pd.read_csv('dataset_med.csv')
df.drop(columns=["id", "diagnosis_date", "end_treatment_date"], inplace=True)
df.ffill(inplace=True)

# Encode categorical columns
label_cols = ['gender', 'country', 'cancer_stage', 'family_history',
              'smoking_status', 'hypertension', 'asthma',
              'cirrhosis', 'other_cancer', 'treatment_type', 'survived']

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Scale numerical columns
scaler = StandardScaler()
num_cols = ['age', 'bmi', 'cholesterol_level']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Split data
X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessors
joblib.dump(model, 'lung_cancer_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and encoders saved successfully.")
