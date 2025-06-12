# 🫁Lung Cancer Survival Prediction using ML and Streamlit for UI
This project is a machine learning-powered web application that predicts the survival chances of lung cancer patients based on their clinical data and medical history. Built using Streamlit, the app provides a user-friendly interface for doctors, researchers, or healthcare providers to input patient information and receive instant predictions.
# 🔍Overview
Lung cancer is one of the most prevalent and deadly cancers worldwide. Early prognosis and survival prediction are critical for effective treatment planning. This app uses a Random Forest Classifier trained on a realistic dataset to predict whether a patient is likely to survive.
# 🚀 Features
-Predict survival chances based on medical and demographic data

-Simple form input for fields like age, gender, cancer stage, BMI, etc.

-Real-time machine learning prediction

-Preprocessing includes Label Encoding and Standard Scaling

-Lightweight and easy to deploy using Streamlit
# 🧠Machine Learning Model
Model:

Random Forest Classifier (sklearn)

Training set: Cleaned and encoded from dataset_med.csv

Preprocessing:

Label Encoding for categorical features

StandardScaler for numerical features

Target variable:

survived (1 = survived, 0 = did not survive)

# Project Structure
```
lung_cancer_project/

├── app.py                           # Streamlit app interface
├── train_model.py                 # Model training and preprocessing
├── dataset_med.csv                # Dataset used for training
├── lung_cancer_model.pkl          # Trained model
├── label_encoder.pkl              # LabelEncoder used for encoding categories
├── scaler.pkl                     # StandardScaler for numeric inputs
└── requirements.txt               # Project dependencies
```
# 🛠️ Installation & Running Locally
1. Clone the repo

git clone https://github.com/Syam-Doppasani/ML-model-to-detect-Lung-Cancer-

cd lung-cancer-survival-prediction

2. Install dependencies

pip install -r requirements.txt

  3. Train the model (only once, unless retraining needed)

python train_model.py

  4. Run the Streamlit app

streamlit run app.py


# 📥Input Features
Feature	Description

Age	Age of the patient

Gender	Male / Female

Country	Country of residence

Cancer Stage	Stage 1 to Stage 4

Family History	Yes / No

Smoking Status	Never / Former / Current

BMI	Body Mass Index

Cholesterol Level	Serum cholesterol level

Hypertension	Yes / No

Asthma	Yes / No

Cirrhosis	Yes / No

Other Cancer	Yes / No

Treatment Type	Chemotherapy / Surgery / etc.


# 👨‍💻 Author
Syam Doppasani

For freelance work or collaborations: syamdoppasani@gmail.com
