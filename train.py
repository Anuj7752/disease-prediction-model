import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Features and Target — 27 symptoms
symptoms = [
    "fever", "cough", "fatigue", "difficulty_breathing",
    "chest_pain", "body_ache", "loss_of_taste_or_smell",
    "headache", "sore_throat", "rash",
    "nausea", "vomiting", "diarrhea", "abdominal_pain",
    "joint_pain", "sweating", "chills", "runny_nose",
    "sneezing", "muscle_weakness", "dizziness", "blurred_vision",
    "frequent_urination", "excessive_thirst", "weight_loss",
    "swollen_lymph_nodes", "high_blood_pressure"
]

diseases = [
    "Flu", "COVID-19", "Common Cold", "Asthma", "Dengue",
    "Malaria", "Typhoid", "Pneumonia", "Diabetes",
    "Migraine", "Gastroenteritis", "Hypertension"
]

# Generate synthetic data
def generate_data(num_samples=1200):
    np.random.seed(42)
    data = []

    for _ in range(num_samples):
        disease = np.random.choice(diseases)
        row = {sym: 0 for sym in symptoms}

        if disease == "Flu":
            row["fever"]      = np.random.choice([0,1], p=[0.1, 0.9])
            row["cough"]      = np.random.choice([0,1], p=[0.2, 0.8])
            row["fatigue"]    = np.random.choice([0,1], p=[0.3, 0.7])
            row["body_ache"]  = np.random.choice([0,1], p=[0.4, 0.6])
            row["chills"]     = np.random.choice([0,1], p=[0.4, 0.6])
            row["headache"]   = np.random.choice([0,1], p=[0.5, 0.5])

        elif disease == "COVID-19":
            row["fever"]                  = np.random.choice([0,1], p=[0.2, 0.8])
            row["cough"]                  = np.random.choice([0,1], p=[0.3, 0.7])
            row["loss_of_taste_or_smell"] = np.random.choice([0,1], p=[0.2, 0.8])
            row["difficulty_breathing"]   = np.random.choice([0,1], p=[0.4, 0.6])
            row["fatigue"]                = np.random.choice([0,1], p=[0.4, 0.6])
            row["body_ache"]              = np.random.choice([0,1], p=[0.5, 0.5])

        elif disease == "Common Cold":
            row["cough"]       = np.random.choice([0,1], p=[0.1, 0.9])
            row["sore_throat"] = np.random.choice([0,1], p=[0.2, 0.8])
            row["runny_nose"]  = np.random.choice([0,1], p=[0.1, 0.9])
            row["sneezing"]    = np.random.choice([0,1], p=[0.2, 0.8])
            row["headache"]    = np.random.choice([0,1], p=[0.6, 0.4])

        elif disease == "Asthma":
            row["difficulty_breathing"] = np.random.choice([0,1], p=[0.1, 0.9])
            row["chest_pain"]           = np.random.choice([0,1], p=[0.4, 0.6])
            row["cough"]                = np.random.choice([0,1], p=[0.3, 0.7])
            row["fatigue"]              = np.random.choice([0,1], p=[0.5, 0.5])

        elif disease == "Dengue":
            row["fever"]      = np.random.choice([0,1], p=[0.05, 0.95])
            row["body_ache"]  = np.random.choice([0,1], p=[0.1, 0.9])
            row["rash"]       = np.random.choice([0,1], p=[0.2, 0.8])
            row["headache"]   = np.random.choice([0,1], p=[0.2, 0.8])
            row["joint_pain"] = np.random.choice([0,1], p=[0.2, 0.8])
            row["nausea"]     = np.random.choice([0,1], p=[0.4, 0.6])

        elif disease == "Malaria":
            row["fever"]    = np.random.choice([0,1], p=[0.05, 0.95])
            row["chills"]   = np.random.choice([0,1], p=[0.1, 0.9])
            row["sweating"] = np.random.choice([0,1], p=[0.1, 0.9])
            row["headache"] = np.random.choice([0,1], p=[0.2, 0.8])
            row["nausea"]   = np.random.choice([0,1], p=[0.3, 0.7])
            row["vomiting"] = np.random.choice([0,1], p=[0.4, 0.6])

        elif disease == "Typhoid":
            row["fever"]         = np.random.choice([0,1], p=[0.05, 0.95])
            row["abdominal_pain"]= np.random.choice([0,1], p=[0.2, 0.8])
            row["headache"]      = np.random.choice([0,1], p=[0.3, 0.7])
            row["fatigue"]       = np.random.choice([0,1], p=[0.3, 0.7])
            row["diarrhea"]      = np.random.choice([0,1], p=[0.4, 0.6])
            row["nausea"]        = np.random.choice([0,1], p=[0.4, 0.6])

        elif disease == "Pneumonia":
            row["fever"]                = np.random.choice([0,1], p=[0.1, 0.9])
            row["cough"]                = np.random.choice([0,1], p=[0.1, 0.9])
            row["difficulty_breathing"] = np.random.choice([0,1], p=[0.2, 0.8])
            row["chest_pain"]           = np.random.choice([0,1], p=[0.3, 0.7])
            row["fatigue"]              = np.random.choice([0,1], p=[0.3, 0.7])
            row["chills"]               = np.random.choice([0,1], p=[0.4, 0.6])

        elif disease == "Diabetes":
            row["frequent_urination"] = np.random.choice([0,1], p=[0.1, 0.9])
            row["excessive_thirst"]   = np.random.choice([0,1], p=[0.1, 0.9])
            row["weight_loss"]        = np.random.choice([0,1], p=[0.2, 0.8])
            row["fatigue"]            = np.random.choice([0,1], p=[0.2, 0.8])
            row["blurred_vision"]     = np.random.choice([0,1], p=[0.3, 0.7])
            row["muscle_weakness"]    = np.random.choice([0,1], p=[0.4, 0.6])

        elif disease == "Migraine":
            row["headache"]         = np.random.choice([0,1], p=[0.05, 0.95])
            row["nausea"]           = np.random.choice([0,1], p=[0.2, 0.8])
            row["vomiting"]         = np.random.choice([0,1], p=[0.3, 0.7])
            row["blurred_vision"]   = np.random.choice([0,1], p=[0.3, 0.7])
            row["dizziness"]        = np.random.choice([0,1], p=[0.3, 0.7])
            row["fatigue"]          = np.random.choice([0,1], p=[0.4, 0.6])

        elif disease == "Gastroenteritis":
            row["nausea"]         = np.random.choice([0,1], p=[0.1, 0.9])
            row["vomiting"]       = np.random.choice([0,1], p=[0.1, 0.9])
            row["diarrhea"]       = np.random.choice([0,1], p=[0.1, 0.9])
            row["abdominal_pain"] = np.random.choice([0,1], p=[0.2, 0.8])
            row["fever"]          = np.random.choice([0,1], p=[0.5, 0.5])
            row["fatigue"]        = np.random.choice([0,1], p=[0.4, 0.6])

        elif disease == "Hypertension":
            row["headache"]          = np.random.choice([0,1], p=[0.2, 0.8])
            row["dizziness"]         = np.random.choice([0,1], p=[0.2, 0.8])
            row["high_blood_pressure"]= np.random.choice([0,1], p=[0.05, 0.95])
            row["chest_pain"]        = np.random.choice([0,1], p=[0.4, 0.6])
            row["blurred_vision"]    = np.random.choice([0,1], p=[0.4, 0.6])
            row["nausea"]            = np.random.choice([0,1], p=[0.5, 0.5])

        row["Disease"] = disease
        data.append(row)

    return pd.DataFrame(data)

df = generate_data()

X = df[symptoms]
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

# Save model and symptoms list
joblib.dump(clf, "model.pkl")
joblib.dump(symptoms, "symptoms.pkl")
print("Model and symptoms saved!")
