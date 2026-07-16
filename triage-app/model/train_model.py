"""
Retrains the triage urgency classifier on the real, publicly available
Kaggle "Synthetic Medical Triage Priority Dataset"
(emirhanakku/synthetic-medical-triage-priority-dataset).

Original dataset has a 4-level triage_level (0=least urgent ... 3=most urgent).
We map this to our 3-tier Low/Medium/Critical scheme used throughout the app:
    0 -> Low
    1 -> Medium
    2, 3 -> Critical   (levels 2 and 3 both represent urgent/emergent cases
                         requiring prompt escalation; merging them keeps the
                         output actionable for a non-physician health worker
                         choosing between "monitor / PHC visit / hospital now")

Note: this dataset does not include a symptom checklist or explicit danger
flags. Those remain part of the app's Safety Rule Engine (deterministic,
not learned) and the LLM explanation layer -- they are NOT fed into this
classifier, since the training data has no ground truth for them.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("/home/claude/triage-app/data/triage_dataset.csv")

def map_urgency(level):
    if level == 0:
        return "Low"
    elif level == 1:
        return "Medium"
    else:
        return "Critical"

df["urgency"] = df["triage_level"].apply(map_urgency)

arrival_map = {"walk_in": 0, "wheelchair": 1, "ambulance": 2}
df["arrival_mode_enc"] = df["arrival_mode"].map(arrival_map)

feature_cols = [
    "age", "heart_rate", "systolic_blood_pressure", "oxygen_saturation",
    "body_temperature", "pain_level", "chronic_disease_count",
    "previous_er_visits", "arrival_mode_enc",
]

X = df[feature_cols]
y = df["urgency"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=14, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion matrix (rows=true, cols=pred):", list(le.classes_))
print(confusion_matrix(y_test, y_pred))

importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

joblib.dump({
    "model": clf,
    "label_encoder": le,
    "feature_cols": feature_cols,
    "arrival_map": arrival_map,
    "trained_on": "kaggle_synthetic_medical_triage_priority_dataset",
}, "/home/claude/triage-app/model/triage_model.joblib")

print("\nModel saved to model/triage_model.joblib")
