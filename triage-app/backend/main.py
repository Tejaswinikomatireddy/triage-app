import os
import json
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="Rural Triage Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "triage_model.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
le = bundle["label_encoder"]
feature_cols = bundle["feature_cols"]
arrival_map = bundle["arrival_map"]

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


class PatientInput(BaseModel):
    # --- Core ML model features (trained on Kaggle Synthetic Medical Triage Priority Dataset) ---
    age: int
    heart_rate: int
    systolic_blood_pressure: int
    oxygen_saturation: int
    body_temperature: float
    pain_level: int = Field(ge=0, le=10)
    chronic_disease_count: int = 0
    previous_er_visits: int = 0
    arrival_mode: str = "walk_in"  # walk_in / wheelchair / ambulance

    # --- Additional context, used by the Safety Rule Engine + LLM layer only (not seen by the ML model) ---
    sex: str = "Male"
    pregnant: str = "No"
    shortness_of_breath: bool = False
    chest_pain: bool = False
    confusion: bool = False
    bleeding: bool = False
    seizure: bool = False
    weakness_paralysis: bool = False
    unable_full_sentences: bool = False
    loss_of_consciousness: bool = False
    pregnancy_complication: bool = False
    severe_bleeding: bool = False
    persistent_seizure: bool = False
    stroke_symptoms: bool = False
    severe_allergic_reaction: bool = False
    other_symptoms: str = ""  # free-text, passed to LLM only


def safety_rule_check(p: PatientInput):
    """
    Deterministic safety net, independent of the ML model. These signals are
    not present in the Kaggle training data, so the model cannot learn them --
    they're checked directly. Any match forces Critical regardless of the
    model's prediction, mirroring how real clinical decision-support systems
    layer hard rules on top of statistical models.
    """
    reasons = []
    if p.oxygen_saturation < 88:
        reasons.append("oxygen saturation critically low (SpO2 < 88%)")
    if p.loss_of_consciousness:
        reasons.append("loss of consciousness")
    if p.severe_bleeding:
        reasons.append("severe bleeding")
    if p.persistent_seizure or p.seizure:
        reasons.append("seizure activity")
    if p.stroke_symptoms:
        reasons.append("stroke symptoms present")
    if p.severe_allergic_reaction:
        reasons.append("severe allergic reaction")
    if p.unable_full_sentences and p.shortness_of_breath:
        reasons.append("severe breathing difficulty (cannot speak full sentences)")
    if p.systolic_blood_pressure < 80:
        reasons.append("dangerously low blood pressure (systolic < 80)")
    if p.pregnancy_complication:
        reasons.append("pregnancy complication")
    return (len(reasons) > 0, reasons)


def build_feature_vector(p: PatientInput):
    row = {
        "age": p.age,
        "heart_rate": p.heart_rate,
        "systolic_blood_pressure": p.systolic_blood_pressure,
        "oxygen_saturation": p.oxygen_saturation,
        "body_temperature": p.body_temperature,
        "pain_level": p.pain_level,
        "chronic_disease_count": p.chronic_disease_count,
        "previous_er_visits": p.previous_er_visits,
        "arrival_mode_enc": arrival_map.get(p.arrival_mode, 0),
    }
    return pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)


def rule_based_explanation(p: PatientInput, urgency: str, top_factors: list):
    """Fallback explanation generator if no LLM API key is configured."""
    factor_text = ", ".join(top_factors) if top_factors else "the overall vital signs and symptoms reported"
    actions = {
        "Critical": "Refer the patient immediately to the nearest hospital or call emergency services. Do not delay transport.",
        "Medium": "Advise the patient to visit the nearest Primary Health Centre (PHC) within 24 hours. Monitor for any worsening symptoms in the meantime.",
        "Low": "The patient can likely be monitored at home. Advise rest, fluids, and watching for any new or worsening symptoms.",
    }
    return (
        f"Based on {factor_text}, this patient has been assessed as {urgency} urgency. "
        f"{actions.get(urgency, 'Please use clinical judgement for next steps.')}"
    )


def llm_explanation(p: PatientInput, urgency: str, top_factors: list):
    if not ANTHROPIC_API_KEY:
        return rule_based_explanation(p, urgency, top_factors), False

    prompt = f"""You are helping a rural community health worker (ASHA) understand an AI triage result.
Do NOT diagnose a disease. Only explain the urgency level and give a clear next action.

Patient observations:
- Age: {p.age}
- Temperature: {p.body_temperature}°C, Heart rate: {p.heart_rate} bpm
- Blood pressure (systolic): {p.systolic_blood_pressure}, SpO2: {p.oxygen_saturation}%
- Pain level (0-10): {p.pain_level}, Chronic conditions: {p.chronic_disease_count}, Prior ER visits: {p.previous_er_visits}
- Key contributing factors: {', '.join(top_factors) if top_factors else 'general vitals'}

Predicted urgency level: {urgency}

Write a short (3-4 sentence) plain-language explanation for the health worker covering:
1. Why this urgency level was assigned (referencing the contributing factors)
2. The single recommended next action (home monitoring / PHC visit within 24h / immediate hospital referral)
Keep it simple, non-technical, and actionable. Do not mention you are an AI or a language model."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join([b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"])
        return text.strip(), True
    except Exception:
        return rule_based_explanation(p, urgency, top_factors), False


@app.get("/")
def root():
    return {"status": "ok", "service": "Rural Triage Assistant API", "model_trained_on": bundle.get("trained_on", "unknown")}


@app.post("/predict")
def predict(p: PatientInput):
    try:
        X = build_feature_vector(p)
        pred_idx = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        model_urgency = le.inverse_transform([pred_idx])[0]

        importances = dict(zip(feature_cols, model.feature_importances_))
        readable_map = {
            "pain_level": "high pain level", "body_temperature": "fever",
            "heart_rate": "abnormal heart rate", "oxygen_saturation": "low oxygen saturation",
            "age": "patient age", "systolic_blood_pressure": "abnormal blood pressure",
            "previous_er_visits": "history of prior ER visits", "chronic_disease_count": "chronic conditions",
            "arrival_mode_enc": "mode of arrival",
        }
        top_factors = sorted(feature_cols, key=lambda c: importances.get(c, 0), reverse=True)[:3]
        readable_factors = [readable_map.get(f, f.replace("_", " ")) for f in top_factors]
        if p.oxygen_saturation < 92:
            readable_factors.insert(0, "critically low oxygen saturation")

        rule_triggered, rule_reasons = safety_rule_check(p)
        if rule_triggered:
            urgency = "Critical"
            decision_source = "safety_rule"
            readable_factors = rule_reasons + [f for f in readable_factors if f not in rule_reasons]
        else:
            urgency = model_urgency
            decision_source = "ml_model"

        explanation, llm_used = llm_explanation(p, urgency, readable_factors)

        return {
            "urgency": urgency,
            "decision_source": decision_source,
            "model_prediction": model_urgency,
            "safety_rule_triggered": rule_triggered,
            "safety_rule_reasons": rule_reasons,
            "probabilities": {cls: round(float(prob), 3) for cls, prob in zip(le.classes_, proba)},
            "top_factors": readable_factors,
            "explanation": explanation,
            "llm_used": llm_used,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
