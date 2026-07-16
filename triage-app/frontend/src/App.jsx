import { useState } from "react";
import "./App.css";
import { LANGUAGES, T, SYMPTOM_LABELS, DANGER_LABELS, URGENCY_LABELS, FACTOR_TRANSLATIONS, ACTION_TEMPLATES } from "./translations";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Symptom checklist + danger flags feed the Safety Rule Engine & LLM layer only
// (the ML classifier itself is trained on the Kaggle vitals/pain dataset below)
const SYMPTOM_KEYS = ["shortness_of_breath", "chest_pain", "confusion", "bleeding", "seizure", "weakness_paralysis"];
const DANGER_KEYS = [
  "unable_full_sentences", "loss_of_consciousness", "pregnancy_complication",
  "severe_bleeding", "persistent_seizure", "stroke_symptoms", "severe_allergic_reaction",
];

const URGENCY_COLORS = { Low: "#2e7d32", Medium: "#e69500", Critical: "#c62828" };

function initialForm() {
  const base = {
    age: 30, sex: "Male", pregnant: "No",
    heart_rate: 80, systolic_blood_pressure: 120, oxygen_saturation: 97,
    body_temperature: 37.0, pain_level: 2, chronic_disease_count: 0,
    previous_er_visits: 0, arrival_mode: "walk_in",
  };
  [...SYMPTOM_KEYS, ...DANGER_KEYS].forEach((key) => { base[key] = false; });
  return base;
}

export default function App() {
  const [form, setForm] = useState(initialForm());
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lang, setLang] = useState("en");
  const t = T[lang];
  const symptomLabels = SYMPTOM_LABELS[lang];
  const dangerLabels = DANGER_LABELS[lang];

  const update = (key, value) => setForm((f) => ({ ...f, [key]: value }));
  // For number inputs: keep the field genuinely empty when cleared, instead of
  // silently becoming 0 (Number("") === 0). This lets the browser's native
  // `required` validation correctly block submission on a blank vital field,
  // rather than silently sending 0 for something like heart rate or SpO2.
  const numOrEmpty = (v) => (v === "" ? "" : Number(v));

  const translateFactor = (factor) => (lang === "en" ? factor : (FACTOR_TRANSLATIONS[lang]?.[factor] || factor));

  const translatedUrgency = (urgency) => URGENCY_LABELS[lang]?.[urgency] || urgency;

  const buildExplanation = (res) => {
    // If the LLM generated this text, we show it as-is (it's a free-form
    // sentence we can't reliably re-translate). Otherwise it's our own
    // rule-based template, which we can rebuild in the selected language
    // from the same factors + urgency the backend already returned.
    if (res.llm_used) return res.explanation;
    const translatedFactors = res.top_factors.map(translateFactor).join(lang === "en" ? ", " : ", ");
    const action = ACTION_TEMPLATES[lang]?.[res.urgency] || ACTION_TEMPLATES.en[res.urgency];
    if (lang === "en") {
      return `Based on ${translatedFactors}, this patient has been assessed as ${res.urgency} urgency. ${action}`;
    } else if (lang === "te") {
      return `${translatedFactors} ఆధారంగా, ఈ రోగిని ${translatedUrgency(res.urgency)} అత్యవసరతగా అంచనా వేయబడింది. ${action}`;
    } else {
      return `${translatedFactors} के आधार पर, इस रोगी को ${translatedUrgency(res.urgency)} तात्कालिकता के रूप में आंका गया है। ${action}`;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!res.ok) throw new Error(`Server error (${res.status})`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        setError(t.errorColdStart);
      } else if (err instanceof TypeError) {
        setError(t.errorConnection);
      } else {
        setError(err.message || t.errorGeneric);
      }
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setForm(initialForm());
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="lang-toggle">
          {Object.entries(LANGUAGES).map(([code, label]) => (
            <button key={code} type="button" className={`lang-btn ${lang === code ? "active" : ""}`}
              onClick={() => setLang(code)}>
              {label}
            </button>
          ))}
        </div>
        <h1>{t.title}</h1>
        <p>{t.subtitle}</p>
      </header>

      <div className="layout">
        <form className="card form-card" onSubmit={handleSubmit}>
          <h2>{t.patientAssessment}</h2>

          <div className="grid-2">
            <label>
              {t.age}
              <input type="number" onFocus={(e) => e.target.select()} min="0" max="120" value={form.age}
                onChange={(e) => update("age", numOrEmpty(e.target.value))} required />
            </label>
            <label>
              {t.sex}
              <select value={form.sex} onChange={(e) => update("sex", e.target.value)}>
                <option value="Male">{t.male}</option>
                <option value="Female">{t.female}</option>
              </select>
            </label>
          </div>

          {form.sex === "Female" && (
            <label>
              {t.pregnant}
              <select value={form.pregnant} onChange={(e) => update("pregnant", e.target.value)}>
                <option value="No">{t.no}</option>
                <option value="Yes">{t.yes}</option>
              </select>
            </label>
          )}

          <h3>{t.vitals}</h3>
          <div className="grid-2">
            <label>
              {t.temperature}
              <input type="number" onFocus={(e) => e.target.select()} step="0.1" value={form.body_temperature}
                onChange={(e) => update("body_temperature", numOrEmpty(e.target.value))} required />
            </label>
            <label>
              {t.heartRate}
              <input type="number" onFocus={(e) => e.target.select()} value={form.heart_rate}
                onChange={(e) => update("heart_rate", numOrEmpty(e.target.value))} required />
            </label>
            <label>
              {t.systolicBp}
              <input type="number" onFocus={(e) => e.target.select()} value={form.systolic_blood_pressure}
                onChange={(e) => update("systolic_blood_pressure", numOrEmpty(e.target.value))} required />
            </label>
            <label>
              {t.spo2}
              <input type="number" onFocus={(e) => e.target.select()} min="0" max="100" value={form.oxygen_saturation}
                onChange={(e) => update("oxygen_saturation", numOrEmpty(e.target.value))} required />
            </label>
          </div>

          <h3>{t.clinicalHistory}</h3>
          <div className="grid-2">
            <label>
              {t.painLevel}
              <input type="range" min="0" max="10" value={form.pain_level}
                onChange={(e) => update("pain_level", Number(e.target.value))} />
              <span className="range-value">{form.pain_level} / 10</span>
            </label>
            <label>
              {t.arrivalMode}
              <select value={form.arrival_mode} onChange={(e) => update("arrival_mode", e.target.value)}>
                <option value="walk_in">{t.walkIn}</option>
                <option value="wheelchair">{t.wheelchair}</option>
                <option value="ambulance">{t.ambulance}</option>
              </select>
            </label>
            <label>
              {t.chronicCount}
              <input type="number" onFocus={(e) => e.target.select()} min="0" value={form.chronic_disease_count}
                onChange={(e) => update("chronic_disease_count", Number(e.target.value))} />
            </label>
            <label>
              {t.priorErVisits}
              <input type="number" onFocus={(e) => e.target.select()} min="0" value={form.previous_er_visits}
                onChange={(e) => update("previous_er_visits", Number(e.target.value))} />
            </label>
          </div>

          <h3>{t.symptoms}</h3>
          <div className="checkbox-grid">
            {SYMPTOM_KEYS.map((key) => (
              <label key={key} className="checkbox">
                <input type="checkbox" checked={form[key]}
                  onChange={(e) => update(key, e.target.checked)} />
                {symptomLabels[key]}
              </label>
            ))}
          </div>

          <h3 className="danger-heading">{t.dangerFlags}</h3>
          <div className="checkbox-grid danger">
            {DANGER_KEYS.map((key) => (
              <label key={key} className="checkbox">
                <input type="checkbox" checked={form[key]}
                  onChange={(e) => update(key, e.target.checked)} />
                {dangerLabels[key]}
              </label>
            ))}
          </div>

          <div className="actions">
            <button type="submit" disabled={loading}>{loading ? t.assessing : t.submit}</button>
            <button type="button" className="secondary" onClick={reset}>{t.reset}</button>
          </div>
        </form>

        <div className="card result-card">
          <h2>{t.resultTitle}</h2>
          {!result && !error && !loading && <p className="placeholder">{t.placeholder}</p>}
          {loading && <p className="placeholder">{t.running}<br /><span className="cold-start-note">{t.coldStartNote}</span></p>}
          {error && <p className="error">{error}</p>}
          {result && (
            <div>
              <div className="urgency-badge" style={{ background: URGENCY_COLORS[result.urgency] }}>
                {translatedUrgency(result.urgency)} {t.urgencySuffix}
              </div>

              <div className="decision-source">
                {result.decision_source === "safety_rule" ? (
                  <span className="badge badge-rule">
                    ⚠️ {t.safetyOverride} — {result.safety_rule_reasons.map(translateFactor).join(", ")}
                    {result.model_prediction !== result.urgency && (
                      <> ({t.modelAloneSaid} {translatedUrgency(result.model_prediction)})</>
                    )}
                  </span>
                ) : (
                  <span className="badge badge-model">🤖 {t.mlModelPrediction}</span>
                )}
              </div>

              <div className="probs">
                {Object.entries(result.probabilities).map(([cls, prob]) => (
                  <div key={cls} className="prob-row">
                    <span>{translatedUrgency(cls)}</span>
                    <div className="prob-bar-bg">
                      <div className="prob-bar" style={{ width: `${prob * 100}%`, background: URGENCY_COLORS[cls] }} />
                    </div>
                    <span>{Math.round(prob * 100)}%</span>
                  </div>
                ))}
              </div>

              {result.top_factors?.length > 0 && (
                <div className="factors">
                  <strong>{t.keyFactors}</strong>
                  <ul>{result.top_factors.map((f) => <li key={f}>{translateFactor(f)}</li>)}</ul>
                </div>
              )}

              <div className="explanation">
                <strong>{t.guidance}</strong>
                <p>{buildExplanation(result)}</p>
              </div>

              <div className="connectivity-note">
                {result.llm_used ? `📶 ${t.onlineGuidance}` : `📴 ${t.offlineGuidance}`}
              </div>

              <p className="disclaimer">{t.disclaimer}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
