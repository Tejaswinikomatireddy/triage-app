# 🏥 Rural AI Triage Assistant

**An offline-first AI-powered triage assistant that helps ASHA/community health workers prioritize patients when a doctor isn't immediately available.**

Built for Idea2Impact 2026 — Theme 3: Crisis Management, HealthTech & Emergency Response.

## Live Demo
- **Frontend:** https://triage-app-two.vercel.app
- **Backend API:** https://rural-triage-assistant.onrender.com
- **Demo Video:** https://drive.google.com/file/d/1aGD9eaTY5mj15pAR50SAfa_kPSSqcySG/view?usp=sharing

## Key Features
✅ Random Forest triage model — trained on a real public dataset, 94% accuracy, 95% recall on Critical cases
✅ Safety Rule Engine — deterministic override for danger signs the ML model was never trained on
✅ Explainable AI — every result shows *why* and *which system* (model vs. safety rule) made the call
✅ Offline-first — full guidance available even with zero internet, no functionality lost
✅ Multilingual — English / Telugu (తెలుగు) / Hindi (हिन्दी)
✅ Not a diagnostic tool by design — estimates urgency only, avoiding the medico-legal risk of AI disease diagnosis

## Try It in 30 Seconds
1. Open the live frontend link above
2. Enter any vitals (defaults are pre-filled) → click **Get Triage Assessment**
3. Check a danger flag (e.g. "Severe bleeding") with otherwise mild vitals → watch the Safety Rule Engine override the ML model to Critical
4. Switch the language toggle at the top → all text updates instantly

---

## The Problem

Community health workers — ASHAs and similar frontline staff — are often the first point of contact for patients in rural India. They frequently work without a doctor nearby, without reliable internet, and without formal clinical training in triage. Deciding *"does this patient need a hospital right now, or can they wait?"* is a high-stakes judgment call made under time pressure with limited support.

Most existing digital health tools assume constant connectivity, real-time doctor access, or attempt disease diagnosis from symptoms — which carries real medico-legal risk and doesn't match what a health worker actually needs in the field.

**This project doesn't diagnose disease. It estimates urgency** — Low / Medium / Critical — and tells the health worker what to do next.

---

## Architecture
```
┌─────────────────────────┐
Patient vitals +  │   Random Forest          │   Urgency: Low / Medium /
symptoms/flags ──▶│   Classifier              │──▶   Critical (model's view)
│   (trained on real data)  │
└─────────────────────────┘
│
▼
┌─────────────────────────┐
Danger flags ────▶│   Safety Rule Engine      │──▶  Overrides to Critical
(bleeding, LOC,    │   (deterministic)         │      if any red flag present,
SpO2<88%, etc.)    └─────────────────────────┘      regardless of ML output
│
▼
┌─────────────────────────┐
│   LLM Explanation Layer   │──▶  Plain-language guidance
│   (optional, has offline   │      for the health worker
│    fallback)               │
└─────────────────────────┘
```

**Why this design?** Healthcare decision-support systems in practice rarely trust a single ML model end-to-end. Splitting responsibilities this way means:

- The **ML model** does what it's good at — finding statistical patterns in structured vitals data — and is only trusted within the scope of what it was actually trained on.
- The **Safety Rule Engine** catches clinically obvious red flags (severe bleeding, loss of consciousness, SpO2 < 88%, stroke symptoms, etc.) that the training data doesn't represent well, and can never be silently overridden by the model.
- The **LLM** never makes the clinical call — it only translates a decision that's already been made into language a non-specialist can act on immediately. This keeps the "AI" in the system auditable and reduces hallucination risk.

---

## The Dataset & Model

- **Dataset:** [Synthetic Medical Triage Priority Dataset](https://www.kaggle.com/datasets/emirhanakku/synthetic-medical-triage-priority-dataset) (Kaggle), 18,000 records with vitals, pain level, comorbidity count, prior ER visits, arrival mode, and a 4-level triage label.
- **Label mapping:** the dataset's 4 triage levels (0–3, least to most urgent) are collapsed into the app's 3-tier scheme: `0 → Low`, `1 → Medium`, `2 and 3 → Critical`. Levels 2 and 3 are merged because both require prompt escalation from a non-physician's perspective — the actionable choice for a health worker is really "wait" vs. "go now."
- **Model:** Random Forest classifier (scikit-learn), 300 trees, class-balanced.
- **Note on scope:** the classifier is trained only on the features the dataset actually contains (vitals, pain, comorbidity burden, arrival mode). It does **not** see the symptom checklist or danger flags collected in the UI — those aren't present in the training data, so we don't pretend the model learned them. They're handled separately by the Safety Rule Engine.

### Evaluation

On a held-out 20% test split (3,600 patients):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Critical | 0.97 | 0.95 | 0.96 |
| Medium | 0.90 | 0.88 | 0.89 |
| Low | 0.95 | 0.97 | 0.96 |
| **Overall accuracy** | | | **0.94** |

**On the accuracy number:** 94% overall accuracy is a strong result *on this public dataset*, but this project is a proof-of-concept demonstrating AI-assisted triage — not a clinically validated device. The dataset is synthetic (Kaggle-published, not sourced from real hospital records), so real-world performance would need validation on genuine clinical data before any real deployment. What we think matters more than the headline number:

- **Recall on Critical cases is 95%** — of 718 true-critical patients in the test set, only 2 were missed to "Low" (the other 37 misclassifications landed on "Medium," a much safer miss).
- Those 2 near-misses are exactly the kind of rare edge case the **Safety Rule Engine** exists to catch independently of the model — a patient with severe bleeding or SpO2 < 88% is flagged Critical regardless of what the classifier says.
- Confusion matrix (rows = actual, columns = predicted; order Critical/Low/Medium):
Critical   Low   Medium
Critical         679      2      37
Low                0   1931      54
Medium            23     89     785
Retraining code: [`model/train_model.py`](model/train_model.py). Dataset: [`data/triage_dataset.csv`](data/triage_dataset.csv).

---

## Design Decisions Behind Each Feature

- ✅ **Safety Rule Engine** — deterministic overrides for danger signs the ML model was never trained on
- ✅ **Explainable decisions** — every prediction shows whether it came from the model or a safety rule, plus the model's own confidence breakdown
- ✅ **Offline-first guidance** — if no LLM API key is configured (or the health worker has no internet), the system still returns full guidance using a built-in rule-based explanation generator. This isn't a fallback bolted on for the demo — it's a first-class mode, since unreliable connectivity is central to the problem this app addresses.
- ✅ **Multilingual UI** — English, Telugu (తెలుగు), and Hindi (हिन्दी), covering all form fields and results, not just static page text
- ✅ **Not a diagnostic tool** — deliberately scoped to urgency estimation only, to avoid the medico-legal and safety risks of AI disease diagnosis

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML model | Python, scikit-learn (Random Forest) |
| Backend API | FastAPI |
| Frontend | React (Vite) |
| LLM layer | Claude API (optional; graceful offline fallback) |
| Backend hosting | Render |
| Frontend hosting | Vercel |

---

## Project Structure
```
backend/            FastAPI service: model inference, safety rules, LLM layer
main.py
requirements.txt
frontend/            React app (Vite)
src/
App.jsx
App.css
translations.js  English / Telugu / Hindi strings
model/
train_model.py     Retraining script
triage_model.joblib
data/
triage_dataset.csv  Kaggle Synthetic Medical Triage Priority Dataset
assets/
screenshots/
Procfile              Render start command
requirements.txt      Root-level, mirrors backend/requirements.txt for Render auto-detect
.env.example
```

---

## Running Locally

**Backend:**
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

**Frontend** (separate terminal):
```bash
cd frontend
npm install
npm run dev
```

The frontend defaults to `http://localhost:8000` for the API; override with a `.env` file (`VITE_API_URL=...`) if needed — see `.env.example`.

**Optional — enable LLM explanations:**
```bash
export ANTHROPIC_API_KEY=your_key_here
```
Without this, the app automatically uses its offline rule-based explanation generator — no functionality is lost, only the phrasing becomes templated instead of model-generated.

---

## Deployment

- **Backend (Render):** root directory blank, build command `pip install -r requirements.txt`, start command `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
- **Frontend (Vercel):** root directory `frontend`, framework preset Vite, environment variable `VITE_API_URL` set to the deployed backend URL.

---

## Screenshots

*(see [`assets/screenshots/`](assets/screenshots/))*

---

## Limitations & Future Work

- Trained on a **synthetic public dataset**, not real hospital records — the natural next step is validating (and likely retraining) on de-identified real triage data, ideally with clinician input on the label scheme.
- The Safety Rule Engine's thresholds are a reasonable clinical starting point but are not clinician-reviewed — a real deployment would need sign-off from a medical professional.
- Currently single-patient, single-session — no persistent patient history across visits.
- Potential extensions: voice input/output for lower-literacy contexts, a nearby PHC/hospital locator, printable referral summaries, and offline-first PWA packaging so the frontend itself works with zero connectivity (currently only the guidance-generation step degrades gracefully offline; the initial page load still needs one).

---

## Disclaimer

This is a hackathon prototype intended to demonstrate the feasibility of AI-assisted triage. It is **not a certified medical device** and does not replace clinical judgment. It estimates urgency, not diagnosis, and its outputs should always be treated as a decision-support aid rather than a final medical decision.