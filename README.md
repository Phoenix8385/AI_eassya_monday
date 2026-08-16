# AI Essay Detector — Callus Hiring Challenge (Project 2D)

A detector for AI-generated admissions essays that measures signals, not opinions — a trained classifier makes the call, not a chat model.

---

## 🧠 What It Does

- Pastes essay in → returns a probability + plain-language explanation, not a bare percentage
- Flags specific sentences with evidence, not just an overall score
- Never sends the essay to a chat model and relays its opinion — this is explicitly NOT a wrapper

---

## 🔬 How Detection Actually Works

- Runs the essay through a local **GPT-2/DistilGPT-2 model** to measure token-level predictability (perplexity)
- Measures **burstiness** — sentence-to-sentence rhythm variance
- Measures **vocabulary diversity**, **phrase repetition**, and **transition-word overuse**
- All 5 signals feed into **one trained classifier** — our own code decides, not the language model
- Output is always framed as a probability with a caveat, never "100% AI detected"

---

## 🧰 Tech Stack

- **Backend:** Python, FastAPI, HuggingFace Transformers, scikit-learn
- **Frontend:** React (TypeScript), Framer Motion, React Three Fiber
- **Deployment:** Render (backend) · Vercel (frontend)

---

## 🚀 Live Links

- **Frontend:** https://ai-eassya-monday.vercel.app
- **Backend:** https://ai-eassya-monday.onrender.com

---

## 📊 Dataset & Evaluation

- Base dataset: Kaggle "LLM - Detect AI Generated Text" (DAIGT)
- Hard-case subset: 40 human essays lightly AI-polished, labeled separately
- Full details in `backend/DATASET.md` and `backend/EVALUATION.md`

---

## ⚠️ Known Limitations

- Update this section with your final, current status before submitting
