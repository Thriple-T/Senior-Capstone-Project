# SayarDesk IELTS Intelligence 🎓

> AI-powered IELTS Writing Task 1 & 2 essay scorer — built as a senior capstone project at Parami University.

---

## Overview

**SayarDesk IELTS Intelligence** is a Django web application that automatically scores IELTS Writing Task 2 essays across all four official band criteria using a three-stage ML ensemble pipeline. Designed for English teachers in Myanmar, it provides instant, criterion-level feedback to support classroom assessment at scale.

---

## ML Pipeline

```
Essay Input
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1: BERT Multi-Task Model │  ← Fine-tuned bert-base-uncased
│         +  XGBoost Regressor    │  ← Hybrid weighted average
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Stage 2: Weighted Ensemble     │  ← Per-criterion performance weights
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Stage 3: Qwen3-VL-8B           │  ← LLM adjudicator via HuggingFace
│           LLM Adjudicator       │    Serverless Inference API
└─────────────────────────────────┘
    │
    ▼
Final Band Scores (TA · CC · LR · GRA)
```

---

## Scoring Criteria

| Criterion | Description |
|---|---|
| **Task Achievement (TA)** | How well the essay addresses the prompt |
| **Coherence & Cohesion (CC)** | Logical flow and use of linking devices |
| **Lexical Resource (LR)** | Range and accuracy of vocabulary |
| **Grammar Range & Accuracy (GRA)** | Grammatical variety and correctness |

---

## Model Performance (Held-Out Test Set)

| Criterion | MAE | RMSE | R² | Within ±0.5 Band |
|---|---|---|---|---|
| Task Achievement | 0.604 | 0.842 | 0.661 | 75.2% |
| Coherence & Cohesion | 0.687 | 0.876 | 0.652 | 62.4% |
| Lexical Resource | 0.798 | 1.123 | 0.551 | 55.6% |
| Grammar Range | 1.030 | 1.283 | 0.356 | 43.6% |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Django |
| BERT Model | `bert-base-uncased` (fine-tuned, HuggingFace) |
| Boosting Model | XGBoost |
| LLM Adjudicator | Qwen3-VL-8B (HuggingFace Serverless Inference API) |
| Dataset | 585 IELTS essays (augmented) |

---

## Dataset

- **Size:** 585 essays (pre-augmentation), expanded via semantic paraphrase augmentation
- **Augmentation quality:** Mean cosine similarity = 0.9763 (threshold: 0.90)
- **Score range:** Band 4–9 across all four criteria
- **Mean scores:** TA=7.03 · CC=6.84 · LR=6.65 · GRA=6.92

---

## Project Context

This project was developed as a **senior capstone thesis** at Parami University, at the intersection of applied NLP and English language education. It targets English teachers in Myanmar who need scalable, automated tools to assess student writing and identify support needs at the classroom level.

---

## Author

**Thomas** — Parami University, Class of 2026

---

*Built with 🤖 BERT · XGBoost · Qwen3 · Django*