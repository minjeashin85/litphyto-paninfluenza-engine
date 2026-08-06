# LitPhyto-PanRNA Engine

**AI-Driven Virtual Natural Extract Digital Twin & Pan-RNA Antiviral Lead Prediction Platform**

The **LitPhyto-PanRNA Engine** is an advanced computational biology and drug discovery backend system designed to mine natural product literature, construct virtual extract digital twins, run PyTorch Geometric Graph Neural Networks (GNN) for binding energy ($\Delta G_{bind}$) prediction against H1N1 and broad-spectrum RNA virus host targets, and compute Causal Mechanism of Action (MOA) synergy models.

---

## 🏗️ System Architecture

```
[Input: Plant/Resource Name]
        │
        ▼
[Module 1: Lit-Chem Mining Engine] ──(Structured Data + Citations)
        │
        ▼
[Module 2: Virtual Extract Profile Twin] ──(SMILES, Concentrations, 3D Graph)
        │
        ▼
[Module 3: Pan-RNA Host-Pathogen GNN] ──(Binding Energies ΔG, Target Scores)
        │
        ▼
[Module 4: Causal MOA & Synergy Engine] ──(Synergy Scores, Novel MOA Hypothesis)
```

---

## 📁 Modular Code Structure

```
litphyto_panrna_engine/
├── miners/
│   ├── __init__.py
│   └── lit_miner.py           # Module 1: Lit-Chem Mining Engine (PubChem, ChEMBL, PubMed)
├── pipeline/
│   ├── __init__.py
│   ├── extract_twin.py        # Module 2: Virtual Extract Profile Twin (RDKit taxonomy, ETKDG 3D, PyG)
│   └── orchestrator.py        # Main Pipeline Orchestrator
├── models/
│   ├── __init__.py
│   ├── gnn_predictor.py       # Module 3: Pan-RNA Host-Pathogen GNN Predictor (ΔG_bind, Target scores)
│   └── causal_moa.py          # Module 4: Causal MOA & Bliss Synergy Engine (NetworkX Bipartite Graphs)
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI backend serving POST /api/v1/predict-extract
├── app.py                     # Streamlit web application (Interactive UI & GitHub Cloud Ready)
├── requirements.txt           # Dependency specifications
└── tests/
    └── test_engine.py         # Test suite
```

---

## 🚀 Quick Start Guide

### 1. Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running local Streamlit Web UI

To launch the interactive web application locally:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your web browser.

### 3. Running local FastAPI Backend

To start the FastAPI REST server:
```bash
uvicorn api.main:app --reload --port 8000
```
Open the interactive API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

#### Endpoint Usage Example:
- **POST** `/api/v1/predict-extract`
- **Request Body**:
```json
{
  "query_resource": "Ginkgo biloba"
}
```

---

## 🌐 Public Deployment (GitHub → Streamlit Community Cloud)

1. Push this folder to your GitHub repository.
2. Log into [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **New App**, select your GitHub repository, set the main file path to `app.py`.
4. Click **Deploy!** Your public live URL will be generated instantly.
