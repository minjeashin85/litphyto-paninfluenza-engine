"""
FastAPI Backend (api/main.py)
-----------------------------
Serves REST API for LitPhyto-PanInfluenza Engine supporting binomial names, tissue parts, and Gemini LLM.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from pipeline.orchestrator import LitPhytoPanRNAEngine

app = FastAPI(
    title="LitPhyto-PanInfluenza Engine API",
    description="AI-Driven Plant Extract Lifecycle Profiling & Antiviral MOA Predictor",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = LitPhytoPanRNAEngine()


class ExtractPredictionRequest(BaseModel):
    query_resource: str = Field(..., example="Ginkgo biloba", description="Binomial scientific plant name (e.g., Ginkgo biloba)")
    target_virus: Optional[str] = Field("H1N1", example="H1N1", description="Target Influenza strain")
    extract_part: Optional[str] = Field("Leaves", example="Leaves", description="Plant tissue extract part")
    gemini_api_key: Optional[str] = Field(None, example="AIzaSy...", description="Optional Google Gemini API Key for LLM mining")


class CitationItem(BaseModel):
    title: str
    doi: str
    url: str
    evidence: Optional[str] = None


class LeadCompound(BaseModel):
    compound_name: str
    smiles: str
    chemical_classes: Optional[List[str]] = None
    ratio_estimate: Optional[float] = None
    tissue_source: Optional[str] = None
    h1n1_pa_binding_affinity_kcal_mol: float
    lifecycle_affinities: Optional[Dict[str, float]] = None
    pan_rna_host_target_affinity: Dict[str, float]
    citations: List[CitationItem]


class PerformanceMetrics(BaseModel):
    yield_estimate_pct: float
    binding_efficiency_index: float
    antiviral_potency_score: float
    selectivity_ratio: float


class VirtualProfileSummary(BaseModel):
    total_identified_compounds: int
    major_chemical_classes: List[str]


class DiscoveredMOA(BaseModel):
    moa_title: str
    synergy_score: float
    confidence_level: str
    broad_spectrum_potential: List[str]
    description: str


class ExtractPredictionResponse(BaseModel):
    query_resource: str
    extract_part: Optional[str] = "Leaves"
    target_virus: Optional[str] = "H1N1"
    status: str
    performance_metrics: Optional[PerformanceMetrics] = None
    virtual_profile_summary: VirtualProfileSummary
    predicted_leads: List[LeadCompound]
    discovered_moa: DiscoveredMOA


@app.get("/")
def read_root():
    return {
        "system": "LitPhyto-PanInfluenza Engine API",
        "version": "2.2.0",
        "status": "ONLINE",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/predict-extract", response_model=ExtractPredictionResponse)
def predict_extract(payload: ExtractPredictionRequest):
    try:
        if not payload.query_resource or not payload.query_resource.strip():
            raise HTTPException(status_code=400, detail="query_resource must not be empty.")

        target_v = payload.target_virus or "H1N1"
        part = payload.extract_part or "Leaves"
        key = payload.gemini_api_key
        result = engine.run_pipeline(payload.query_resource, target_v, part, key)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
