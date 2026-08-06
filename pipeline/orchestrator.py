"""
pipeline/orchestrator — LitPhytoPanRNAEngine
-----------------------------------------------
[HEURISTIC 인프로세스 엔진]

원래 설계는 app.py가 FastAPI 백엔드(http://localhost:8009)에 GNN 도킹 +
causal MOA 추론을 위임하는 구조였으나, 백엔드 소스가 제공되지 않아 항상
"Backend 연결 실패"로 끝났음.

이 모듈은 별도 서버 없이 Streamlit 프로세스 안에서 바로 완결되는 결정론적
(deterministic) 휴리스틱 파이프라인임. 동일 (종, 표적 바이러스, 추출 부위)
입력이면 항상 동일한 결과가 나오도록 시드를 고정함 — 재현성 확보 목적.

[근거 없음] 최종적으로 반환되는 결합 에너지(kcal/mol), Bliss synergy score,
antiviral potency score, yield/selectivity 지표는 실제 3D GNN 도킹이나
세포실험 결과가 아니라 입력값 해시 기반으로 생성된 placeholder 수치임.
화합물명 자체(Quercetin, Curcumin 등)는 miners/lit_miner.py의 SPECIES_COMPOUND_DB에
있는 해당 식물의 실제 보고된 phytochemical이지만, 그 화합물이 지정된 바이러스
표적에 실제로 결합한다는 실험적 근거는 본 파이프라인에 포함되어 있지 않음.

실제 GNN 도킹/문헌 마이닝 백엔드가 준비되면 이 클래스의 run() 본문만
해당 백엔드 호출로 교체하면 됨 — app.py 쪽 인터페이스(반환 dict 스키마)는
그대로 유지하면 됨.
"""

import miners.lit_miner as lit_miner
import pipeline.extract_twin as extract_twin
import models.gnn_predictor as gnn_predictor
import models.causal_moa as causal_moa

MODULE_STATUS = "HEURISTIC"


class LitPhytoPanRNAEngine:
    """[HEURISTIC] 외부 API/서버 없이 단일 프로세스 안에서 완결되는 예측 파이프라인."""

    def __init__(self, use_live_api: bool = False, **kw):
        self.use_live_api = use_live_api
        self.kwargs = kw
        self.miner = lit_miner.LitMiner()
        self.twin_builder = extract_twin.ExtractTwinBuilder()
        self.gnn = gnn_predictor.GNNPredictor()
        self.causal = causal_moa.CausalMOAEngine()

    def run(self, query_resource: str, target_virus: str = "H1N1",
            extract_part: str = "Leaves", gemini_api_key: str = None) -> dict:
        query_resource = (query_resource or "").strip() or "Unknown species"
        target_virus = (target_virus or "H1N1").strip()
        extract_part = (extract_part or "Leaves").strip()

        # Stage 1: Literature/compound mining
        compound_names = self.miner.mine(query_resource)

        # Stage 2: Virtual extract profile twin
        twin = self.twin_builder.build(query_resource, extract_part, compound_names)

        # Stage 3: Per-compound binding affinity ("GNN docking" placeholder)
        leads = []
        for idx, name in enumerate(compound_names):
            affinity = self.gnn.predict(name, target_virus, rank_idx=idx)
            leads.append(self._build_lead(name, affinity, idx, twin, extract_part))
        leads.sort(key=lambda l: l["h1n1_pa_binding_affinity_kcal_mol"])

        # Stage 4: MOA inference
        moa = self.causal.infer(query_resource, target_virus, leads)

        perf = self._build_performance_metrics(leads, query_resource, target_virus)
        summary = {
            "major_chemical_classes": twin["major_chemical_classes"],
            "compound_count": len(leads),
        }

        return {
            "query_resource": query_resource,
            "target_virus": target_virus,
            "extract_part": extract_part,
            "virtual_profile_summary": summary,
            "predicted_leads": leads,
            "discovered_moa": moa,
            "performance_metrics": perf,
        }

    def _build_lead(self, name: str, affinity: float, idx: int, twin: dict, extract_part: str) -> dict:
        import hashlib
        import random

        rng = random.Random(int(hashlib.sha256(f"{name}|{extract_part}|lead".encode()).hexdigest()[:16], 16))
        smiles = lit_miner.KNOWN_SMILES.get(name.split("(")[0].strip().lower(), "")
        chem_class = twin["compound_classes"][idx] if idx < len(twin["compound_classes"]) else "Polyphenols"

        return {
            "compound_name": name,
            "h1n1_pa_binding_affinity_kcal_mol": affinity,
            "chemical_classes": [chem_class],
            "tissue_source": f"{extract_part} extract",
            "citations": [],  # 비어있으면 app.py가 lit_miner.REAL_CITATION_DB로 자동 대체함
            "compound_id": "",
            "smiles": smiles,
            "ratio_estimate": round(0.05 + rng.random() * 0.30, 3),
            "scores": {
                "s_viral": round(0.50 + rng.random() * 0.45, 2),
                "s_host": round(0.05 + rng.random() * 0.25, 2),
            },
        }

    def _build_performance_metrics(self, leads: list, query_resource: str, target_virus: str) -> dict:
        import hashlib
        import random

        rng = random.Random(int(hashlib.sha256(f"{query_resource}|{target_virus}|perf".encode()).hexdigest()[:16], 16))
        best_aff = min((l["h1n1_pa_binding_affinity_kcal_mol"] for l in leads), default=-6.0)
        potency = round(min(99.0, max(50.0, (abs(best_aff) / 15.8) * 100.0)), 1)

        return {
            "yield_estimate_pct": round(1.0 + rng.random() * 4.0, 2),
            "binding_efficiency_index": round(abs(best_aff) * 1.8 + rng.random() * 2, 1),
            "antiviral_potency_score": potency,
            "selectivity_ratio": round(2.0 + rng.random() * 6.0, 1),
        }
