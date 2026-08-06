"""
models/causal_moa — CausalMOAEngine
-------------------------------------
[HEURISTIC] MOA(Mechanism of Action) 서술은 리드 화합물들의 결합 에너지
패턴을 바탕으로 자동 조합된 요약 문장이며, 실제 causal inference 알고리즘이나
세포/동물 실험으로 규명된 MOA가 아님.

[근거 없음] synergy_score / confidence_level / broad_spectrum_potential은
전부 결정론적 난수 기반 placeholder임.
"""

import hashlib
import random

MODULE_STATUS = "HEURISTIC"


def _seeded_rng(*parts) -> random.Random:
    key = "|".join(str(p) for p in parts).lower().strip()
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


class CausalMOAEngine:
    def __init__(self, *a, **kw):
        self.args, self.kwargs = a, kw

    def infer(self, query_resource: str, target_virus: str, leads: list) -> dict:
        rng = _seeded_rng(query_resource, target_virus, "moa")
        top = leads[0] if leads else {}
        top_name = top.get("compound_name", "Lead compound")
        n = len(leads)

        synergy = round(0.55 + rng.random() * 0.35, 2)
        confidence = "High" if synergy >= 0.75 else ("Moderate" if synergy >= 0.60 else "Low")

        description = (
            f"{query_resource} 유래 후보 화합물 {n}종 가운데 {top_name}이(가) "
            f"가장 강한 결합 에너지를 나타냄. 다중 표적(PA endonuclease, HA, NA 등) "
            f"결합 패턴을 종합하면, 단일 표적보다 다중 표적 동시 억제를 통한 항바이러스 "
            f"효과가 우세할 것으로 휴리스틱 모델이 추정함. "
            f"(실험적으로 검증된 결론 아님, [근거 없음])"
        )

        return {
            "moa_title": f"{target_virus} 다중표적 동시 억제 가설 (Multi-target Inhibition Hypothesis)",
            "description": description,
            "synergy_score": synergy,
            "confidence_level": confidence,
            "broad_spectrum_potential": (
                ["Influenza A", "Influenza B"] if synergy >= 0.70 else ["Influenza A"]
            ),
        }
