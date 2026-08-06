"""
models/gnn_predictor — GNNPredictor
-------------------------------------
[HEURISTIC] 실제 3D conformer GNN 도킹 모델이 아니라, (화합물명, 표적 바이러스)
조합을 시드로 한 결정론적(deterministic) 난수로 결합 에너지(kcal/mol)를 생성하는
placeholder 모듈임. 동일 입력이면 항상 동일한 값을 반환함(재현성 보장).

[근거 없음] 반환값은 실험적으로도 전산화학적으로도 검증되지 않은 수치임.
"""

import hashlib
import random

MODULE_STATUS = "HEURISTIC"

# 실제 인플루엔자 억제제(Baloxavir marboxil 등)의 공개된 결합 에너지 범위를
# 참고 삼아 물리적으로 그럴듯한 구간(-6.0 ~ -11.0 kcal/mol)으로만 값을 제한함.
# 이 구간 자체가 본 화합물의 실제 결합력을 의미하지는 않음.
_MIN_AFFINITY = -11.0
_MAX_AFFINITY = -6.0


def _seeded_rng(*parts) -> random.Random:
    key = "|".join(str(p) for p in parts).lower().strip()
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


class GNNPredictor:
    def __init__(self, *a, **kw):
        self.args, self.kwargs = a, kw

    def predict(self, compound_name: str, target_virus: str, rank_idx: int = 0) -> float:
        rng = _seeded_rng(compound_name, target_virus)
        span = _MAX_AFFINITY - _MIN_AFFINITY
        base = _MAX_AFFINITY - rng.random() * span
        # 동률 방지 + 상위 랭크일수록 근소하게 더 강하게 표시
        return round(base - (rank_idx * 0.3), 1)
