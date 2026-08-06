"""
pipeline/extract_twin — ExtractTwinBuilder
--------------------------------------------
[HEURISTIC] 종/부위 조합에 대한 "virtual profile twin"을 구성하는 경량 모듈.
실제 LC-MS/GC-MS 실험 데이터가 아니라, lit_miner가 반환한 후보 화합물 목록을
화학 분류군(chemical class)으로 태깅하는 규칙 기반(rule-based) 로직임.

[근거 없음] chemical class 분류는 화합물명 문자열 매칭 규칙이며,
실제 구조 기반(SMARTS 등) 정밀 분류가 아님.
"""

MODULE_STATUS = "HEURISTIC"

_CLASS_RULES = [
    (("quercetin", "kaempferol", "rutin"), "Flavonoids"),
    (("curcumin",), "Curcuminoids"),
    (("ginsenoside",), "Triterpenoid Saponins"),
    (("cyanidin",), "Anthocyanins"),
    (("catechin", "gallate"), "Catechins"),
    (("justicidin",), "Lignans"),
    (("ginkgolide",), "Diterpene Lactones"),
]


def _classify(compound_name: str) -> str:
    n = compound_name.lower()
    for keys, cls in _CLASS_RULES:
        if any(k in n for k in keys):
            return cls
    return "Polyphenols"


class ExtractTwinBuilder:
    def __init__(self, *a, **kw):
        self.args, self.kwargs = a, kw

    def build(self, species: str, extract_part: str, compounds: list) -> dict:
        classes = [_classify(name) for name in compounds]
        return {
            "species": species,
            "extract_part": extract_part,
            "compound_names": list(compounds),
            "compound_classes": classes,
            "major_chemical_classes": sorted(set(classes)) or ["Polyphenols"],
        }
