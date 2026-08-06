"""
miners/lit_miner — LitMiner
----------------------------
[HEURISTIC 엔진] 실시간 PubMed/PubChem API 호출 없이 로컬 DB로 동작하는
경량 문헌/화합물 마이닝 모듈.

- 종(species) → 후보 phytochemical 목록 매핑은 해당 식물에서 실제로 널리
  보고된 대표 성분(예: Curcuma longa -> Curcumin)을 사용함. 일반적으로
  통용되는 식물화학 상식 수준의 정보임.
- REAL_CITATION_DB는 app.py가 화합물별 개별 인용을 찾지 못했을 때 쓰는
  범용 폴백(fallback)이며, 개별 논문을 검증해서 긁어온 것이 아니라 PubMed
  검색 결과로 연결되는 실제 작동 링크임.

[근거 없음] 화합물명 자체는 실제 phytochemical이 맞으나, 본 모듈이
반환하는 결합 에너지/효능 수치/MOA 서술은 이 파일이 아니라
pipeline/orchestrator.py, models/gnn_predictor.py, models/causal_moa.py에서
결정론적 난수로 생성되며 전부 미검증 placeholder임.
"""

MODULE_STATUS = "HEURISTIC"

# 종(학명/영문명/국문명) -> 대표 phytochemical 후보 목록
SPECIES_COMPOUND_DB = {
    "ginkgo": ["Ginkgolide B", "Quercetin", "Kaempferol"],
    "은행": ["Ginkgolide B", "Quercetin", "Kaempferol"],
    "panax": ["Ginsenoside Rg1", "Quercetin", "Kaempferol"],
    "ginseng": ["Ginsenoside Rg1", "Quercetin", "Kaempferol"],
    "인삼": ["Ginsenoside Rg1", "Quercetin", "Kaempferol"],
    "sambucus": ["Cyanidin-3-glucoside", "Quercetin", "Rutin"],
    "elderberry": ["Cyanidin-3-glucoside", "Quercetin", "Rutin"],
    "엘더베리": ["Cyanidin-3-glucoside", "Quercetin", "Rutin"],
    "curcuma": ["Curcumin", "Demethoxycurcumin", "Quercetin"],
    "turmeric": ["Curcumin", "Demethoxycurcumin", "Quercetin"],
    "강황": ["Curcumin", "Demethoxycurcumin", "Quercetin"],
    "울금": ["Curcumin", "Demethoxycurcumin", "Quercetin"],
    "justicia": ["Justicidin A", "Quercetin", "Kaempferol"],
    "procumbens": ["Justicidin A", "Quercetin", "Kaempferol"],
    "쥐꼬리망초": ["Justicidin A", "Quercetin", "Kaempferol"],
    "camellia": ["Epigallocatechin gallate (EGCG)", "Quercetin", "Kaempferol"],
    "sinensis": ["Epigallocatechin gallate (EGCG)", "Quercetin", "Kaempferol"],
    "녹차": ["Epigallocatechin gallate (EGCG)", "Quercetin", "Kaempferol"],
}
DEFAULT_COMPOUNDS = ["Quercetin", "Kaempferol", "Rutin"]

# app.py의 KNOWN_COMPOUND_CIDS / OFFICIAL_COMPOUND_IMAGES / WIKIPEDIA_COMPOUND_URLS 와
# 이름을 맞춰서, 이미 존재하는 실제 PubChem CID 기반 2D 구조 이미지 및 Wikipedia
# 링크가 자동으로 붙도록 함. 표에 없는 화합물은 SMILES를 비워 두고
# app.py의 이름 기반 PubChem PUG REST 폴백(get_official_wiki_pubchem_image_url)이
# 처리하도록 위임함.
KNOWN_SMILES = {
    "quercetin": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
    "kaempferol": "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
    "curcumin": "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",
}

# app.py 2172번대 UI가 "검증 학술 논문 레퍼런스"라고 표기하는 부분에 붙는 범용 폴백.
# 개별 논문을 검증한 것이 아니라 PubMed 검색 결과로 연결되는 실동작 링크임을
# evidence 필드에 명시함.
REAL_CITATION_DB = [
    {
        "title": "Literature search: plant-derived phytochemicals with antiviral activity against Influenza",
        "journal": "PubMed (검색 링크, 개별 논문 미검증)",
        "pmid": "SEARCH",
        "doi": None,
        "url": "https://pubmed.ncbi.nlm.nih.gov/?term=plant+extract+antiviral+influenza",
        "evidence": "PubMed 실시간 검색 결과 링크임. 자동 생성된 폴백이며 개별 논문 인용은 검증되지 않음. [근거 없음]",
        "assay_metric": "N/A",
    },
    {
        "title": "Literature search: influenza PA endonuclease natural inhibitors",
        "journal": "PubMed (검색 링크, 개별 논문 미검증)",
        "pmid": "SEARCH",
        "doi": None,
        "url": "https://pubmed.ncbi.nlm.nih.gov/?term=influenza+PA+endonuclease+natural+inhibitor",
        "evidence": "PubMed 실시간 검색 결과 링크임. 자동 생성된 폴백이며 개별 논문 인용은 검증되지 않음. [근거 없음]",
        "assay_metric": "N/A",
    },
]


class LitMiner:
    """[HEURISTIC] 종 이름으로 후보 phytochemical 목록을 찾는 로컬 DB 조회기."""

    def __init__(self, *a, **kw):
        self.args, self.kwargs = a, kw

    def mine(self, query_resource: str) -> list:
        q_low = (query_resource or "").lower()
        for key, compounds in SPECIES_COMPOUND_DB.items():
            if key in q_low:
                return list(compounds)
        return list(DEFAULT_COMPOUNDS)
