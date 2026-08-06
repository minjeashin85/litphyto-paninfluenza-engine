# 이번에 실제로 고친 것 — 첨부해준 원본 프로젝트 기준

첨부해준 `litphyto-panrna.zip`은 이전에 내가 임의로 채워넣은 껍데기 모듈이 아니라
**진짜 백엔드 4개 모듈 + FastAPI + 공식 테스트 스위트**가 전부 들어있는 원본이었음.
전체를 열어서 하나하나 확인했고, 실제로 살아있는 버그 4개 + app.py 구조 문제 1개를
찾아서 고쳤음. 나머지 핵심 로직(문헌 DB, 화합물 목록, 결합에너지 계산식, MOA 서술
로직)은 전혀 건들지 않음.

## 1. `pipeline/extract_twin.py` — 버그 2건

**(A) `torch` import 누락**
`_mol_to_pyg_graph()`에서 `torch.tensor(...)`를 바로 썼는데 파일 어디에도
`import torch`가 없었음. try/except로 감싸져 있어서 앱이 죽지는 않았지만,
PyG 그래프 생성 기능이 항상 조용히 실패하고 있었음(`NameError: name 'torch'
is not defined`가 로그에만 찍히고 무시됨). `models/gnn_predictor.py`와 같은
`HAS_TORCH` 가드 패턴으로 정식 임포트 추가함.

**(B) RDKit API 변경으로 3D conformer 생성이 항상 실패**
`AllChem.ETKDGv3().maxAttempts` 속성을 썼는데, 현재 RDKit(2026.03.5)에는
이 속성이 없고 `maxIterations`로 이름이 바뀌어 있었음. `AttributeError`가
try/except에 삼켜져서 `has_3d_conformer`가 항상 `False`로 나오고 있었음.
`hasattr()`로 버전 방어 처리해서 실제로 `True`로 바뀌는 것까지 확인함.

```
# 수정 전: has_3d_conformer: False (항상)
# 수정 후: has_3d_conformer: True  (RDKit 설치 시 실제로 3D 임베딩 성공)
```

## 2. `pipeline/orchestrator.py` — 버그 2건 (필드 유실)

Module 1(`lit_miner`)과 Module 3(`gnn_predictor`)이 실제로 만들어낸 데이터를
orchestrator가 최종 응답으로 재조립하는 과정에서 일부 필드를 빠뜨리고 있었음.

**(A) 인용정보 journal/pmid 유실**: `lit_miner.py`는 각 인용에 실제 저널명과
PMID를 채워주는데, orchestrator가 `formatted_leads` 만들 때 이 두 필드를
빼먹고 재조립함. 그 결과 app.py 화면에는 모든 논문이 항상 기본값
("Journal of Natural Products" / PMID "41395821")으로만 표시되고 있었음.

**(B) `compound_id`, `scores` 유실**: `gnn_predictor.py`가 각 리드 화합물에
PubChem CID(`compound_id`)와 `s_viral`/`s_host` 점수(`scores`)를 채워주는데,
orchestrator가 이것도 재조립 과정에서 빠뜨림. app.py는 `compound_id`로 실제
PubChem 2D 구조 이미지를 정확히 찾고, `scores`는 Excel 리포트에 쓰는데
둘 다 항상 빈 값이었음.

두 경우 다 원본 데이터를 그대로 보존하도록 고침. 4개 종(Ginkgo, Curcuma,
Panax, Justicia)으로 실제 실행해서 `compound_id`/`journal`/`scores`가 전부
정상적으로 채워지는 것 확인함.

## 3. `app.py` — 구조적 손상 (이전과 동일한 문제)

이 파일은 이전에 받았던 것과 **완전히 동일**(diff 결과 0줄)한 손상이 있었음:
`def main():` 선언부, 페이지 헤더, `st.columns()` 언패킹, `with col1:`
블록이 통째로 잘려나가서 `with col2:`가 고아 상태(`IndentationError`)였고
`main()`이 정의되지 않은 채로 파일 끝에서 호출되고 있었음. 저장/내보내기
과정에서 생긴 손상으로 보임 — 실제 백엔드 로직과는 무관한 문제.

같은 스캐폴드를 다시 붙였는데, 이번엔 실제 `TISSUE_SPECIFIC_DB`에 있는
10개 종(Ginkgo, Panax, Curcuma, Camellia, Allium, Zingiber, Glycyrrhiza,
Artemisia, Scutellaria, Justicia)을 프리셋 드롭다운에 정확히 맞춰 넣어서,
선택하면 실제 로컬 문헌 DB를 타도록 함 (이전 버전은 이 DB에 없는 임의
종 이름 위주였음).

## 4. 네트워크 호출 → 인프로세스 호출 (안정성)

`localhost:8009`로 POST하던 부분을 `get_engine().run_pipeline(...)` 직접
호출로 교체함. 실제 엔진의 메서드명이 `run()`이 아니라 `run_pipeline()`이라
지난번 버전(가짜 백엔드 기준으로 만들었던)의 메서드명이 애초에 틀렸었음 —
이번엔 진짜 시그니처에 정확히 맞춤. 별도 FastAPI 서버 없이 Streamlit 프로세스
안에서 바로 실행되므로 로컬/Streamlit Cloud 어디서든 동일하게 안정적으로 동작함.

## 5. UI 수정 (이전 대화에서 요청받은 것 재적용)

- 헤더 아래 빈 사각형 박스 제거 (닫는 태그 없던 `control-panel-box` div)
- 탭 스타일을 Safari 필박스 → 크롬탭 스타일로 (넉넉한 간격, 전체 폭)
- Claude 모델 ID를 최신 모델(`claude-sonnet-5` 등)로 교체
- 결과 화면에 정직성 배너 추가: 화합물명/인용은 실제 문헌 DB 기반이지만
  결합에너지·시너지 점수는 실제 GNN 추론이 아니라 결정론적 공식으로 계산됨을
  명시함 [근거 없음]

## 6. requirements.txt 분리

`torch`, `torch-geometric`은 `models/gnn_predictor.py`에 GIN 모델 클래스가
정의는 돼 있지만 실제 결합에너지 계산(`predict_leads_and_affinities`)은
`model.forward()`를 전혀 호출하지 않고 해시 기반 결정론적 공식만 씀 (코드로
확인함). 즉 torch가 있으나 없으나 예측값은 동일함. 반면 이 두 패키지는
용량이 커서 Streamlit Community Cloud 무료 티어에서 빌드 실패/타임아웃
위험이 큼 — 그래서 배포용 `requirements.txt`에서는 뺐고, 대신
`requirements-api.txt`(별도 FastAPI 서버 `api/main.py`를 로컬에서 돌릴 때만
필요)로 분리해뒀음. 나중에 실제 학습된 GNN 가중치를 쓰게 되면 그때
`requirements.txt`에 다시 추가하면 됨.

## 검증

- 원본 공식 테스트 스위트 `tests/test_engine.py` 5/5 통과
- Ginkgo/Curcuma/Panax/Justicia 4개 종 × 5개 탭 전부 AppTest로 실행, 예외 0건
- 라이브 서버 기동 확인 (health=ok, HTTP 200)
