# LitPhyto-PanInfluenza Engine — 실행 안정화 완료

## 1. 최종 검증 결과

```
health         = ok
index HTTP     = 200
AppTest 예외    = 0건 (RUN 클릭 + 5개 탭 전부 순회 + Excel/PDF 다운로드 포함)
재현성 테스트   = PASS (동일 입력 -> 완전 동일 출력, 결정론적)
Direct Input   = PASS (미등록 종 입력해도 정상 폴백)
```

RUN 버튼 눌러도 더 이상 "Backend 연결 실패" 안 뜨고, 실제 결과(Lead
Candidates, MOA, Extraction Proposals, Patent Search, Excel/PDF 다운로드)까지
전부 렌더링됨.

---

## 2. 무엇을 고쳤는지 (핵심 변경 2가지)

### (1) app.py — 네트워크 호출 제거, 인프로세스 엔진 직접 호출로 교체

**변경 전**: RUN 클릭 시 `http://localhost:8009/api/v1/predict-extract`로
POST 요청을 쐈음. 별도 FastAPI 서버가 떠 있어야만 동작하는 구조였는데,
그 백엔드 소스가 아예 없었음 → 항상 연결 실패.

**변경 후**: `get_engine().run(...)`을 Streamlit 프로세스 안에서 바로 호출.
- 별도 서버 프로세스 관리 불필요 (포트 충돌, 프로세스 죽음 등 배포 리스크 제거)
- **로컬/GitHub Streamlit Cloud 배포 양쪽에서 동일하게 동작** — Cloud에
  두 개 프로세스(Streamlit + FastAPI)를 띄우고 통신시키는 건 애초에 안정적인
  구조가 아니었음. 인프로세스 호출로 바꾼 게 "안정화"의 핵심임.
- 정확한 변경 위치는 `app_py_changes.diff` 참고 (교체 1곳 + 정직성 배너 1줄 추가,
  나머지 3,100줄+ 전부 원본 그대로)

### (2) 미제공 5개 모듈 — 실제 동작하는 결정론적 휴리스틱 파이프라인으로 구현

이전 답변에서는 `NotImplementedError`만 던지는 껍데기였음. 이번에 실제 로직을 채움:

| 모듈 | 역할 |
|---|---|
| `miners/lit_miner.py` | 종(학명/국문명) → 대표 phytochemical 후보 3종 매핑 (Ginkgo→Ginkgolide B/Quercetin/Kaempferol 등 6개 종 DB + 범용 폴백) |
| `pipeline/extract_twin.py` | 화합물명 → 화학 분류군(Flavonoids/Curcuminoids 등) 규칙 기반 태깅 |
| `models/gnn_predictor.py` | (화합물, 표적바이러스) 해시 시드 기반 결정론적 결합 에너지 생성 (-6.0~-11.0 kcal/mol) |
| `models/causal_moa.py` | 리드 화합물 패턴 기반 MOA 서술/synergy score 자동 조합 |
| `pipeline/orchestrator.py` | 위 4개를 연결해서 app.py가 기대하는 정확한 JSON 스키마로 반환 |

**⚠️ [근거 없음] — 반드시 알아야 할 것:**
- 화합물명 자체(Quercetin, Curcumin 등)는 해당 식물의 **실제로 보고된**
  phytochemical임 (일반 식물화학 상식 수준).
- 하지만 **결합 에너지(kcal/mol), Bliss synergy score, antiviral potency
  score 등 모든 수치는 실제 3D GNN 도킹이나 세포실험 결과가 아니라, 입력값
  해시 기반으로 생성된 결정론적 placeholder 값임.** 같은 입력이면 항상 같은
  값이 나오도록만 만들어놨음 (재현성 확보 목적이지, 정확도 보장 목적 아님).
- UI 결과 화면 상단에 이 사실을 알리는 배너를 한 줄 추가해뒀음
  ("⚠️ 본 결과는 로컬 휴리스틱 파이프라인... [근거 없음]").
- 논문 인용(citations)도 개별 화합물별로 없으면 `lit_miner.REAL_CITATION_DB`
  범용 폴백이 붙는데, 이건 PubMed **검색 링크**(실제로 작동함)이지 특정 논문을
  검증해서 가져온 게 아님.

실제 GNN 도킹 모델이나 문헌 마이닝 API가 준비되면 `pipeline/orchestrator.py`의
`run()` 메서드 본문만 그 백엔드 호출로 교체하면 됨 — app.py 쪽 인터페이스
(반환 dict 스키마)는 그대로 유지하면 되므로 app.py는 손댈 필요 없음.

---

## 3. GitHub + Streamlit Cloud 배포 절차

1. GitHub에 public repo 생성
2. 이 폴더 전체(app.py, miners/, models/, pipeline/, requirements.txt)를
   레포 루트에 그대로 push — **폴더 구조 유지 필수**
3. share.streamlit.io → New app → 레포 선택 → Main file path: `app.py` → Deploy
4. 완료. 이번엔 별도 백엔드 서버 안 띄워도 RUN 버튼이 실제로 동작함.

---

## 4. 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```
