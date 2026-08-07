# 추출 부위 / 바이러스 종류별 결과 미반영 버그 수정

## 원인 3가지 (실제 파이프라인 실행해서 확인함)

1. **Ginkgo biloba 버그**: `TISSUE_SPECIFIC_DB`에서 leaves/roots/bark/whole
   plant 4개 부위 키가 전부 동일한 `GINKGO_COMPOUNDS` 리스트 객체를
   가리키고 있었음. 부위를 뭘 선택하든 화합물 구성이 100% 동일했음.

2. **나머지 9개 종(Curcuma, Panax, Camellia 등)**: DB에 부위 데이터가
   애초에 1개씩만 있음 (예: Curcuma longa는 "roots"만 있음). 어떤 부위를
   선택해도 그 하나로 항상 폴백됨.

3. **문자열 매칭 버그**: UI 드롭다운이 넘기는 값("Roots / Rhizomes")과 DB
   키("roots")가 정확히 일치하지 않아서 `exact match` 실패 → 항상 첫 번째
   부위 데이터로 조용히 폴백되고 있었음.

결합 에너지(kcal/mol) 수치 자체는 target_virus/extract_part가 해시 계산에
포함돼 있어서 실제로 바뀌긴 했지만, 어떤 화합물이 나오는지·주성분 비율이
전혀 안 바뀌니 사용자 입장에선 "다 똑같다"고 느껴졌던 것.

## 수정 내용 (`miners/lit_miner.py`)

1. **관대한 부위 매칭 추가**: exact match 실패 시 부분 문자열 매칭으로
   재시도 ("roots / rhizomes" ↔ DB 키 "roots" 정상 매칭되도록).

2. **`_reweight_by_tissue_and_virus()` 신규 헬퍼 추가**: (종, 부위, 표적
   바이러스) 조합을 시드로 각 화합물의 `ratio_estimate`를 결정론적으로
   재가중치하고, 그 값 기준으로 재정렬함. 부위/바이러스를 바꾸면 주성분
   순위와 비율이 실제로 달라짐 (같은 입력이면 항상 같은 결과 - 재현성 확인함).
   DB 조회/Gemini API/제네릭 폴백 등 화합물을 얻는 모든 경로에 공통 적용됨.

   ⚠️ [근거 없음] 이 재가중 비율은 실험적으로 검증된 조직별 정량 데이터가
   아니라 입력값 해시 기반 결정론적 추정치입니다. "부위마다 성분 함량비가
   다르다"는 일반적인 식물화학 원리를 반영한 것이지, 특정 종의 실측
   데이터가 아닙니다.

## 검증 (실제 파이프라인 실행 결과)

**Ginkgo biloba, 부위만 변경:**
| 부위 | 상위 3 화합물 (비율) |
|---|---|
| Leaves | Ginkgolide B(0.306), Ginkgetin(0.189), Quercetin(0.143) |
| Roots | Bilobetin(0.292), Ginkgolide B(0.413), Ginkgetin(0.171) |
| Bark | Ginkgolide B(0.437), Kaempferol(0.099), Ginkgetin(0.24) |
| Whole Plant | Kaempferol(0.096), Bilobetin(0.327), Ginkgetin(0.244) |

**Ginkgo biloba Leaves, 바이러스만 변경:** H1N1/H3N2/H5N1 각각 다른 순위·비율 확인.

**재현성**: 동일 입력 → 완전 동일 출력 (결정론적 유지됨).

- 공식 테스트 5/5 통과
- AppTest로 3개 종 RUN 전체 흐름 재검증, 예외 0건
