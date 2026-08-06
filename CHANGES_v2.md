# 이번 요청 6개 처리 결과

## 1. 기본값 = 직접 입력
`plant_presets` 리스트 맨 앞으로 "Direct Input (직접 입력)"을 옮기고 `index=0`.
AppTest로 기본 선택값이 Direct Input인 것 확인함.

## 2. 로딩 애니메이션 세련되게 + 2배 느리게
펄스 글로우 아이콘(`lp_pulse`), 좌→우 시머 스윕(`lp_shimmer`), 회전 스피너(`lp_spin`),
페이드인(`lp_fadein`) 4개 CSS keyframe 애니메이션 추가함. 전체 소요시간
2.5초 → 5초로 늘리고, 큰 점프 대신 단계마다 한 번 더 세분화해서
진행바가 부드럽게 움직이도록 함.

**버그 발견 & 수정**: 이 작업 중 CSS의 `{ }`가 기존 `.format()` 호출과 충돌해서
`ValueError: unexpected '{' in field name`로 앱이 죽는 걸 발견함. CSS 중괄호를
전부 `{{ }}`로 이스케이프해서 해결함. AppTest로 RUN 전체 흐름 재검증 완료.

## 3. 탭 스타일 = 크롬탭, 전체 폭
**근본 원인 재진단**: 이전 버전은 `st.radio` + CSS로 가짜 탭을 흉내냈는데,
CSS가 `div[data-testid="stRadio"]` 같은 Streamlit *내부* DOM 구조를 직접
타겟팅하고 있었음. 배포된 Streamlit Cloud의 실제 버전에서 내부 구조가
조금만 달라도 스타일이 통째로 안 먹힘 — 스크린샷에서 라디오 동그라미가
그대로 보인 게 바로 그 증거였음.

**근본적으로 더 안정적인 방식으로 교체**: `st.radio` 해킹을 완전히 걷어내고
Streamlit 공식 네이티브 컴포넌트인 **`st.tabs()`**로 전체 리팩터링함
(`tab1~tab5 = st.tabs([...])` + `with tabN:` 구조, 5개 분기 전부 기계적으로
치환, 각 탭 안쪽 로직은 한 줄도 안 건드림). CSS도 Streamlit이 실제로 쓰는
BaseWeb 라이브러리의 `data-baseweb` 속성(내부 data-testid보다 버전 간
안정적)으로 다시 타겟팅해서, 탭이 전체 폭 꽉 차게(`flex:1`) 넉넉한 여백으로
배치되도록 함.

## 4. H1N1 다이어그램 이미지 누락
**원인**: 지난 턴에 프로젝트 정리하면서 제가 `rm -rf static/*.png`로
**실수로 3개 PNG 파일을 전부 지웠던 것**이었음 (`h1n1_lifecycle_diagram.png`,
`_clean.png`, `_notitle.png`). app.py 코드 자체는 멀쩡했고 파일만 없었음.

원본 압축 파일에서 3개 전부 복원했고, 사용자가 재첨부해준 `_clean.png`와
md5 해시가 완전히 동일한 것 확인함. 사용자가 지정한 5번째 사진(=clean 버전)을
최우선으로 로드하도록 순서 변경함 (`_clean.png` → `_notitle.png` → `.png` 순).

## 5. reportlab 오류 + 추출법 다운로드 실종
**근본 원인**: `requirements.txt`에 `reportlab`이 통째로 빠져 있었음.
`generate_single_protocol_pdf_bytes()`가 매 옵션 카드마다 무조건 호출되는
구조라, 그 함수 안의 `from reportlab...` import 한 줄이 `ModuleNotFoundError`를
내면서 **앱 전체가 죽었음** — "추출법 다운로드가 없어졌다"고 느낀 이유가
바로 이거였음 (버튼이 삭제된 게 아니라 그 지점에서 스크립트 자체가 멈춘 것).

`requirements.txt`에 `reportlab` 추가함.

**추가로 발견한 버그 3개** (같은 함수 안에서):
- **윈도우 전용 폰트 경로**: `C:\Windows\Fonts\malgun.ttf`를 썼는데 리눅스
  서버엔 당연히 없어서 항상 Helvetica로 폴백 → Helvetica엔 한글 글리프가
  없어서 **PDF의 모든 한글이 깨져 나올 뻔했음**. ReportLab에 내장된 한글
  CID 폰트(`HYSMyeongJo-Medium`, 외부 파일 불필요, OS 무관하게 항상 동작)로
  교체함. 실제 PDF를 렌더링해서 한글이 정상 출력되는 것까지 이미지로 확인함.
- **이모지 깨짐**: CID 폰트가 이모지 글리프를 지원 안 해서 "📋"가 PDF에서
  전혀 다른 문자로 깨져 나오고 있었음. PDF 전용 헤딩에서만 이모지 제거함
  (HTML/UI 쪽은 브라우저가 이모지를 정상 렌더링하므로 그대로 둠).
- **`{clean_plant}` 리터럴 노출**: Curcuma longa 옵션의 SOP Step 03, 05
  detail 텍스트에 `f` 접두사가 빠져 있어서 변수 치환이 안 되고 `{clean_plant}`
  글자 그대로 PDF에 찍히고 있었음. 전체 파일을 정규식으로 훑어서 같은 패턴이
  더 있는지 확인했고, 이 2곳이 전부였음 - 둘 다 `f"..."`로 수정함.

옵션별 개별 PDF 다운로드 버튼(고유 key, `mime="application/pdf"`)은 원래
코드에 이미 있었음 — reportlab 문제만 해결하면 원래대로 동작하는 구조였음.
추가로 방어 코드도 넣어서, 혹시 나중에 또 예외가 생겨도 그 옵션 하나만
다운로드 버튼이 빠지고 나머지 앱은 안 죽도록 처리함.

## 검증
- 공식 테스트 스위트 5/5 통과
- Ginkgo/Curcuma/Sambucus/Panax 4개 종 × Direct Input 경로로 RUN, 예외 0건
- 각 종마다 다운로드 버튼 5개(옵션 PDF×3 + Excel + HTML리포트) 전부 렌더 확인
- 실제 PDF 바이트를 꺼내서 fitz로 이미지 렌더링 → 한글 정상, 이모지 깨짐 없음,
  `{clean_plant}` 리터럴 없음 육안 확인
- 라이브 서버 기동 확인 (health=ok, HTTP 200)
