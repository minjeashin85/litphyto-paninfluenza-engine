# 이번 요청 5개 처리 결과

## 1. 추출법 옵션도 닫힌 형태로
Option #1~#3을 각각 `st.expander(expanded=False)`로 감쌈 (제목에
Option 번호+이름+수율 지표 표시). 기존의 `st.container(border=True)`는
expander 자체가 테두리를 제공하므로 걷어냄.

## 2. 엔진 설정 박스 테두리 진하게
`st.container(border=True, key="ai_engine_panel")`로 key를 부여하면
Streamlit이 자동으로 `st-key-ai_engine_panel` CSS 클래스를 만들어준다는
점을 이용해, 이 컨테이너만 2.5px 인디고색 테두리 + 그림자로 강조해서
다른 영역과 확실히 구분되게 함.

## 3. 다운로드 버튼 색을 옵션 색으로
**1차 시도 실패 → 원인 파악 → 재수정**: 처음엔 `.st-key-dl_single_pdf_N
button`으로 스코프드 CSS를 넣었는데 실제로는 안 먹혔음(전부 옅은 초록
그대로). 원인: 기존 전역 CSS(`div[data-testid="stDownloadButton"]
button`)가 속성 선택자를 포함해서 명시도가 더 높았음. 스코프드 선택자에도
동일한 속성 선택자를 포함시켜(`.st-key-dl_single_pdf_N
div[data-testid="stDownloadButton"] button`) 명시도를 역전시킴.
`getComputedStyle`로 실제 배경색을 읽어서 Option #1=빨강(220,38,38),
#2=파랑(37,99,235), #3=초록(5,150,105) 정확히 일치하는 것 확인함.

## 4. 원문 검색 링크 색 제거
PubMed/Scholar/Patent 검색 링크 3개를 solid 색상 배경(초록/파랑/하늘)에서
흰 배경+회색 테두리의 무채색 스타일로 변경.

## 5. 로딩 애니메이션 재설계 (2차) + 텍스트 잘림 근본 원인 해결
**진짜 원인 발견**: 진행률 텍스트를 `st.progress(value, text=...)`의
네이티브 text 파라미터로 넘기고 있었는데, 이건 Streamlit 자체 내장
위젯이라 커스텀 CSS가 전혀 먹히지 않고 Streamlit 기본 스타일
(white-space:nowrap + ellipsis)이 그대로 적용되어 계속 잘리고 있었음
(1차 재설계 때 pill 3개로 나눈 건 카드 쪽 배지였지 진행바 텍스트 쪽은
그대로 방치돼 있었음).

**해결**: `st.progress`를 완전히 걷어내고 진행바 자체를 순수 HTML/CSS로
직접 그림 - 텍스트 잘림 가능성을 원천 차단함.

**디자인**: 첨부 참고 이미지(방사형 버스트 + LOADING)에서 착안해 완전히
새로 그림 - 이중 반대방향 회전 conic-gradient 링(방사형 버스트 패턴) +
펄스 코어 + 떠다니는 색상 블룸 효과. 연산 로그는 새 줄이 추가될 때마다
`transform: translateY()`로 전체가 부드럽게 위로 밀려 올라가는 티커
방식으로 바꿈 (기존엔 마지막 9줄만 잘라서 갈아끼우는 방식).

## 검증
- 공식 테스트 5/5 통과
- 3개 종 AppTest 전체 흐름 예외 0건
- playwright 실제 브라우저로 확인: 방사형 로딩 디자인, 로그 위로
  스크롤되는 티커 효과, 진행바 텍스트 안 잘림(원래 버그 재현 조건인
  "Roots/Rhizomes"로 재확인), 엔진설정 박스 진한 테두리, 옵션 3개 전부
  닫힌 상태, 다운로드 버튼 실제 배경색이 옵션 색과 정확히 일치, 원문검색
  링크 무채색 전환 전부 스크린샷/computed style로 확인
