"""First-use guide for the shared dashboard; navigation only, with no API calls."""

from __future__ import annotations

import streamlit as st


def _open_page(tab: str) -> None:
    """Leave the dashboard router in charge of the actual page transition."""
    st.session_state["_pending_tab"] = tab
    st.rerun()


def render_guide() -> None:
    """Show the three main tasks, demo boundaries, and short help on demand."""
    st.markdown(
        """
        <style>
        .guide-eyebrow {
            color: var(--mg-accent, var(--mg-text, #0f766e));
            font-size: .88rem; font-weight: 700; letter-spacing: .04em;
            margin: 0 0 .35rem;
        }
        .guide-lead {
            color: var(--mg-text-muted, #64748b);
            font-size: 1.05rem; line-height: 1.7; margin: 0 0 1.1rem;
        }
        .guide-step {
            color: var(--mg-text, #172b32);
            font-size: 1rem; line-height: 1.65; margin: .25rem 0 .8rem;
        }
        .guide-step strong {
            color: var(--mg-accent, var(--mg-text, #0f766e));
        }
        </style>
        <div class="guide-eyebrow">마가렛트 · 처음 사용하시나요?</div>
        """,
        unsafe_allow_html=True,
    )
    st.title("전력 현황부터 점검 보고서까지")
    st.markdown(
        '<p class="guide-lead">원하는 작업을 선택해 시작하세요. '
        '어느 화면에서든 시작 가이드로 돌아올 수 있어요.</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "과거 데이터로 체험하는 서비스예요. 실시간으로 설비에 연결되어 있지 않아요. "
        "기준 시점은 2025년 4월 30일 00:00이며, 센서 실측은 4월 29일까지 제공돼요."
    )

    budget, inspection, report = st.columns(3, gap="medium")
    with budget:
        with st.container(border=True):
            st.subheader("예산 확인")
            st.write("전력 사용 추이와 예상 전기요금·탄소 배출량을 확인해요.")
            st.caption("실측과 예측을 구분해서 볼 수 있어요.")
            if st.button("현황·예측 보기 →", key="guide_budget", use_container_width=True):
                _open_page("현황·예측")
    with inspection:
        with st.container(border=True):
            st.subheader("점검 대상 확인")
            st.write("설비 13대를 한 번에 비교하고, 큰 변화·지속 이탈의 근거를 살펴봐요.")
            st.caption("이상 표시는 고장 확정을 뜻하지 않아요.")
            if st.button("설비 점검 시작 →", key="guide_inspection", use_container_width=True):
                _open_page("이상탐지")
    with report:
        with st.container(border=True):
            st.subheader("보고서 작성")
            st.write("데이터를 질문하고 점검 보고서를 Excel로 내려받아요.")
            st.caption("초안을 확인하고 담당자·조치 내용을 보완해요.")
            if st.button("보고서 도우미 열기 →", key="guide_report", use_container_width=True):
                _open_page("에이전트")

    st.subheader("처음이라면 이 순서로 해보세요")
    first, second, third = st.columns(3, gap="medium")
    with first:
        st.markdown(
            '<p class="guide-step"><strong>01 · 현황 보기</strong><br>'
            '날짜를 확인하고 전력 사용량과 예상 요금을 살펴보세요.</p>',
            unsafe_allow_html=True,
        )
    with second:
        st.markdown(
            '<p class="guide-step"><strong>02 · 근거 확인하기</strong><br>'
            '전체 설비 카드에서 하나를 고르고, 확인 정보와 원래 AI 기록을 살펴보세요.</p>',
            unsafe_allow_html=True,
        )
    with third:
        st.markdown(
            '<p class="guide-step"><strong>03 · 보고서 완성하기</strong><br>'
            '확인한 내용을 담고, 필수 정보를 입력한 뒤 파일로 내려받으세요.</p>',
            unsafe_allow_html=True,
        )

    st.subheader("화면을 읽는 데 필요한 세 가지")
    with st.container(border=True):
        st.markdown(
            "**설정 기준 이탈**은 정한 전압·역률 범위를 벗어난 기록이고, "
            "**AI 탐지**는 모델이 평소와 다르다고 표시한 기록이에요. "
            "둘 다 실제 고장 여부는 현장 확인이 필요해요."
        )
        st.markdown(
            "**탐지 기록 수와 점검할 설비 수는 달라요.** "
            "같은 설비에서 여러 기록이 나올 수 있으니, "
            "숫자 옆의 ‘대·개·구간’과 시간 단위를 함께 확인하세요."
        )
        st.markdown(
            "**자료 없음은 정상이나 0을 뜻하지 않아요.** "
            "선택한 날짜에 실측이 없으면 상태를 판단하지 않으며, "
            "미래 날짜의 예측값으로 전압 상태를 판단하지 않아요."
        )

    st.subheader("궁금할 때 확인하세요")
    with st.expander("전체 설비와 상세 기록은 어떻게 보나요?"):
        st.write("현황·예측과 설비 점검 화면에서는 등록된 설비 13대를 한 번에 보여줘요. 카드의 전압·역률은 선택일 R/S/T상 전체의 최솟값~최댓값이에요. 카드의 ‘보기 →’를 누르거나 상세 설비를 선택하면 아래 근거만 좁혀 보고, 전체 카드는 유지돼요.")
        st.write("‘확인할 정보’는 조건에 맞는 기록을 묶은 목록, ‘AI 원기록’은 모델이 탐지한 전체 구간, ‘기록만 보관’은 확인 정보와 연결되지 않은 AI 기록이에요. 보관 기록을 정상으로 재분류하거나 삭제하지 않아요. 추가 확인 정보가 없어도 정상으로 확정할 수는 없어요.")
        st.caption("화면 표에는 주요 값과 시각을 먼저 보여줘요. CSV에는 적용 조건과 원기록 연결 ID를 포함한 상세 근거가 남아요. 시각은 모두 한국시간이에요.")

    with st.expander("AI 기록과 ‘확인할 정보’는 어떻게 다른가요?"):
        st.write("AI 탐지는 학습된 모델의 기존 판정이에요. ‘확인할 정보’는 이 탐지 근거와 실제 센서 조건이 함께 확인될 때 만드는 점검 정보예요. AI 모델·점수 기준·원래 판정은 바꾸지 않고 모든 원기록을 보관해요. AI 점수는 고장 확률이 아니에요.")
        st.write("큰 전압 이탈과 역률 급락은 측정 조건과 AI 탐지가 같은 5초 표본에서 확인될 때 기록해요. 지속 정보는 같은 설비·같은 상의 센서값이 조건을 계속 충족하고, 그 연속 구간 안에 AI 탐지 근거도 있을 때 만들어요. 모든 표본이 AI 이상이어야 한다는 뜻은 아니에요.")
        st.write("‘확인 시각’은 필요한 지속 시간과 AI 근거를 모두 확보한 시점이에요. AI 근거가 나중에 나오면 시각도 그때로 늦어져요. 미래 기록으로 이전 시점의 판정을 바꾸지 않아요. 팝업이나 외부 메시지를 반복해서 보내는 기능은 없어요.")

    with st.expander("확인 기준은 무엇이고 어떻게 바꾸나요?"):
        st.caption("220V 상전압을 가정한 과제용 기본값이에요. 설비의 안전·고장을 판정하는 검증된 기준이 아니며, 원래 AI 라벨이나 모델 평가 기준과도 별개예요.")
        st.markdown(
            "| 확인 항목 | 과제용 기본값 |\n"
            "| --- | --- |\n"
            "| 큰 전압 이탈 | 198V 미만 또는 242V 초과, 한 번 관측 |\n"
            "| 전압 이탈 지속 | 207V 미만 또는 233V 초과가 1분 지속 |\n"
            "| 역률 급락 | 70% 미만이면서 급락 구간 시작 전 5분 중앙값보다 20%p 이상 하락 |\n"
            "| 낮은 역률 지속 | 80% 미만이 3분 지속 |"
        )
        st.write("‘확인 기준 조정’에서 수치와 시간을 입력하고 ‘이 기준 적용’을 누르세요. 현재 브라우저의 확인 정보에만 반영되고 AI 판정은 바뀌지 않아요. ‘과제용 기본값 복원’으로 되돌릴 수 있어요. 역률이 90%에서 70%가 되면 20%p 하락이에요.")
        st.write("역률 급락의 기준선은 같은 상에서 급락 구간이 시작되기 전 연속 5분 기록의 중앙값이며, 그 구간 동안 고정해요. 현재 값과 미래 값은 기준선에 넣지 않아요. AI 근거가 늦게 확인되면 확인 시각 바로 직전의 중앙값과 다를 수 있어요. 비교할 기록이 부족한 구간은 급락 판단에서 제외하지만 AI 원기록은 보관하고 다른 조건은 별도로 확인해요.")
        st.write("팀 공유본은 기본 기준으로 미리 계산한 결과를 바로 볼 수 있어요. 원본 5초 실측 자료가 연결되지 않은 환경에서는 기준 변경이 잠기며, 기본값 복원만 가능해요. 원본 자료를 연결하면 조건을 다시 계산할 수 있어요.")

    with st.expander("정보 묶음·AI 구간·지속 시간은 어떻게 세나요?"):
        st.write("AI 원구간은 같은 설비에서 AI 탐지가 연속된 5초 기록을 묶은 것이에요. 확인 정보는 센서 항목별로 만들어서, 하나의 AI 원구간이 여러 상의 정보에 연결될 수도 있어요. AI 구간 수·정보 묶음 수·설비 대수는 단위가 다르고 어느 것도 고장 건수가 아니에요.")
        st.write("센서 조건이 계속되면 한 발생 구간으로 만들어요. 정상값·누락·관측 중단·날짜 경계에서 나누고, 빈 시간을 건너뛰어 지속 시간을 늘리지 않아요. 설정 기준 기록은 센서별 시간이 겹칠 수 있고, 누적 시간은 여러 짧은 구간의 합일 수 있어요.")
        st.write("기본값에서는 같은 설비·센서·유형·방향의 다음 정보가 마지막 확인 시각 뒤 10분 미만에 나오면 한 묶음으로 보여줘요. 간격이 10분 이상이면 새 묶음이에요. 묶음의 처음부터 마지막까지 같은 상태가 계속됐다는 뜻은 아니에요. 묶을 간격을 0분으로 설정하면 각각 표시해요.")
        st.write("‘첫 확인 값’은 처음 조건을 확인한 시점의 값이에요. ‘구간 대표값’은 이후 기록까지 포함한 가장 큰 이탈 값이라 서로 다를 수 있어요. 개별 발생 구간을 펼치면 실제 연속 지속 시간을 확인할 수 있어요.")
        st.caption("시간별 최저값 그래프는 각 시간의 최저 관측값 참고용이에요. 선이 낮게 이어져도 1시간 내내 이탈했다는 뜻은 아니에요. 상한 초과·실제 지속 시간·역률 급락은 5초 기록 기반 근거에서 확인하세요.")

    with st.expander("실측·예측·월 예상값은 어떻게 다른가요?"):
        st.write(
            "실측은 센서에서 수집한 과거 기록이고, 예측은 학습한 모델이 계산한 값이에요. "
            "월 예상값은 해당 월의 실측과 남은 기간의 예측을 합칠 수 있으므로, "
            "자료가 없는 시간이 평균으로 보정되면 별도로 표시해요. 화면의 기간과 구성 설명을 함께 확인하세요. "
            "월 예상 요금은 월 전체 예산을 보는 값이고, 일평균 전월 대비는 월별 일수 차이를 제외한 사용 패턴 비교예요."
        )
        st.write(
            "월별 화면에 쓰는 예측 범위는 2025년 4월 30일~5월 31일이에요. "
            "과제의 성능 평가 구간은 5월 10일 00:00~5월 31일 00:00의 "
            "505개 시간별 값으로, 5월 전체 사용량과는 범위가 달라요."
        )
        st.caption("실측과 예측 그래프는 자료가 없는 시간을 선으로 연결하지 않아요. 월 총량을 비교할 때는 월별 일수 차이도 확인하세요.")
    with st.expander("이상 기록이 나온 시간과 실제 지속 시간은 같은가요?"):
        st.write(
            "달라요. 특정 시간대에 이상 기록이 한 번 나왔다고 해서 "
            "그 상태가 1시간 내내 이어졌다는 뜻은 아니에요."
        )
        st.write(
            "현재 4월은 5초 기록, 이전 달은 시간별 집계를 사용해요. 5초 기록을 확인할 수 있는 구간에서만 탐지 누적 시간과 "
            "연속 구간을 계산해요. 시간별 집계만 있으면 ‘관측된 시간대’로 안내해요. "
            "탐지 누적 시간도 여러 짧은 구간의 합일 수 있으니, 최장 연속 시간과 구분하세요."
        )
    with st.expander("MAE와 전기요금·탄소 배출량은 어떻게 읽나요?"):
        st.write(
            "MAE는 예측값과 실제값이 평균적으로 얼마나 차이 났는지 나타내요. "
            "예를 들어 전력량 MAE가 1kWh이면 평가한 값의 절대 오차가 평균 1kWh라는 뜻이에요. "
            "같은 기간·같은 단위로 평가했을 때 작을수록 오차가 적어요."
        )
        st.markdown(
            "- **예상 전기요금:** 전력량(kWh) × 180원\n"
            "- **탄소 배출량:** 전력량(kWh) × 0.424kgCO₂"
        )
        st.caption(
            "과제에서 정한 고정 계수를 사용한 추정치예요. 실제 요금 청구서나 공식 탄소 산정서를 대신하지 않아요."
        )
    with st.expander("보고서 도우미에는 무엇을 질문하면 되나요?"):
        st.markdown(
            "- ‘4월 29일 점검할 설비를 정리해줘.’\n"
            "- ‘3호기 전압 상태를 알려줘.’\n"
            "- ‘확인한 내용을 점검 보고서로 정리해줘.’"
        )
        st.write(
            "날짜와 설비를 함께 말하면 범위를 더 명확히 정할 수 있어요. "
            "보고서 초안의 수치와 판정 근거를 확인하고, 담당자가 수행한 조치와 "
            "점검자 등 필요한 정보를 보완한 뒤 다운로드하세요."
        )

    with st.expander("보고서의 근거와 수정 내용은 어떻게 보관하나요?"):
        st.write("‘이 설비로 보고서 시작’을 누르면 선택한 날짜·설비·확인 기준과 조회 근거를 초안 생성 시점에 고정해요. 이후 화면의 날짜나 기준을 바꿔도 기존 초안이 자동으로 바뀌지는 않아요. 새 조건을 담으려면 새 보고서를 시작하세요.")
        st.write("Excel에는 AI 원기록 전체와 센서 확인 정보·정보 묶음을 별첨해요. 확인 시점의 값과 나중에 관측한 대표값은 구분해서 읽어야 해요. AI가 실제 고장 원인이나 조치 완료 여부를 확정하지 않으므로 담당자가 확인한 판정·조치 내용을 보완하세요.")
        st.write("담당자 판정이나 내용을 수정하면 다시 확인한 뒤 내보내세요. 입력 내용을 지우고 반영하면 출력에서도 삭제돼요. 초안은 현재 세션에 보관되므로 화면을 종료하기 전에 Excel로 내려받으세요.")
