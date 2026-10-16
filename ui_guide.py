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
            st.write("확인이 필요한 설비를 고르고, 해당 기록과 그래프를 살펴봐요.")
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
            '점검할 설비를 선택하고 언제, 어떤 항목이 달라졌는지 확인하세요.</p>',
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
    with st.expander("실측·예측·월 예상값은 어떻게 다른가요?"):
        st.write(
            "실측은 센서에서 수집한 과거 기록이고, 예측은 학습한 모델이 계산한 값이에요. "
            "월 예상값은 해당 월의 실측과 남은 기간의 예측을 합칠 수 있으므로, "
            "자료가 없는 시간이 평균으로 보정되면 별도로 표시해요. 화면의 기간과 구성 설명을 함께 확인하세요."
        )
        st.write(
            "월별 화면에 쓰는 예측 범위는 2025년 4월 30일~5월 31일이에요. "
            "과제의 성능 평가 구간은 5월 10일 00:00~5월 31일 00:00의 "
            "505개 시간별 값으로, 5월 전체 사용량과는 범위가 달라요."
        )
        st.caption("월 총량을 비교할 때는 월별 일수가 다르다는 점도 확인하세요.")
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
