#!/usr/bin/env python3
"""HSI-Drive domain RX 실험 코드 가이드.

실행:
    python domain_0_guide.py
"""

GUIDE = r"""
HSI-Drive Domain RX 실험 가이드
===============================

1. 권장 실행 순서
-----------------
1) domain_1_aligned_rx.py
   - 모든 날씨 도메인의 normal Road feature를 CORAL로 정렬한 뒤,
     하나의 domain-integrated RX 모델을 학습한다.
   - 기본 결과 폴더: domain_aligned_rx_results/

2) domain_2_visualize_aligned_rx.py 또는 domain_3_aligned_rx_viewer.py
   - domain 1의 결과를 이미지·분포·interactive viewer로 확인한다.

3) domain_4_compare_single_weather_rx.py
   - 날씨별 single-weather RX, pooled RX, CORAL-aligned RX를 독립적으로
     학습하고 cross-weather false-alarm rate를 비교한다.
   - 기본 결과 폴더: single_weather_rx_comparison_results/

4) 비교·집계 도구
   - domain_5_single_weather_rx_viewer.py
   - domain_6_rx_comparison_viewer.py
   - domain_6_compare_cross_weather_heatmaps.py
   - domain_7_aggregate_rx_comparison.py


2. 파일별 역할과 의존성
------------------------
domain_1_aligned_rx.py
  - 핵심 domain-integrated CORAL RX 학습 코드.
  - 결과: results.csv, rx_background.npz, coral_alignment.npz,
    이미지별 *_scores.npy.

domain_2_visualize_aligned_rx.py
  - domain 1 결과의 feature 정렬, score 분포, 개별 score map, 날씨별 FAR을
    정적 이미지로 만든다.
  - domain 1 결과가 먼저 필요하다.

domain_3_aligned_rx_viewer.py
  - domain 1의 held-out normal 이미지에 대한 interactive score-map viewer.
  - domain 1 결과가 먼저 필요하다.

domain_4_compare_single_weather_rx.py
  - 각 학습 날씨별 single-weather RX와 pooled / CORAL-aligned RX를 학습해
    train weather x test weather FAR을 비교한다.
  - domain_1_aligned_rx.py의 데이터 로딩·RX·CORAL 유틸 함수를 import한다.
  - 하지만 domain 1의 결과 파일은 읽지 않으며, 자체 결과를 새로 생성한다.

domain_5_single_weather_rx_viewer.py
  - domain 4에서 저장된 단일 날씨 RX 모델을 선택해 held-out 이미지의
    score map과 false-alarm overlay를 interactive하게 보여 준다.
  - domain 4 결과가 먼저 필요하다.

domain_6_rx_comparison_viewer.py
  - 동일 held-out 이미지에서 domain-integrated CORAL RX와 모든 단일 날씨
    RX의 score map / false-alarm overlay를 한 화면에서 비교한다.
  - domain 1과 domain 4 결과가 모두 필요하다.

domain_6_compare_cross_weather_heatmaps.py
  - single-weather RX 행들 아래에 all (CORAL) 행을 붙인 하나의
    cross-weather FAR heatmap을 그룹별로 생성한다.
  - domain 1과 domain 4 결과가 모두 필요하다.

domain_7_aggregate_rx_comparison.py
  - 모든 그룹의 single / pooled / CORAL-aligned RX 결과를 집계한다.
  - macro-average heatmap, 그룹별 paired 비교 그림, CSV 요약을 생성한다.
  - domain 4 결과가 먼저 필요하다.


3. 중요한 주의 사항
--------------------
1) domain 1과 domain 4의 관계
   domain 4는 domain 1의 함수를 재사용하지만 domain 1의 학습 모델이나
   score map을 그대로 사용하지 않는다. 같은 데이터와 같은 split 규칙으로
   별도의 비교 실험을 수행한다.

2) 결과 폴더를 지우지 말 것
   domain 2·3은 domain_aligned_rx_results/를, domain 5·7은
   single_weather_rx_comparison_results/를 읽는다. domain 6 계열은 두 폴더를
   모두 읽는다.

3) 공정한 전체 집계 방법
   전체 픽셀을 합쳐 FAR을 계산하면 픽셀 수가 큰 이미지·그룹이 결과를
   지배한다. domain_7은 그룹별 FAR을 먼저 계산한 뒤 그룹에 동일 가중치를
   주는 macro-average를 사용한다.

4) single-weather와 integrated RX의 해석
   single-weather RX는 대각선(학습 날씨 = 테스트 날씨)보다 비대각선
   cross-weather FAR이 날씨 domain shift를 보여 주는 핵심 지표다.
   CORAL-aligned RX는 모든 날씨를 함께 학습하므로 heatmap에서
   "all (CORAL)" 한 행으로 표시된다.

5) 실행 환경
   정적 그림 생성에는 matplotlib, interactive viewer에는 tkinter와 Pillow가
   필요하다. GUI가 없는 환경에서는 viewer 창을 띄울 수 없다.
""".strip()


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
