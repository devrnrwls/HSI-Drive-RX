#!/usr/bin/env python3
"""Print the current HSI-Drive Domain RX guide."""

GUIDE = """
HSI-Drive Domain RX 가이드
===========================

도메인 factor: 0=season, 1=weather, 2=daytime, 3=roadtype.
--domain-factor에는 이름 또는 인덱스를 쓸 수 있습니다.

기본 실행
---------
python domain_1_aligned_rx.py
  - 네 factor 전체를 순차 학습하고 score map, score 분포, FAR 시각화를 생성합니다.
  - --no-visualize로 정적 시각화를 생략합니다.

python domain_2_compare_single_domain_rx.py
  - 네 factor 전체에 대해 single-domain, pooled, CORAL-aligned RX를 비교합니다.

특정 factor만 실행하려면:
  python domain_1_aligned_rx.py --domain-factor 0
  python domain_2_compare_single_domain_rx.py --domain-factor season

결과 폴더
---------
Integrated RX: domain_1_aligned_rx_results/ 아래의 season/, weather/, daytime/, roadtype/.
Single-domain 비교: domain_2_single_domain_rx_results/ 아래의 factor별 하위 폴더.
통합 heatmap: domain_3_integrated_cross_domain_comparisons/ 아래의 factor별 하위 폴더.
전체 집계: domain_3_overall_rx_comparisons/ 아래의 factor별 하위 폴더.
두 핵심 코드는 완료 시 결과 폴더의 절대 경로를 출력합니다.

파일 역할
---------
domain_1_aligned_rx.py: Integrated CORAL RX 학습, score 저장, 정적 시각화.
domain_1_aligned_rx_viewer.py: Integrated RX interactive viewer.
domain_2_compare_single_domain_rx.py: Single-domain / pooled / aligned RX 비교 학습.
domain_2_single_domain_rx_viewer.py: Single-domain RX interactive viewer.
domain_3_integrated_cross_domain_comparison.py: 그룹별 결합 FAR heatmap.
domain_3_integrated_vs_single_domain_viewer.py: Integrated와 모든 single-domain RX 병렬 viewer.
domain_4_overall_rx_comparison.py: 전체 그룹 macro-average 및 paired 비교.

주의 사항
---------
domain 2 비교 학습은 domain 1의 유틸을 재사용하지만 domain 1 결과 파일을
읽지 않고 별도로 학습합니다. 전체 성능은 raw pixel이 아니라 그룹별 FAR을
동일 가중치로 집계합니다. 정적 시각화에는 matplotlib, viewer에는 tkinter와
Pillow가 필요합니다.
""".strip()


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
