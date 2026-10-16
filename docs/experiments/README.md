# 실험 기록

VPP 구성과 예측 개선 실험의 결론·기각 근거를 남기는 디렉터리다.
scratch 결과는 이 디렉터리에 기록되기 전까지 근거로 쓰지 않는다.

## 색인

- `vpp_composition.md`: 반클러스터링 이론, 상쇄 손실 분해, `geo64-s0` 채택 근거와 기각된 대안.
- `feature_headroom.md`: 특성 묶음 효과, 흐림 시간 분석, 중단된 특성 방향.
- `validation_noise.md`: 학습 잡음 규모, 비교 규칙의 근거, 코드 버전 고정 원칙.

## 항목 형식

가설 / 설정 / 결과 / 판정(채택·기각·보류) / 근거 출처.

## 근거 출처 등급

- `[재현]`: registry run_id 또는 운영 CLI 명령으로 다시 만들 수 있음.
- `[registry]`: registry 기록만 있음.
- `[탐색]`: scratch에서 측정했고 코드는 폐기됨. 수치만 남음.

## 문서 정비 범위

`docs/architecture/overview.md`의 계획 단계 compose 모듈 목록과 잔차 기반 탐색 서술은
별도 문서 정비 작업의 대상이다.
