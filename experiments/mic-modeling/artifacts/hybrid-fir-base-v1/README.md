# Hybrid FIR base v1

앱 통합 후보로 보존하는 iPhone → KM184 선형 전처리 필터입니다. 이 artifact는
`hybrid-fir-residual-16ch-relative-active` run에서 선택된 **epoch 0** 상태입니다.
따라서 현재 선택 출력은 residual neural network를 더하지 않은 FIR-only입니다.

## 구성

- `fir_70_4k_train_only.npy`: 48 kHz, 4,097-tap, float32 선형위상 FIR 계수
- 설계 데이터: KM184 원본 시간 0–50초와 이에 정렬한 iPhone 입력만 사용
- 목표 대역: 70 Hz–4.0 kHz, 전이 대역은 50–70 Hz 및 4.0–4.25 kHz
- 설계 방법: iPhone/KM184 median STFT power ratio를 1/6 옥타브로 평활화하고,
  gain을 -8 dB~+10 dB로 제한한 뒤 `scipy.signal.firwin2`로 설계
- 적용: 중앙 정렬 선형위상 FIR. 현재 방식은 오프라인 `fftconvolve(..., mode="same")`입니다.

## 현재 근거와 한계

mic-fr 페이지와 같은 전체 공통 구간의 상대 스펙트럼 계산에서 70–4 kHz 평균 절대 오차는
Hybrid FIR base가 1.824 dB, 기존 `iphone2km184` FIR가 2.045 dB였습니다. 이 수치는 같은
연주·같은 배치의 programme-dependent 비교이며, 보편적 마이크 FR이나 다른 녹음 환경의
성능을 뜻하지 않습니다.

앱에 넣기 전에는 다음을 결정해야 합니다.

1. 실시간 모드: 4,097-tap 선형위상 FIR의 약 42.7 ms group delay를 허용할지,
   최소위상 IIR 또는 짧은 FIR로 근사할지 결정합니다.
2. 입력 보정: 현재 학습에는 iPhone 극성, DC, 고정 gain 정규화가 포함됩니다.
   앱에는 해당 전처리를 명시적으로 구현하거나 calibration 절차를 제공해야 합니다.
3. 일반화: 다른 거리·연주·iPhone 설정에서 별도 paired test를 수행해야 합니다.
