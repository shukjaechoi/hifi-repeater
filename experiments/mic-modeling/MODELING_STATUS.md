# iPhone → KM184 모델링 현황 및 외부 검토용 요약

## 목표와 데이터

목표는 비슷한 위치의 동시 녹음에서 iPhone 파형을 KM184 녹음에 가깝게 변환하는 것이다.
48 kHz 단일 연주 약 76초를 사용했다. KM184 시간 기준 0–50초는 fitting, 51–60초는
validation, 마지막 14.724초는 held-out test다. iPhone/KM184의 시작 지연과 약한 clock
drift는 fitting 구간의 반복 xcorr 및 sinc resampling으로 보정했다.

이 데이터는 장비 전달함수만 담고 있지 않다. 캡슐 위치·지향성·방 반사·iPhone DSP·잡음이
같이 포함된다. 따라서 결과는 이 녹음쌍의 paired conversion 성능이며, 보편적 microphone
emulation의 증거는 아니다.

## 시도한 방법과 결과

| 방법 | 학습/설계 목적 | 주요 결과 | 판단 |
|---|---|---|---|
| NAM A2 Lite | 공식 A2 Lite submodel, MSE + 0.0005 MR-STFT | held-out ESR 0.4296 | Full보다 약함 |
| NAM A2 Full-500 | 공식 A2 Full submodel(23 dilated-conv 층, 8 channels), 500 epoch | held-out ESR **0.2049**, SNR **6.88 dB** | 파형 지표 최선 |
| 기존 `iphone2km184` FIR | 전체 mic-fr 공통 구간의 평활화 상대 스펙트럼을 보정 | mic-fr 70–4 kHz MAE 2.045 dB | 보수적 스펙트럼 기준선 |
| Hybrid FIR base v1 | 0–50초만 사용한 4,097-tap 70 Hz–4 kHz FIR | mic-fr 조건 MAE **1.824 dB** | 앱 후보로 보존 |
| Hybrid FIR + residual WaveNet | Hybrid FIR 뒤 16채널 WaveNet으로 잔차 학습. 무음 창 제외, band ESR/MR-STFT/relative-spectrum loss | 158 epoch까지 validation relative-spectrum MAE가 FIR-only **2.213 dB**를 넘지 못함 | residual 미채택 |

NAM Full은 waveform ESR에는 강하지만 상대 스펙트럼에서는 FIR보다 좋지 않았다. Hybrid FIR는
상대 스펙트럼에는 유리하지만 local held-out waveform ESR은 1.059로 좋지 않았다. 즉 현재
데이터에서는 하나의 모델이 두 평가를 동시에 지배하지 못했다.

## Hybrid residual 학습 조건

- 20 ms iPhone RMS frame gate: 95백분위 RMS보다 45 dB 낮은 값 및 quiet floor 1.5배 이상
- 8,192 sample window 중 활성 프레임이 25% 이상일 때만 batch 후보로 사용
- loss: full-band Smooth L1 + full MR-STFT + 70–4 kHz ESR + 70–4 kHz MR-STFT +
  500–2,000 Hz 중앙값 정규화 relative-spectrum L1
- checkpoint 선택: validation 70–4 kHz relative-spectrum MAE
- 안전장치: epoch 0의 FIR-only를 항상 후보 checkpoint로 보존

## 사전 진단: coherence와 레벨별 전달 특성

`diagnose_transfer.py`가 정렬된 신호에서 Welch magnitude-squared coherence와 RMS tier별
상대 스펙트럼을 계산한다. validation 51–60초의 중앙 coherence는 70–250 Hz 0.975,
250–500 Hz 0.931, 500–2,000 Hz 0.475, 2–4 kHz 0.533이었다. 따라서 저역·저중역은
상대적으로 안정적인 선형 전달관계가 있지만, 500 Hz 이상은 단일 고정 전달함수로 설명하기
어렵다. 이는 비선형성의 증거 자체는 아니며 위치·반사·정렬 잔차·iPhone DSP·낮은 SNR도
가능한 원인이다.

활성 validation STFT 프레임을 iPhone RMS로 low/mid/high 세 그룹으로 나누면 70–4 kHz
상대 스펙트럼 곡선 사이 MAE가 2.76–2.96 dB였다. 이 수치는 level-conditioned EQ 또는
dynamics를 시험할 근거는 되지만, 기타 연주 방식도 RMS와 함께 변하므로 현재 단일 연주만으로
레벨 의존 DSP를 확정할 수는 없다. 그래프와 수치는
`runs/diagnostics/coherence-level-validation.png`, `transfer-diagnostics.json`에 저장한다.

## Frequency-of-interest 정책

`frequency_policy.py`는 validation 데이터마다 1/6 옥타브별 처리 정책을 만든다. 입력
programme support, iPhone↔KM184 coherence, RMS tier별 전달곡선 spread를 동시에 사용한다.

| 정책 | 조건 | 처리 |
|---|---|---|
| `neural_residual` | support ≥ 0.45, coherence ≥ 0.75, tier spread ≤ 1.5 dB | 작은 residual controller/model을 허용 |
| `bounded_linear` | support ≥ 0.25지만 위 조건 전체는 미충족 | gain이 제한된 smooth FIR/IIR만 적용 |
| `conservative_reference` | 신호 지원이 낮음 | neural 생성 금지, boost 제한·완만한 reference 근사만 적용 |

현재 validation 녹음에서는 neural-residual 후보가 약 92–596 Hz에만 나타났고,
52 Hz–2.6 kHz는 주로 bounded-linear 후보였다. 이는 현재 기타 연주 한 파일에 대한
진단일 뿐이며, 앞으로 더 넓은 대역·다양한 주법·긴 녹음이 들어오면 매 데이터셋에서 다시
계산한다. `frequency-policy.json`, `frequency-policy-validation.png`를 학습 설정과 함께
저장해 어떤 대역에 neural model을 허용했는지 재현 가능하게 기록한다.

## 검토를 요청할 질문

1. paired acoustic recording에서 FIR 이후 남는 차이를 causal residual WaveNet이 학습할 수
   있는 결정론적 성분으로 볼 수 있는가, 아니면 위치·반사·DSP 변화가 지배적인가?
2. 목표가 70 Hz–4 kHz 음색 매칭이라면, neural residual보다 level-dependent parametric EQ,
   compressor, noise model을 분리한 구조가 더 적절한가?
3. 실시간 앱 목표에서 4,097-tap 선형위상 FIR의 약 42.7 ms delay를 유지할지,
   최소위상 IIR/짧은 FIR로 근사할지 어떤 청감·지연 trade-off가 타당한가?
4. 다음 데이터 수집에서 거리·각도·음량·연주를 어떻게 분리해야 실제 장비 변환과
   room/placement 변화를 구분할 수 있는가?

## 권장 다음 단계

새 신경망을 더 크게 만들기 전, Hybrid FIR base를 parametric IIR로 근사하고,
RMS-tier별 고정 EQ 또는 단순 보간 EQ를 먼저 비교한다. 그 뒤 서로 다른 연주와 거리에서
반복된 동시 녹음 paired dataset을 수집해, 고정 FIR, level-conditioned EQ/dynamics,
residual neural model을 같은 독립 test 세트에서 비교한다. 앱에는 우선 Hybrid FIR base v1을
calibration preset으로 두고, 실시간 지연 요구가 확정되면 IIR 근사를 별도 평가한다.
