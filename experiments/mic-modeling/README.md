# Nonlinear Microphone Conversion Model — iPhone → KM184

실험 A: 고정된 배치에서 같은 연주를 녹음한 iPhone을 KM184에 가깝게 변환합니다.
원본은 `/Users/sjchoi/dev/mic-fr/iphone.wav`, `km184.wav`이며 변경하지 않습니다.

> iPhone은 독립 레코더 동기 편차가 있어 mic-fr 원래 진단에서 4.8 ms의 창별 sync spread가
> 보입니다. 아래 결과는 같은 배치·같은 연주의 paired 변환 성능입니다. 다른 연주·거리·공간에
> 일반화된 마이크 특성으로 해석하려면 별도 동시 녹음 test가 필요합니다.

이름은 **Nonlinear Microphone Conversion Model(비선형 마이크 녹음 변환 모델)**로 부릅니다.
`nonlinear IR-conversion model`은 일반적인 IR 컨볼루션의 선형 시불변 가정과 혼동될 수 있고,
IR 파일을 입력·출력하는 모델로 읽힐 수도 있습니다. 본 실험은 IR을 추정하거나 변환하지 않고
파형 쌍으로 `ŷ[n] = fθ(x[n], …, x[n−6346])`를 학습합니다.
마이크 자체뿐 아니라 배치, 공간, 프리앰프, 휴대폰 DSP의 차이도 학습 대상에 섞여 있습니다.

## 레퍼런스와 실제 사용 범위

| 레퍼런스 | 이 실험에서의 역할 |
|---|---|
| [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler/tree/0072676419459f5d39e36f5b9fd4172f28d62cbf) | 실제 사용하는 WaveNet 구현, MR-STFT loss, `.nam` export |
| [공식 A2 Packed 설정](https://github.com/sdatkinson/neural-amp-modeler/blob/0072676419459f5d39e36f5b9fd4172f28d62cbf/nam/train/_resources/config_model_packed.json) | Lite·Full의 층, dilation, 채널 수, 활성함수 및 초기 optimizer 설정 출처 |
| [mic-fr 정렬 코드](https://github.com/shukjaechoi/mic-fr/blob/ace60033e2c5d93e3b0ee134a3e8d9babb8e2d4a/analyze.py) | 모노 변환·상관 기반 정렬·극성 확인 참고. 이번 실험은 원본에서 정렬을 다시 수행 |

공식 모델을 사용하되 실행 진입점은 `nam-full`이 아닌 이 폴더의 `train.py`입니다.
Packed 두 모델을 함께 학습하지 않고 독립적으로 학습하며, 스케줄러 gamma는 공식 기본값
0.994에서 0.98로 변경했습니다. 임의의 녹음 쌍에 맞춘 전처리·분할·평가도 이 실험 코드에 있습니다.

## 코드 구성과 학습 흐름

```text
iphone.wav + km184.wav
  → prepare.py: fitting 데이터로 정렬·정규화 계수 추정
  → data/aligned.npz + data/alignment.json
  → train.py: Lite / Full 학습 및 validation으로 best.pt 선택
  → baseline.py: FIR 및 게인 기준선 학습·선택
  → evaluate.py: 선택을 끝낸 모델의 마지막 구간 평가, WAV 출력
  → report.py: 비교표·그래프·음원 전환 페이지 생성
```

| 코드 / 파일 | 내용 |
|---|---|
| [prepare.py](prepare.py) | 입력 WAV 읽기, 모노화, 지연·극성·드리프트 보정, 데이터 분할 |
| [train.py](train.py) | 공식 A2 구성 로드, MPS 학습, early stopping, checkpoint 및 NAM 저장 |
| [baseline.py](baseline.py) | 선형 FIR의 tap 수·정규화 강도를 validation으로 선택 |
| [evaluate.py](evaluate.py) | test ESR·스펙트럼 오차 계산, `.nam` 가중치 복원 확인, 비교 음원 저장 |
| [convert.py](convert.py) | 학습된 모델과 기존 전처리 계수로 새 휴대폰 WAV 변환 |
| [report.py](report.py) | 학습 곡선과 테스트 스펙트럼, 로컬 비교 HTML 생성 |
| [requirements-lock.txt](requirements-lock.txt) | 이번 실행에서 사용한 Python 패키지 버전 |
| [provenance.json](provenance.json) | 이번 원본 데이터의 SHA256 및 참조 코드 커밋 |

학습에서는 입력 context 6,346샘플 뒤에 유효 출력 길이 8,192샘플을 붙여 batch를 만듭니다.
모델 출력과 같은 시점의 KM184를 비교하고 `MSE + 0.0005 × MR-STFT`를 역전파합니다.
epoch마다 51–60초 전체 validation ESR를 계산하며 test는 checkpoint 선택에 사용하지 않습니다.
`evaluate.py`가 저장하는 ESR는 오차 에너지 / 정답 에너지이고, 스펙트럼 지표는
512·1024·2048 FFT의 magnitude spectral convergence 평균입니다. 학습 MR-STFT loss와
보고하는 스펙트럼 지표는 같은 수치가 아닙니다.

## 데이터 입력 방법

현재 `prepare.py`는 **이번 녹음 쌍을 위한 고정 경로·분할 스크립트**입니다.
`--input` 또는 `--reference` CLI 옵션은 아직 없습니다. 현재 기대하는 파일 배치는 다음과 같습니다.

```text
/Users/sjchoi/dev/mic-fr/
├── iphone.wav   # 입력 x: 휴대폰 녹음
└── km184.wav    # 정답 y: 같은 연주의 레퍼런스 녹음
```

두 WAV는 같은 연주를 동시에 녹음한 것이어야 하며, 서로 다른 테이크는 정답 쌍으로 사용할 수
없습니다. 두 파일 모두 48,000 Hz여야 합니다. 코드가 채널 평균으로 모노화하므로 독립된 스테레오
공간감을 학습하는 구성은 아닙니다. 이번 원본은 16-bit PCM이지만 `soundfile`이 읽을 수 있는
다른 WAV 비트 깊이도 입력할 수 있습니다. 가능하면 녹음 설정과 게인을 고정하고 클리핑을 피합니다.

새 학습 데이터로 바꾸는 경우:

1. 기존 결과를 보존하려면 별도 실험 폴더를 사용합니다. `prepare.py`는 `data/`를,
   학습은 `runs/lite`·`runs/full`의 checkpoint를 덮어쓰며 history 로그는 이어 씁니다.
2. `prepare.py`의 `read()` 안 기본 디렉터리와 `main()`의 `read('iphone')`,
   `read('km184')`를 새 파일에 맞춥니다. `read()`가 `.wav` 확장자를 붙입니다.
3. `fit_end`, `splits`, 정렬 관측의 시간 범위를 녹음 길이에 맞춥니다. 현재는 validation이
   60초에 끝나고 마지막 15초가 test이므로 reference가 75초보다 길어야 합니다.
   단순히 두 파일을 교체하는 것만으로 임의 길이의 데이터셋을 지원하지는 않습니다.
4. `prepare.py` 실행 후 `data/alignment.json`의 관측, drift, 극성, split을 확인하고
   `data/validation_iphone.wav`와 `data/validation_km184.wav`를 비교합니다.
   기본 시작 지연 탐색 범위는 ±2초이며 이보다 큰 차이는 코드 조정이 필요합니다.
5. 새 원본의 경로·SHA256·사용 커밋을 별도 기록합니다. `provenance.json`은 이번 실행 기록이며
   `prepare.py`가 자동 갱신하지 않습니다. 그 뒤 학습 → 기준선 → 평가 순서로 실행합니다.

새 데이터의 test를 보고 모델이나 정렬 설정을 바꾸면 그 구간은 개발 데이터가 됩니다.
새로운 최종 test를 별도로 확보해야 합니다.

## 이번 실행 결과

| 테스트 출력 | ESR ↓ | SNR dB ↑ | MR spectral convergence ↓ |
|---|---:|---:|---:|
| 정렬·RMS 보정 iPhone | 1.13385 | -0.55 | 0.50409 |
| fitting에서 최소제곱 게인만 학습 | 0.89549 | 0.48 | 0.61524 |
| FIR (1024 taps, ridge 0.0001) | 0.40882 | 3.88 | **0.43551** |
| NAM A2 Lite | 0.42957 | 3.67 | 0.49408 |
| NAM A2 Full, 500 epoch run | **0.20493** | **6.88** | **0.30412** |
| Full 보정의 70 Hz–4 kHz blend | **0.20377** | **6.91** | **0.30229** |

500 epoch Full은 파형·전체 스펙트럼 지표 모두 가장 좋습니다. 그러나 mic-fr 방식의
상대 대역 비교에서는 80–250 Hz -8.8 dB, 250–500 Hz -5.2 dB, 2–5 kHz -2.7 dB,
8–16 kHz -7.1 dB로 KM184와의 차이가 남습니다. 따라서 낮은 ESR만으로 청감상 고급 마이크
복원이 달성됐다고 판단하면 안 됩니다.
학습 대상에 불확실한 위상·잡음 성분이 있으면 MSE가 출력 에너지를 줄이는 쪽으로
작용할 수 있지만, 여기서 원인을 분리 검증한 것은 아닙니다.

Lite 1,870 parameters / Full 12,145 parameters. 첫 실험은 각각 50 epoch 실행 후
Lite epoch 3 / Full epoch 1 checkpoint가 선택됐습니다. Lite는 train loss가 감소하는 동안
validation ESR가 0.5091에서 0.7488로 악화되어 validation 관점에서 과적합했습니다.

그 뒤 Full을 별도 run `full-500-global-xcorr`에서 500 epoch 학습했습니다. 이 run은 학습률
0.0005, gamma 0.994, early stopping 비활성화로 설정했고, 최선 checkpoint는 epoch 470,
validation ESR 0.33106입니다. 마지막 15초 홀드아웃 ESR는 0.20493, 상관계수는 0.89585,
MR spectral convergence는 0.30412입니다. 처음 Full보다 실제 test 수치도 개선됐습니다.
단일 녹음·단일 seed 결과이므로 일반화 성능을 뜻하지는 않습니다.

`report.py`는 [mic-fr](https://shukjaechoi.github.io/mic-fr/)의 상대 스펙트럼과 같은
median STFT power 방법으로 `runs/comparison/relative-spectrum.json`과 비교 페이지의 대역표를
생성합니다. 각 출력/KM184 power ratio의 500–2,000 Hz 중앙값을 0 dB로 맞추므로 절대 음량이
아닌 음색 차이를 봅니다. 이번 test에서 Full은 80–250 Hz -8.8 dB, 250–500 Hz -5.2 dB,
2–5 kHz -2.7 dB, 8–16 kHz -7.1 dB입니다.

70 Hz–4 kHz에만 관심을 둔 비교용 출력 `test_full_band_70_4k.wav`도 만듭니다. 식은
`iPhone + B(Full-500 − iPhone)`이며, `B`는 4,097-tap zero-phase FIR band-pass입니다.
그러므로 해당 대역에서는 NAM A2 Full의 보정을 반영하고, 대역 밖에서는 iPhone 원본을
유지합니다. 이 출력은 파형 비교와 청취를 위한 오프라인 blend이며, 인과적 실시간 FIR
배포 구현은 아닙니다.

`.nam`에 내보낸 가중치를 동일한 Python 아키텍처에 복원해 CPU와 MPS 출력을 비교했습니다.
최대 절대 오차는 Lite 3.28e-7 / Full 2.09e-7로 허용 오차 내였습니다.
`convert.py`로 원본 76초를 보존한 새 WAV 생성도 확인했습니다.

다음 실험에서는 별도 연주 데이터 확보, 잔여 정렬 오차 분석, 음량·고역 보존을 반영한
validation 기준을 먼저 개선하는 편이 타당합니다. 이번 테스트 결과를 보고 다시 조정한다면
그 테스트는 개발 데이터가 되므로 새로운 최종 test 녹음을 확보해야 합니다.

## 데이터 및 정렬

- 48 kHz, 16-bit, 정확히 76초. 두 파일 모두 좌우가 동일하므로 모노로 변환합니다.
- KM184 원본 시간 기준 **0–50초 fitting**, **51–60초 validation**, **61초 이후 test**.
  50–51초와 60–61초는 경계 누출 방지용 간격입니다. 앞쪽 1분 중 9초를 모델 선택에 사용합니다.
- 정렬 가능한 공통 구간은 75.72354초까지입니다. 따라서 마지막 15초 중 14.72354초를 평가합니다.
- 처음 50초 안의 상관 분석만으로 약 13,164.566샘플 오프셋, 약 20.149 ppm 상대 드리프트,
  극성 -1을 추정합니다. 장비 클록을 직접 측정한 값이 아닌 음원 기반 추정치입니다.
- 기타의 주기적 신호로 잘못된 상관 피크가 나오는 것을 줄이기 위해 극성을 고정하고,
  상관이 높은 관측에 Theil–Sen 회귀를 적용합니다. 64-tap windowed-sinc로 분수 샘플 정렬합니다.
- DC, RMS 입력 게인 및 공통 학습 스케일 역시 fitting 구간만으로 산출합니다.
- 테스트에서 추가 정렬, 게인 최적화, 모델 선택을 하지 않습니다.

`data/alignment.json`에 관측과 전처리 계수를, `provenance.json`에 원본 SHA256과
참조 저장소 커밋을 기록합니다. 과거 mic-fr 비교용 WAV는 전체 구간 레벨 보정이 있으므로
학습 데이터로 재사용하지 않고 원본에서 새로 전처리합니다.

## 모델 및 학습

공식 `nam/train/_resources/config_model_packed.json`의 두 submodel을 각각 독립적인
`nam.models.wavenet.WaveNet`으로 초기화합니다. Packed 동시 학습은 아닙니다.

| 설정 | Lite | Full |
|---|---:|---:|
| 내부 채널 | 3 | 8 |
| convolution 층 | 23 | 23 |
| receptive field | 6,347 samples | 6,347 samples |
| 시간 문맥 @48kHz | 132.229 ms | 132.229 ms |

- 공식 NAM 네트워크와 공식 MR-STFT 구현을 사용한 별도 PyTorch 학습 루프입니다.
- 네트워크와 역전파는 Apple M5 **MPS GPU**. MR-STFT 계산만 CPU에서 수행하며
  장치 복사에도 gradient는 유지됩니다. nam-cpu-trainer는 사용하지 않습니다.
- loss: MSE + 0.0005 × MR-STFT. 입력·목표에 같은 스케일을 적용해 목표 train RMS를 0.1로 설정합니다.
- Adam lr 0.004, weight decay 3.17e-7, ExponentialLR gamma 0.98,
  gradient clipping 1.0, seed 42.
- Batch 8, 유효 출력 8,192샘플 및 앞쪽 context 6,346샘플, epoch당 무작위 40 batch.
- 최대 120 epoch, 최소 50 epoch 후 검증 개선이 25 epoch 동안 없으면 종료합니다.
- validation ESR가 가장 낮은 checkpoint를 저장합니다. 전체 에너지로 정규화한 validation ESR입니다.
- FIR은 fitting에서 선형 최소제곱을 학습하고 tap 수와 ridge를 validation에서 선택합니다.

## 실행

Python 3.12와 macOS의 MPS 지원 PyTorch가 필요합니다. 아래는 저장소 루트에서 시작하는
최초 설치 순서입니다. `vendor/`와 `.venv/`는 Git에 포함되지 않으므로 새 checkout에서는
직접 준비해야 합니다. 이미 설치했다면 환경 설치를 건너뛰고 학습 명령으로 이동합니다.

```sh
cd experiments/mic-modeling
mkdir -p vendor
git clone https://github.com/sdatkinson/neural-amp-modeler.git vendor/neural-amp-modeler
git -C vendor/neural-amp-modeler checkout 0072676419459f5d39e36f5b9fd4172f28d62cbf
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install --no-deps -e vendor/neural-amp-modeler
.venv/bin/python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

`MPS: True`를 확인한 뒤 이 디렉터리에서 실행합니다. 학습기는 MPS가 없으면 오류를 내고
종료하며 CPU 학습으로 자동 전환하지 않습니다. 원본 음원은 저장소 다운로드에 포함되지 않습니다.

```sh
.venv/bin/python prepare.py
.venv/bin/python train.py --model lite
.venv/bin/python train.py --model full
.venv/bin/python baseline.py
.venv/bin/python evaluate.py
.venv/bin/python report.py
```

epoch 상한은 `--epochs 120`, epoch당 batch 수는 `--steps 40`, batch 크기는 `--batch 8`,
출력 길이는 `--length 8192`로 지정할 수 있습니다. 중단한 동일 실험은
`train.py --model lite --resume --epochs 120`으로 `last.pt`에서 이어갈 수 있습니다.
`--epochs`는 추가 횟수가 아니라 총 epoch 상한입니다. 재개할 때 데이터와 batch·길이 설정을
유지해야 하며, 현재 코드는 early-stopping의 연속 미개선 횟수까지 복원하지는 않습니다.

비교 페이지 실행:

```sh
.venv/bin/python -m http.server 8877 --bind 127.0.0.1 --directory runs/comparison
```

[http://127.0.0.1:8877](http://127.0.0.1:8877)에서 확인합니다. 로컬 서버가 실행 중이어야 합니다.

모델은 `runs/lite`, `runs/full`의 `best.pt`와 `.nam`에 저장됩니다.
`runs/comparison/index.html`은 테스트 WAV의 재생 위치를 유지하며 트랙을 전환합니다.
`complete_*.wav`에는 학습에 사용한 부분도 들어 있으므로 전체 파일을 일반화 평가에 쓰지 마십시오.

새 녹음 변환:

```sh
.venv/bin/python convert.py /absolute/phone.wav /absolute/converted.wav --model full
```

이 변환기는 원래 휴대폰의 타임라인을 유지합니다. 다른 녹음에 기존의 0.275초 시작 지연을
잘라내거나, reference와 맞추기 위한 드리프트 보정을 적용하지 않습니다.
같은 휴대폰 설정·레벨·배치를 가정하며 48 kHz 입력이 필요합니다.

`.nam` 단독 파일에는 Python 전처리가 포함되지 않습니다. 플러그인에서 사용하려면
입력에 극성 및 `input_gain × shared_scale`을 적용하고, 출력에 `1/shared_scale`을 적용해야 합니다.
DC 계수까지 동일하게 재현하려면 `convert.py`를 사용하십시오. `.nam` weight round-trip은
검증하지만 실제 LV2/NeuralAudio 플레이어에서의 로딩은 이 실험 범위에 포함하지 않습니다.

## 후속 실험: FIR + residual network

`train_hybrid.py`는 NAM A2를 그대로 내보내는 경로가 아니라, 이 실험을 위한 오프라인
`FIR + residual WaveNet`입니다. 첫 50초만 사용해 70 Hz–4 kHz의 4,097-tap 선형위상 FIR을
만들고, 16채널 WaveNet이 FIR 출력과 KM184 사이의 잔차를 예측합니다. 학습 창은 iPhone
입력의 20 ms RMS를 검사해, 95백분위 RMS보다 45 dB 낮은 기준 및 quiet-frame floor의 1.5배를
넘는 프레임이 25% 이상인 경우에만 뽑습니다. 따라서 무음 창은 학습에 들어가지 않되 조용한
연주는 남깁니다.

loss는 full-band Smooth L1, full-band MR-STFT, 70 Hz–4 kHz ESR/MR-STFT와 함께,
mic-fr 페이지와 같은 500–2,000 Hz 기준 정규화 후 70 Hz–4 kHz dB 오차를 직접 줄이는
relative-spectrum L1 항을 사용합니다. validation checkpoint는 이 상대 스펙트럼 MAE가 최소인
것으로 고르며, epoch 0의 FIR-only 상태도 후보로 저장합니다. 따라서 residual network가
스펙트럼을 악화시키면 FIR-only 결과보다 나쁜 checkpoint를 선택하지 않습니다.

## 해석 범위

테스트는 같은 76초 녹음 안의 마지막 구간입니다. 다른 곡·날짜·거리·휴대폰 설정에 대한
일반화는 검증하지 않았습니다. ESR은 시간·위상 차이에도 민감하며 청감 품질 자체가 아닙니다.
마이크 간 공간/위상 차이와 잡음 때문에 파형을 완벽하게 일치시키는 데 한계가 있습니다.
모델 크기가 크거나 학습을 오래 한다고 항상 개선되는 것은 아닙니다.
