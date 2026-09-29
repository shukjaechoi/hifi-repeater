"""Render static experiment figures and a local synchronized listening comparison."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mic-mpl')
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/comparison'


def programme_spectrum(x, sample_rate):
    """Same programme-dependent median STFT estimate used by mic-fr/analyze.py."""
    nfft, hop = 8192, 4096
    frames = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hanning(nfft), axis=1)) ** 2
    frame_level = np.mean(frames ** 2, axis=1)
    active = frame_level > max(np.quantile(frame_level, .3), 1e-9)
    return np.median(power[active], axis=0), np.fft.rfftfreq(nfft, 1 / sample_rate)


def relative_spectra(labels):
    tracks = {name: sf.read(OUT / f'test_{name}.wav')[0] for name in labels}
    sample_rate = sf.info(OUT / 'test_km184.wav').samplerate
    spectra = {name: programme_spectrum(x, sample_rate)[0] for name, x in tracks.items()}
    _, freq = programme_spectrum(tracks['km184'], sample_rate)
    mid = (freq >= 500) & (freq <= 2000)
    bands = [('bass', '80–250 Hz', 80, 250), ('low_mid', '250–500 Hz', 250, 500),
             ('presence', '2–5 kHz', 2000, 5000), ('air', '8–16 kHz', 8000, 16000)]
    output = {}
    for name, spec in spectra.items():
        curve = 10 * np.log10(np.maximum(spec, 1e-20) / np.maximum(spectra['km184'], 1e-20))
        curve -= np.median(curve[mid])
        output[name] = dict(
            curve_db=curve.tolist(),
            bands_db={key: float(np.median(curve[(freq >= low) & (freq < high)]))
                      for key, _, low, high in bands},
        )
    payload = dict(
        method='Median 8192-point Hann STFT power over active test frames; each ratio is normalized to 0 dB median from 500–2000 Hz.',
        test_seconds=float(len(tracks['km184']) / sample_rate),
        frequencies_hz=freq.tolist(),
        bands=[dict(key=key, label=label) for key, label, _, _ in bands],
        tracks=output,
    )
    (OUT / 'relative-spectrum.json').write_text(json.dumps(payload, indent=2))
    return payload

def main():
    result=json.loads((OUT/'metrics.json').read_text())
    labels={'iphone':'입력 · iPhone (정렬/레벨 보정)','gain_only':'입력 · iPhone (최소제곱 게인)',
            'fir':'선형 기준 · 1,024-tap FIR (파형 적합)',
            'lite':'NAM A2 Lite · 1,870 params','full':'NAM A2 Full-500 · epoch 470',
            'full_band_70_4k':'NAM Full-500 · 70 Hz–4 kHz 보정만',
            'hybrid_fir_residual':'Hybrid 기준 · train-only 4,097-tap FIR (epoch 0 선택)',
            'policy_conditioned_tcn32':'정책 잔차 TCN32 · FIT 근거 대역만',
            'km184':'정답 · KM184'}
    relative = relative_spectra(labels)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    model_runs={'lite':'lite', 'full':result.get('full_run', 'full')}
    for name, run in model_runs.items():
        rows=[json.loads(s) for s in (ROOT/f'runs/{run}/history.jsonl').read_text().splitlines()]
        axs[0].plot([r['epoch'] for r in rows],[r['val_esr'] for r in rows],label=name)
    base=json.loads((ROOT/'runs/baseline.json').read_text())
    axs[0].axhline(base['best']['val_esr'],color='gray',linestyle='--',label='FIR')
    axs[0].set(xlabel='Epoch',ylabel='Validation ESR (lower is better)',title='Model selection (51–60 s)')
    axs[0].legend()
    freq=np.asarray(relative['frequencies_hz'])
    for name in ['iphone','fir','lite','full','full_band_70_4k','hybrid_fir_residual','policy_conditioned_tcn32']:
        axs[1].semilogx(freq, relative['tracks'][name]['curve_db'], label=name)
    axs[1].axhline(0, color='black', linewidth=.8, label='KM184 (0 dB)')
    axs[1].set(xlim=(40,18000),ylim=(-25,25),xlabel='Hz',ylabel='Relative dB',
               title='Held-out relative spectrum (mid-band normalized)')
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png',dpi=160)
    rows=''.join(f"<tr><td>{labels[k]}</td><td>{v['esr']:.4f}</td><td>{v['snr_db']:.2f}</td><td>{v['mr_spectral_convergence']:.4f}</td></tr>" for k,v in result['metrics'].items())
    band_rows=''.join(
        f"<tr><td>{labels[name]}</td>" + ''.join(
            f"<td>{relative['tracks'][name]['bands_db'][band['key']]:+.1f} dB</td>" for band in relative['bands']) + '</tr>'
        for name in ['iphone','fir','lite','full','full_band_70_4k','hybrid_fir_residual','policy_conditioned_tcn32'])
    band_head=''.join(f"<th>{band['label']}</th>" for band in relative['bands'])
    buttons=''.join(f'<button data-track="{k}">{v}</button>' for k,v in labels.items())
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>iPhone → KM184 · NAM A2</title>
<style>body{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 24px;background:#10151c;color:#e5ecf4}h1{font-size:32px}p{color:#b6c4d4;line-height:1.7}button{padding:12px;margin:5px;border:1px solid #607086;border-radius:8px;background:#202b38;color:white;cursor:pointer}button.active{background:#376353}audio{width:100%;margin:20px 0}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{text-align:left;padding:12px;border-bottom:1px solid #384454}img{width:100%;background:white;border-radius:12px}small{color:#a9bacd}</style>
<h1>iPhone → KM184</h1><p>동일 연주 · 선형 FIR · NAM A2 · Hybrid FIR + residual 비교</p>
<p><strong>현재 선택 모델: Full, 500 epoch 실행 중 epoch 470 checkpoint.</strong> 이 run은 lr 0.0005와 gamma 0.994를 사용했습니다. iPhone에는 독립 레코더 동기 편차가 있으므로, 이 결과는 같은 배치·같은 연주의 paired 변환 성능이며 다른 녹음 환경에 대한 일반화는 아직 확인하지 않았습니다.</p>
<p>테스트: 원본 KM184 61초부터 공통 녹음 끝까지, 14.724초. 마지막 약 0.276초는 iPhone의 대응 녹음이 없어 제외했습니다. 버튼을 누르면 같은 재생 위치에서 소리를 바꿉니다. 모든 트랙은 같은 기준 레벨이며 테스트 구간별 음량 보정은 하지 않았습니다.</p>
<p><strong>이름 안내:</strong> “선형 기준”은 1,024-tap 파형 적합 FIR이고, “Hybrid 기준”은 첫 50초로만 만든 4,097-tap 70 Hz–4 kHz FIR입니다. Hybrid residual network는 이 FIR 뒤의 잔차를 학습했지만 validation 상대 스펙트럼을 개선하지 못해 epoch 0, 즉 FIR-only checkpoint가 선택돼 있습니다. “정책 잔차 TCN32”는 raw iPhone, Hybrid FIR 출력, envelope를 입력으로 받는 32채널·10층 causal TCN입니다. FIT 구간의 programme support, coherence, level-tier 안정성이 모두 통과한 약 92–596 Hz에만 residual을 적용합니다. selected epoch 3의 held-out 상대 스펙트럼 MAE는 Hybrid 기준 2.874 dB에서 2.846 dB로 0.028 dB만 낮아졌습니다.</p>
<div>BUTTONS</div><audio controls id="player" src="test_iphone.wav"></audio><label><input type="checkbox" id="loop"> 반복</label>
<table><thead><tr><th>모델</th><th>ESR ↓</th><th>SNR dB ↑</th><th>MR spectral convergence ↓</th></tr></thead><tbody>METRIC_ROWS</tbody></table>
<h2>상대 스펙트럼: KM184와의 차이</h2><p>mic-fr와 같은 median STFT power 방식입니다. 각 트랙을 KM184 대비로 계산한 뒤 500–2,000 Hz 중앙값을 0 dB로 맞췄습니다. 따라서 표는 절대 음량이 아니라 대역별 음색 차이입니다. 0 dB에 가까울수록 KM184의 상대 스펙트럼과 가깝습니다.</p>
<table><thead><tr><th>모델</th>BAND_HEAD</tr></thead><tbody>BAND_TABLE_ROWS</tbody></table>
<h2>파형 비교</h2><p>70 Hz–4 kHz blend는 iPhone에 Full 출력과 iPhone의 차이만 FIR 대역통과해 더한 결과입니다. 현재 비교용 zero-phase FIR은 오프라인 분석용이며 실시간 배포용 필터가 아닙니다.</p><img src="waveform-band-limited.png" alt="KM184, Full, 70 Hz–4 kHz blend 파형 비교">
<img src="comparison.png" alt="검증 ESR 및 KM184 기준 상대 스펙트럼"><p>학습 0–50초 · 검증 51–60초 · 테스트 61초 이후. 같은 녹음 안의 홀드아웃으로, 다른 연주·배치에 대한 일반화는 아직 확인하지 않았습니다.</p>
<script>const a=document.querySelector('audio');let request=0;document.querySelectorAll('[data-track]').forEach(b=>b.onclick=()=>{const t=a.currentTime,playing=!a.paused,r=++request;a.pause();a.src='test_'+b.dataset.track+'.wav';a.onloadedmetadata=()=>{if(r!==request)return;a.currentTime=Math.min(t,a.duration);if(playing)a.play()};document.querySelectorAll('button').forEach(c=>c.classList.toggle('active',c===b))});document.querySelector('[data-track="iphone"]').classList.add('active');document.querySelector('#loop').onchange=e=>a.loop=e.target.checked;</script></html>'''
    (OUT/'index.html').write_text(html.replace('BUTTONS',buttons).replace('METRIC_ROWS',rows).replace('BAND_HEAD',band_head).replace('BAND_TABLE_ROWS',band_rows))
if __name__=='__main__': main()
