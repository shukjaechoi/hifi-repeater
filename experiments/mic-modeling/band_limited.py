"""Blend Full-500 correction only into the trusted 70 Hz–4 kHz band."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

from evaluate import metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs' / 'comparison'
LOW_HZ, HIGH_HZ = 70.0, 4000.0


def correction_band(x, sample_rate):
    """Offline zero-phase FIR bandpass for analysis and listening comparison."""
    taps = signal.firwin(4097, [LOW_HZ, HIGH_HZ], pass_zero=False, fs=sample_rate)
    return signal.filtfilt(taps, [1.0], x)


def write_waveform_figure(phone, full, blended, km184, sample_rate):
    # Locate an active one-second test passage, then show its first 40 ms in detail.
    window = sample_rate
    energy = np.convolve(km184 ** 2, np.ones(window) / window, mode='valid')
    start = int(np.argmax(energy))
    one_second = slice(start, start + window)
    zoom = slice(start, start + int(.04 * sample_rate))
    tracks = {'iPhone': phone, 'Full-500': full, '70 Hz–4 kHz blend': blended, 'KM184': km184}
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    for label, track in tracks.items():
        axes[0].plot(np.arange(window) / sample_rate, track[one_second], linewidth=.8, label=label)
        axes[1].plot(np.arange(zoom.stop - zoom.start) / sample_rate * 1000, track[zoom], linewidth=.9, label=label)
    axes[0].set(title='Most active 1 s in held-out test', xlabel='Seconds from selected passage', ylabel='Amplitude')
    axes[1].set(title='First 40 ms of that passage', xlabel='Milliseconds', ylabel='Amplitude')
    for axis in axes:
        axis.axhline(0, color='black', linewidth=.5)
        axis.legend(loc='upper right')
    fig.savefig(OUT / 'waveform-band-limited.png', dpi=160)


def main():
    phone, sample_rate = sf.read(OUT / 'test_iphone.wav')
    full, full_rate = sf.read(OUT / 'test_full.wav')
    km184, target_rate = sf.read(OUT / 'test_km184.wav')
    assert sample_rate == full_rate == target_rate == 48000
    blended = phone + correction_band(full - phone, sample_rate)
    sf.write(OUT / 'test_full_band_70_4k.wav', blended, sample_rate, subtype='PCM_24')
    result = json.loads((OUT / 'metrics.json').read_text())
    result['metrics']['full_band_70_4k'] = metrics(blended, km184)
    result['band_limited_blend'] = {
        'method': 'iPhone + zero-phase FIR bandpass(Full-500 − iPhone)',
        'band_hz': [LOW_HZ, HIGH_HZ],
        'fir_taps': 4097,
        'scope': 'Offline analysis/listening only; zero-phase filtering is non-causal.',
    }
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2))
    write_waveform_figure(phone, full, blended, km184, sample_rate)
    print(json.dumps({'full': result['metrics']['full'], 'full_band_70_4k': result['metrics']['full_band_70_4k']}, indent=2))


if __name__ == '__main__':
    main()
