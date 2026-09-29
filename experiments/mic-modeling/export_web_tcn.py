"""Export the selected Hybrid FIR + TCN128 checkpoint for browser inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/tcn-hybrid-128ch-band-residual-v3-relative-spectrum-200"
OUT = ROOT.parent.parent / "public/filters"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    config = json.loads((RUN / "config.json").read_text())
    calibration = json.loads((ROOT / "data/alignment.json").read_text())

    arrays: list[np.ndarray] = []
    offsets: dict[str, dict[str, int | list[int]]] = {}
    cursor = 0
    for name, tensor in checkpoint["state"].items():
        values = tensor.detach().cpu().numpy().astype("<f4", copy=False).ravel()
        offsets[name] = {"offset": cursor, "length": int(values.size), "shape": list(tensor.shape)}
        arrays.append(values)
        cursor += values.size

    # The training code's smooth FFT mask: 50–70 Hz and 4–4.25 kHz cosine ramps.
    nfft = 8192
    freq = np.fft.rfftfreq(nfft, 1 / 48000)
    low = np.clip((freq - 50) / 20, 0, 1)
    high = np.clip((4250 - freq) / 250, 0, 1)
    ramp = lambda value: .5 - .5 * np.cos(np.pi * value)
    response = ramp(low) * ramp(high)
    periodic_impulse = np.fft.irfft(response, n=nfft)
    residual_band = np.array([periodic_impulse[lag % nfft] for lag in range(-4095, 4096)], dtype="<f4")
    band_offset = cursor
    arrays.append(residual_band)
    cursor += residual_band.size

    (OUT / "iphone-hybrid-tcn128.f32le").write_bytes(np.concatenate(arrays).astype("<f4", copy=False).tobytes())
    manifest = {
        "sampleRate": 48000,
        "channels": config["channels"],
        "layers": 10,
        "receptiveFieldSamples": config["receptive_field_samples"],
        "residualScale": config["residual_scale"],
        "selectedEpoch": checkpoint["epoch"],
        "validationRelativeSpectrumMaeDb": checkpoint["val_relative_spectral_mae_db"],
        "floatCount": cursor,
        "weights": offsets,
        "residualBand": {"offset": band_offset, "length": int(residual_band.size)},
        "calibration": {key: calibration[key] for key in ("polarity", "input_dc", "input_gain", "shared_scale", "target_dc")},
    }
    (OUT / "iphone-hybrid-tcn128.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {cursor:,} float32 values ({(cursor * 4) / 1024 / 1024:.2f} MiB), epoch {checkpoint['epoch']}.")


if __name__ == "__main__":
    main()
