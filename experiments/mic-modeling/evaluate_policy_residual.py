"""Evaluate the selected policy-gated residual TCN once on the held-out tail."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from evaluate import metrics
from train_hybrid import fir_apply
from train_policy_residual import ConditionedTCN, envelope

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/policy-conditioned-tcn32-fir-v1"
OUT = ROOT / "runs/comparison"
FS = 48000


def gated_residual(residual, policy_mask):
    """Apply the FIT-derived mask at the FFT resolution of this signal."""
    frequencies = np.fft.rfftfreq(len(residual), 1 / FS)
    source_frequencies = np.fft.rfftfreq((len(policy_mask) - 1) * 2, 1 / FS)
    mask = np.interp(frequencies, source_frequencies, policy_mask)
    return np.fft.irfft(np.fft.rfft(residual) * mask, n=len(residual))


def main():
    info = json.loads((ROOT / "data/alignment.json").read_text())
    data = np.load(ROOT / "data/aligned.npz")
    x, y = data["x"], data["y"]
    checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    taps = np.load(RUN / "fir.npy")
    base = fir_apply(x, taps)

    model = ConditionedTCN(32)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    inputs = np.stack([x, base, envelope(x)])[None]
    with torch.no_grad():
        residual = model(torch.from_numpy(inputs)).numpy()[0]
    prediction = base + gated_residual(residual, np.load(RUN / "neural_policy_mask_8192.npy"))

    a, b = info["splits"]["test"]
    OUT.mkdir(exist_ok=True)
    for prefix, z in [("test", prediction[a:b]), ("complete", prediction)]:
        sf.write(OUT / f"{prefix}_policy_conditioned_tcn32.wav", z / info["shared_scale"], 48000, subtype="PCM_24")

    result = json.loads((OUT / "metrics.json").read_text())
    result["metrics"]["policy_conditioned_tcn32"] = metrics(prediction[a:b], y[a:b])
    config = json.loads((RUN / "config.json").read_text())
    result["policy_conditioned_tcn32"] = dict(
        run=RUN.name,
        selected_epoch=checkpoint["epoch"],
        selected_validation_esr=checkpoint["val_esr"],
        selected_validation_relative_spectral_mae_db=checkpoint["val_relative_spectral_mae_db"],
        architecture=config,
        note="FIR baseline plus a TCN residual limited to the FIT-derived policy band."
    )
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"test": result["metrics"]["policy_conditioned_tcn32"], "selection": result["policy_conditioned_tcn32"]}, indent=2))


if __name__ == "__main__":
    main()
