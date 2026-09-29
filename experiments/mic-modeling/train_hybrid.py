"""Train an offline FIR + residual temporal model for iPhone → KM184.

The FIR is fitted only on the first 50 s and handles the stable, linear
70 Hz–4 kHz coloration.  A larger WaveNet predicts only what remains.  This is
deliberately not a `.nam` model: the final processor is FIR followed by a
residual network, selected by validation relative-spectrum error.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from nam.models.wavenet import WaveNet
from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss

from train import ROOT, bandpass, predict, relative_spectral_mae_db

FS = 48000


def median_power(x: np.ndarray, nfft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    frames = np.lib.stride_tricks.sliding_window_view(x, nfft)[:: nfft // 2]
    power = np.abs(np.fft.rfft(frames * np.hanning(nfft), axis=1)) ** 2
    level = np.mean(frames**2, axis=1)
    active = level > max(np.quantile(level, 0.3), 1e-9)
    return np.median(power[active], axis=0), np.fft.rfftfreq(nfft, 1 / FS)


def design_fir(x: np.ndarray, y: np.ndarray, taps: int = 4097) -> np.ndarray:
    """Magnitude-only, phase-linear FIR fitted exclusively from training audio."""
    px, f = median_power(x)
    py, _ = median_power(y)
    target_db = 10 * np.log10(np.maximum(py, 1e-20) / np.maximum(px, 1e-20))
    # Smooth one sixth octave to avoid inverting room-placement combing.
    centers = np.geomspace(40, 18000, 160)
    smoothed = []
    for c in centers:
        use = (f >= c * 2 ** (-1 / 12)) & (f <= c * 2 ** (1 / 12))
        smoothed.append(np.median(target_db[use]))
    lookup = np.interp(np.log(np.maximum(f, 40)), np.log(centers), smoothed)
    # Preserve signal outside the trustworthy band with gentle transitions.
    low = np.clip((f - 50) / 20, 0, 1)
    high = np.clip((4250 - f) / 250, 0, 1)
    taper = (low * low * (3 - 2 * low)) * (high * high * (3 - 2 * high))
    amplitude = 10 ** (np.clip(lookup, -8, 10) * taper / 20)
    return signal.firwin2(taps, f, amplitude, fs=FS, window=("kaiser", 8.6)).astype("float32")


def fir_apply(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    # Centered, linear-phase offline processing. The neural residual is causal;
    # its sum with this base is evaluated at the same sample locations.
    return signal.fftconvolve(x, taps, mode="same").astype("float32")


def active_window_starts(x: np.ndarray, fit_end: int, length: int, rf: int) -> tuple[np.ndarray, dict]:
    """Return training starts whose *input* contains enough programme signal.

    The threshold is 45 dB below the input's 95th-percentile short-term RMS,
    but never below 1.5× its quiet-frame floor.  A window needs 25% active
    20 ms frames; this rejects silence while retaining quiet playing.
    """
    frame = 960  # 20 ms at 48 kHz
    frames = x[:fit_end][:fit_end // frame * frame].reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    quiet_floor = float(np.quantile(frame_rms, .10))
    reference = float(np.quantile(frame_rms, .95))
    threshold = max(quiet_floor * 1.5, reference * 10 ** (-45 / 20))
    starts = np.arange(rf - 1, fit_end - length, 256)
    valid = []
    for start in starts:
        portion = x[start:start + length]
        rms = np.sqrt(np.mean(portion[:len(portion) // frame * frame].reshape(-1, frame)**2, axis=1))
        if np.mean(rms >= threshold) >= .25:
            valid.append(start)
    if not valid:
        raise RuntimeError("Activity gate excluded every training window")
    return np.asarray(valid), dict(frame_ms=20, threshold_rms=threshold,
                                   threshold_db_below_p95=-45, min_active_fraction=.25,
                                   candidate_stride_samples=256, candidates=len(starts), accepted=len(valid))


def relative_band_spectrum_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable counterpart of the page's mid-normalised spectral curve."""
    n=pred.shape[-1]
    f=torch.fft.rfftfreq(n, 1 / FS, device=pred.device)
    ratio=20*torch.log10(torch.fft.rfft(pred,dim=-1).abs().clamp_min(1e-7) /
                           torch.fft.rfft(target,dim=-1).abs().clamp_min(1e-7))
    mid=(f>=500)&(f<=2000)
    ratio=ratio-ratio[...,mid].median(dim=-1,keepdim=True).values
    use=(f>=70)&(f<=4000)
    return ratio[...,use].abs().mean()


def model_config(channels: int) -> dict:
    source = ROOT / "vendor/neural-amp-modeler/nam/train/_resources/config_model_packed.json"
    cfg = json.loads(source.read_text())["net"]["config"]["submodels"][1]["config"]
    cfg["layers_configs"][0]["channels"] = channels
    cfg["sample_rate"] = FS
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="hybrid-fir-residual-16ch")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=.994)
    args = p.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this experiment")
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    out = ROOT / "runs" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((ROOT / "data/alignment.json").read_text())
    d = np.load(ROOT / "data/aligned.npz")
    fit_end = info["splits"]["fit"][1]
    x, y = d["x"], d["y"]
    taps = design_fir(x[:fit_end], y[:fit_end])
    base = fir_apply(x, taps)
    np.save(out / "fir_70_4k_train_only.npy", taps)
    config = model_config(args.channels)
    net = WaveNet.init_from_config(config).to("mps")
    # The base FIR is a valid processor on its own. Start exactly there rather
    # than letting random residual output damage its measured spectrum.
    with torch.no_grad():
        net._net._layer_arrays[0]._head_rechannel.weight.zero_()
        net._net._layer_arrays[0]._head_rechannel.bias.zero_()
    rf = net.receptive_field
    active_starts, activity_gate = active_window_starts(x, fit_end, args.length, rf)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    mrstft = MultiResolutionSTFTLoss()
    va, vb = info["splits"]["validation"]
    baseline_residual=predict(net,x[va-rf+1:vb])[rf-1:]
    baseline_prediction=base[va:vb]+baseline_residual
    baseline_esr=float(np.mean((baseline_prediction-y[va:vb])**2)/np.mean(y[va:vb]**2))
    best=relative_spectral_mae_db(baseline_prediction,y[va:vb])
    meta = dict(kind="offline FIR + residual WaveNet", fir_taps=len(taps), fir_fit_seconds=[0,50],
                trusted_band_hz=[70,4000], channels=args.channels, receptive_field=rf,
                input_activity_gate=activity_gate,
                selection="minimum validation 70 Hz–4 kHz relative-spectrum MAE (dB); no test access",
                loss="SmoothL1 full waveform + 0.002 full MRSTFT + 0.005 band ESR + 0.01 band MRSTFT + 0.02 relative-band spectral L1")
    (out / "config.json").write_text(json.dumps({"args":vars(args), **meta}, indent=2))
    began=time.time()
    torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()}, config=config,
                    epoch=0,val_esr=baseline_esr,val_relative_spectral_mae_db=best),out/"best.pt")
    initial=dict(epoch=0,loss=None,val_esr=baseline_esr,val_relative_spectral_mae_db=best,
                 best_selection=best,seconds=0.0,lr=args.lr,note="zero residual / FIR-only baseline")
    print(json.dumps(initial),flush=True)
    with (out/"history.jsonl").open("w") as f: f.write(json.dumps(initial)+"\n")
    for epoch in range(args.epochs):
        net.train(); losses=[]
        for _ in range(args.steps):
            starts=rng.choice(active_starts, size=args.batch, replace=True)
            xx=torch.from_numpy(np.stack([x[s-rf+1:s+args.length] for s in starts])).to("mps")
            yy=torch.from_numpy(np.stack([y[s:s+args.length] for s in starts])).to("mps")
            bb=torch.from_numpy(np.stack([base[s:s+args.length] for s in starts])).to("mps")
            optimizer.zero_grad(set_to_none=True)
            pred=bb+net(xx,pad_start=False)
            smooth=torch.nn.functional.smooth_l1_loss(pred,yy,beta=.02)
            full_spec=mrstft(pred[:,None].cpu(),yy[:,None].cpu()).to("mps")
            bp,by=bandpass(pred.cpu()),bandpass(yy.cpu())
            band_esr=((bp-by).square().mean(dim=1)/by.square().mean(dim=1).clamp_min(1e-12)).mean().to("mps")
            band_spec=mrstft(bp[:,None],by[:,None]).to("mps")
            relative_spec=relative_band_spectrum_loss(pred.cpu(),yy.cpu()).to("mps")
            loss=smooth+.002*full_spec+.005*band_esr+.01*band_spec+.02*relative_spec
            if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        # Validation has preceding context for the causal residual network.
        residual=predict(net,x[va-rf+1:vb])[rf-1:]
        prediction=base[va:vb]+residual
        esr=float(np.mean((prediction-y[va:vb])**2)/np.mean(y[va:vb]**2))
        spectral_mae=relative_spectral_mae_db(prediction,y[va:vb])
        if spectral_mae < best:
            best=spectral_mae
            torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()}, config=config,
                            epoch=epoch+1,val_esr=esr,val_relative_spectral_mae_db=spectral_mae),out/"best.pt")
        scheduler.step()
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),val_esr=esr,
                 val_relative_spectral_mae_db=spectral_mae,best_selection=best,
                 seconds=round(time.time()-began,1),lr=scheduler.get_last_lr()[0])
        print(json.dumps(row),flush=True)
        with (out/"history.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
    print("COMPLETE",json.dumps(torch.load(out/"best.pt",map_location="cpu",weights_only=False),default=str)[:300])


if __name__ == "__main__":
    main()
