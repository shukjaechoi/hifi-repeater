"""Policy-gated conditioned residual TCN on top of Hybrid FIR base.

Unlike the earlier residual WaveNet, this network receives raw iPhone, FIR
output, and a smoothed envelope. Its residual is allowed only in frequency
bands supported by FIT-split evidence; all other frequencies stay on the FIR
path exactly.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy import signal
from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss

from train import ROOT, relative_spectral_mae_db
from train_hybrid import active_window_starts, design_fir, fir_apply
from frequency_policy import active_support, tier_spread, smooth_log

FS=48000

def policy_mask(x,y,n):
    """Turn fit evidence into smooth 1/6-octave frequency regions.

    Thresholding raw FFT bins made the first policy mask sparse and dependent
    on individual bins.  The policy is intentionally decided on log-spaced
    1/6-octave bands, then each accepted band gets cosine skirts.
    """
    f,sup=active_support(x); fc,coh=signal.coherence(x,y,fs=FS,nperseg=8192,noverlap=4096,window='hann'); fs,spread=tier_spread(x,y)
    centers=np.geomspace(40,18000,96)
    support=smooth_log(f,sup,centers); c=smooth_log(fc,coh,centers); s=smooth_log(fs,spread,centers)
    accepted=centers[(support>=.45)&(c>=.75)&(s<=1.5)]
    freq=np.fft.rfftfreq(n,1/FS); allowed=np.zeros_like(freq)
    for center in accepted:
        lo,hi=center*2**(-1/12),center*2**(1/12)
        edge_lo,edge_hi=center*2**(-1/8),center*2**(1/8)
        ramp_up=np.clip((freq-edge_lo)/(lo-edge_lo),0,1)
        ramp_down=np.clip((edge_hi-freq)/(edge_hi-hi),0,1)
        allowed=np.maximum(allowed,(.5-.5*np.cos(np.pi*ramp_up))*(.5-.5*np.cos(np.pi*ramp_down)))
    return allowed,dict(neural_band_centers_hz=accepted.tolist(),min_support=.45,min_coherence=.75,max_tier_spread_db=1.5)

def envelope(x,window=960):
    kernel=np.ones(window,dtype=np.float32)/window
    return np.sqrt(np.maximum(signal.fftconvolve(x*x,kernel,mode='same'),1e-10)).astype('float32')

class CausalBlock(nn.Module):
    def __init__(self,ch,dilation):
        super().__init__(); self.pad=2*dilation
        self.conv=nn.Conv1d(ch,ch,3,dilation=dilation); self.mix=nn.Conv1d(ch,ch,1)
    def forward(self,x):
        y=self.conv(nn.functional.pad(x,(self.pad,0))); return x+torch.tanh(self.mix(torch.tanh(y)))

class ConditionedTCN(nn.Module):
    def __init__(self,ch=32):
        super().__init__(); self.input=nn.Conv1d(3,ch,1); self.blocks=nn.ModuleList(CausalBlock(ch,2**i) for i in range(10)); self.head=nn.Conv1d(ch,1,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,x):
        h=torch.tanh(self.input(x))
        for b in self.blocks: h=b(h)
        return self.head(h).squeeze(1)

def gate_frequency(x,mask):
    return torch.fft.irfft(torch.fft.rfft(x,dim=-1)*mask,n=x.shape[-1],dim=-1)

def main():
    run='policy-conditioned-tcn32-fir-v2-smooth-bands'; out=ROOT/'runs'/run; out.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS required')
    info=json.loads((ROOT/'data/alignment.json').read_text());d=np.load(ROOT/'data/aligned.npz');x,y=d['x'],d['y']; fit=info['splits']['fit'][1]; va,vb=info['splits']['validation']
    taps=design_fir(x[:fit],y[:fit]);base=fir_apply(x,taps); env=envelope(x)
    mask_np,policy=policy_mask(x[:fit],y[:fit],8192); mask=torch.from_numpy(mask_np.astype('float32')).to('mps')
    net=ConditionedTCN(32).to('mps'); opt=torch.optim.AdamW(net.parameters(),lr=4e-4,weight_decay=1e-6); sched=torch.optim.lr_scheduler.ExponentialLR(opt,.994); stft=MultiResolutionSTFTLoss()
    length=8192; rf=2047; starts,gate=active_window_starts(x,fit,length,rf); rng=np.random.default_rng(42); best=float('inf')
    def inputs(a,b): return np.stack([x[a:b],base[a:b],env[a:b]])
    # Epoch zero protects the FIR baseline under the policy used for training.
    def evaluate():
        with torch.no_grad():
            q=torch.from_numpy(inputs(va-rf+1,vb)[None]).to('mps'); r=net(q)[:,rf-1:].cpu().numpy()[0]
        eval_mask=np.interp(np.fft.rfftfreq(len(r),1/FS),np.fft.rfftfreq(8192,1/FS),mask_np)
        pred=base[va:vb]+np.fft.irfft(np.fft.rfft(r)*eval_mask,n=len(r))
        return pred,float(np.mean((pred-y[va:vb])**2)/np.mean(y[va:vb]**2),),relative_spectral_mae_db(pred,y[va:vb])
    pred,esr,best=evaluate(); torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},epoch=0,val_esr=esr,val_relative_spectral_mae_db=best),out/'best.pt')
    np.save(out/'fir.npy',taps);np.save(out/'neural_policy_mask_8192.npy',mask_np)
    meta=dict(kind='FIR + policy-gated conditioned residual TCN',channels=32,causal_layers=10,receptive_field_samples=rf,fit_policy=policy,input_activity_gate=gate,selected_by='validation 70–4 kHz relative spectrum MAE')
    (out/'config.json').write_text(json.dumps(meta,indent=2)); history=[]
    for epoch in range(100):
        net.train(); losses=[]
        for _ in range(40):
            s=rng.choice(starts,size=8); xx=torch.from_numpy(np.stack([inputs(i-rf+1,i+length) for i in s])).to('mps'); yy=torch.from_numpy(np.stack([y[i:i+length] for i in s])).to('mps'); bb=torch.from_numpy(np.stack([base[i:i+length] for i in s])).to('mps')
            opt.zero_grad(set_to_none=True); residual=gate_frequency(net(xx)[:,rf-1:],mask); pred=bb+residual
            # Spectral metric only scores the policy residual band; FIR governs elsewhere.
            loss=nn.functional.smooth_l1_loss(pred,yy,beta=.02)+.005*stft(pred[:,None].cpu(),yy[:,None].cpu()).to('mps')+.02*nn.functional.l1_loss(gate_frequency(pred,mask),gate_frequency(yy,mask))
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(float(loss.detach().cpu()))
        pred,esr,mae=evaluate();
        if mae<best:
            best=mae;torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},epoch=epoch+1,val_esr=esr,val_relative_spectral_mae_db=mae),out/'best.pt')
        sched.step(); row=dict(epoch=epoch+1,loss=float(np.mean(losses)),val_esr=esr,val_relative_spectral_mae_db=mae,best_selection=best);history.append(row);print(json.dumps(row),flush=True)
    (out/'history.json').write_text(json.dumps(history,indent=2))
if __name__=='__main__':main()
