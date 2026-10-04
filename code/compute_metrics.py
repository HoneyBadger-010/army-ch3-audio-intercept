"""Before/after metrics for the denoise: tone suppression (dB) and pulse-beat autocorrelation."""
import numpy as np, soundfile as sf, sys
from scipy.signal import correlate
sys.stdout.reconfigure(encoding="utf-8")

RAW = r"C:/Users/Admin/Desktop/army internship/war-intercept.wav"
CLEAN = r"C:/Users/Admin/Desktop/army internship/challenge3_output/war-intercept_cleaned.wav"
TONES = [60, 700, 1400, 2600, 2800, 4200]

def avg_spectrum(x, fs, N=16384):
    win = np.hanning(N); acc = np.zeros(N//2+1); n=0
    for i in range(0, len(x)-N, N//2):
        acc += np.abs(np.fft.rfft(x[i:i+N]*win))**2; n+=1
    return acc/n, np.fft.rfftfreq(N, 1/fs)

def load(p):
    x, fs = sf.read(p)
    if x.ndim>1: x=x.mean(1)
    return x.astype(float), fs

def tone_db(P, f, fs, bw=12):
    k = (fs/ (2*(len(P)-1)))  # Hz per bin
    band = [i for i in range(len(P)) if abs(i*k - f) <= bw]
    tot = P.sum()
    return 10*np.log10(sum(P[i] for i in band)/tot + 1e-12)

def pulse_ac(x, fs):
    b = x - x.mean(); X = np.fft.rfft(b); fr = np.fft.rfftfreq(len(b),1/fs)
    X[fr<5000]=0; hi=np.fft.irfft(X,n=len(b)); env=np.abs(hi)
    env=np.convolve(env,np.ones(64)/64,mode="same"); env=env[::fs//2000]; env-=env.mean()
    ac=correlate(env,env,mode="full",method="fft")[len(env)-1:]; ac/=ac[0]
    lo,hi2=int(0.11*2000),int(0.14*2000)
    lag=lo+int(np.argmax(ac[lo:hi2])); return ac[lag], 1000*lag/2000

xr, fs = load(RAW); xc, _ = load(CLEAN)
Pr,_ = avg_spectrum(xr,fs); Pc,_ = avg_spectrum(xc,fs)
print("TONE SUPPRESSION (energy at each jammer frequency, dB below total)")
print("  freq      raw dB     cleaned dB     reduction")
for f in TONES:
    r=tone_db(Pr,f,fs); c=tone_db(Pc,f,fs)
    print(f"  {f:5d} Hz   {r:7.1f}    {c:8.1f}      {r-c:6.1f} dB quieter")
acr,lagr = pulse_ac(xr,fs); acc,lagc = pulse_ac(xc,fs)
print(f"\nPULSE BEAT (envelope autocorrelation at ~125 ms, 1.0 = strongest)")
print(f"  raw     : {acr:0.3f}")
print(f"  cleaned : {acc:0.3f}   ({100*(1-acc/acr):0.0f}% weaker)")
print(f"\nRMS level  raw {20*np.log10(np.sqrt((xr**2).mean())+1e-12):0.1f} dBFS -> cleaned {20*np.log10(np.sqrt((xc**2).mean())+1e-12):0.1f} dBFS")
