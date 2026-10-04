"""Measure the jammer in war-intercept.wav straight from the samples:
narrowband tone peaks (averaged FFT) and the pulse-train period (envelope autocorrelation)."""
import numpy as np, soundfile as sf, sys
sys.stdout.reconfigure(encoding="utf-8")

x, fs = sf.read(r"C:/Users/Admin/Desktop/army internship/war-intercept.wav")
if x.ndim > 1: x = x.mean(1)
print(f"file        : war-intercept.wav")
print(f"sample rate : {fs} Hz   samples: {len(x):,}   duration: {len(x)/fs:0.2f} s\n")

# --- averaged power spectrum (Welch) -> steady tone peaks ---
N = 16384
win = np.hanning(N)
acc = np.zeros(N // 2 + 1)
nfr = 0
for i in range(0, len(x) - N, N // 2):
    acc += np.abs(np.fft.rfft(x[i:i+N] * win)) ** 2
    nfr += 1
P = acc / nfr
freqs = np.fft.rfftfreq(N, 1 / fs)
Pdb = 10 * np.log10(P / P.max() + 1e-12)

# pick narrowband peaks that stand well above their neighbourhood
print("STEADY TONES  (narrowband peaks in the averaged spectrum)")
print("  freq (Hz)   level (dB)   prominence")
cand = []
k = 10
for i in range(k, len(P) - k):
    local = np.concatenate([Pdb[i-k:i-2], Pdb[i+3:i+k]])
    if Pdb[i] == max(Pdb[i-2:i+3]) and Pdb[i] - np.median(local) > 12 and freqs[i] > 40:
        cand.append((freqs[i], Pdb[i], Pdb[i] - np.median(local)))
cand.sort(key=lambda c: -c[2])
seen = []
for f, d, pr in sorted(cand, key=lambda c: c[0]):
    if any(abs(f - s) < 40 for s in seen): continue
    seen.append(f)
    print(f"   {f:8.1f}   {d:8.1f}    +{pr:4.1f} dB")

# --- pulse train period via envelope autocorrelation on a speech-free band ---
from numpy.fft import rfft, irfft
from scipy.signal import correlate
b = x - np.mean(x)
X = rfft(b)
fr = np.fft.rfftfreq(len(b), 1/fs)
X[fr < 5000] = 0                          # keep 5 kHz+ (above the voice) to isolate the clicks
hi = irfft(X, n=len(b))
env = np.abs(hi)
env = np.convolve(env, np.ones(64)/64, mode="same")
dsr = 2000                                # decimate the envelope to 2 kHz so autocorr is fast
step = fs // dsr
env = env[::step]
env -= env.mean()
ac = correlate(env, env, mode="full", method="fft")[len(env)-1:]
ac /= ac[0]
lo, hiL = int(0.04*dsr), int(0.5*dsr)     # search 40–500 ms
lag = lo + int(np.argmax(ac[lo:hiL]))
fsE = dsr
print(f"\nPULSE / CLICK TRAIN  (envelope autocorrelation, 5 kHz+ band)")
print(f"  peak lag    : {1000*lag/fsE:0.1f} ms   ->   {fsE/lag:0.2f} Hz")
print(f"  correlation : {ac[lag]:0.3f}   (1.0 = perfectly periodic)")
print(f"  bursts/279s : ~{int(round(279*fsE/lag))}")
print("\nInterpretation: exact tones + an exact pulse period = engineered jamming, not natural noise.")
