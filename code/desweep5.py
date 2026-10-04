import sys, wave, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
SP = r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-army-internship\7ea60c92-ac86-48fc-b033-c9e1c4831b1e\scratchpad"

w = wave.open(SP + r"\tone_removed.wav", 'rb'); sr = w.getframerate(); n = w.getnframes()
x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768.0

N = 512; hop = 128; win = np.hanning(N)
freqs = np.fft.rfftfreq(N, 1 / sr)
band = np.where((freqs >= 450) & (freqs <= 5100))[0]
nfr = 1 + (len(x) - N) // hop
t = (np.arange(nfr) * hop + N / 2) / sr

# per-frame peak + prominence
pf = np.empty(nfr); prom = np.empty(nfr)
MAG = np.empty((nfr, len(freqs)))
for i in range(nfr):
    m = np.abs(np.fft.rfft(x[i * hop:i * hop + N] * win)); MAG[i] = m
    pk = band[np.argmax(m[band])]; pf[i] = freqs[pk]; prom[i] = m[pk] / (np.median(m[band]) + 1e-12)
sel = prom > 3.0

def template(P, K):
    ph = (t[sel] % P) / P
    idx = np.clip((ph * K).astype(int), 0, K - 1)
    med = np.full(K, np.nan)
    for k in range(K):
        v = pf[sel][idx == k]
        if len(v): med[k] = np.median(v)
    return med

def score(P):
    K = 200; med = template(P, K)
    ph = (t[sel] % P) / P; idx = np.clip((ph * K).astype(int), 0, K - 1)
    r = np.abs(pf[sel] - med[idx]); r = r[~np.isnan(r)]
    return np.median(r)

# refine period (synthetic sweep -> constant period); coarse then fine
bestP = min(np.arange(1.470, 1.531, 0.001), key=score)
bestP = min(np.arange(bestP - 0.002, bestP + 0.002, 0.0002), key=score)
print(f"period={bestP:.4f}s  fold-residual={score(bestP):.1f}Hz")

# build smooth circular template at fine resolution
K = 360; med = template(bestP, K)
# fill gaps + circular smooth
good = ~np.isnan(med); xp = np.where(good)[0]
med = np.interp(np.arange(K), xp, med[good], period=K)
ks = 9; ker = np.ones(ks) / ks
med = np.convolve(np.r_[med[-ks:], med, med[:ks]], ker, 'same')[ks:-ks]
print(f"sweep range {med.min():.0f}-{med.max():.0f} Hz")

# notch along template phase
y = np.zeros(len(x)); wsum = np.zeros(len(x)) + 1e-9
for i in range(nfr):
    X = np.fft.rfft(x[i * hop:i * hop + N] * win)
    ph = (t[i] % bestP) / bestP
    fc = med[int(ph * K) % K]
    loc = band[np.abs(freqs[band] - fc) <= 90]
    if len(loc):
        c = loc[np.argmax(MAG[i][loc])]
        if MAG[i][c] > 2.5 * (np.median(MAG[i][band]) + 1e-12): fc = freqs[c]
    sigma = 55.0 if fc < 900 else 85.0
    X = X * (1 - 0.997 * np.exp(-((freqs - fc) / sigma) ** 2))
    rec = np.fft.irfft(X, n=N) * win
    y[i * hop:i * hop + N] += rec; wsum[i * hop:i * hop + N] += win * win
y /= wsum; y = np.clip(y, -1, 1)
out = (y * 32767).astype(np.int16)
ww = wave.open(SP + r"\sweep_removed.wav", 'wb'); ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(sr)
ww.writeframes(out.tobytes()); ww.close()
print("desweep5 (phase-fold median tracker) written")
