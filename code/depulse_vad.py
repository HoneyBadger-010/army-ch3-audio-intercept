import sys, wave, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
SP = r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-army-internship\7ea60c92-ac86-48fc-b033-c9e1c4831b1e\scratchpad"

w = wave.open(SP + r"\sweep_removed.wav", 'rb'); sr = w.getframerate(); n = w.getnframes()
x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768.0

P = int(round(0.125 * sr)); phase0 = 56
e = np.convolve(x * x, np.ones(65) / 65, 'same')            # fast energy
se = np.convolve(x * x, np.ones(1600) / 1600, 'same')       # ~100 ms speech envelope
sp_med = np.median(se)

gain = np.ones(len(x)); c = phase0; deep = 0; soft = 0
while c < len(x):
    a0, b0 = max(0, c - 140), min(len(x), c + 140)
    cc = a0 + int(np.argmax(e[a0:b0])) if b0 > a0 else c
    base = np.median(np.r_[e[max(0, cc - 900):max(0, cc - 350)], e[min(len(x), cc + 350):min(len(x), cc + 900)]])
    speech = np.median(se[max(0, cc - 800):min(len(x), cc + 800)])
    if speech < 1.5 * sp_med:
        cap = 1.0 * base; deep += 1          # in a pause -> remove the naked tick fully
    else:
        cap = 2.2 * base; soft += 1          # under speech -> stay gentle
    a, b = max(0, cc - 120), min(len(x), cc + 120)
    seg = e[a:b]; g = np.sqrt(cap / np.maximum(seg, cap))
    gain[a:b] = np.minimum(gain[a:b], g)
    c += P
gain = np.convolve(gain, np.hanning(41) / np.hanning(41).sum(), 'same')
y = np.clip(x * gain, -1, 1)
print(f"VAD gate: deep(pause)={deep} soft(speech)={soft}, gain min {gain.min():.2f}")
out = (y * 32767).astype(np.int16)
ww = wave.open(SP + r"\pulse_removed.wav", 'wb'); ww.setnchannels(1); ww.setsampwidth(2); ww.setframerate(sr)
ww.writeframes(out.tobytes()); ww.close()
print("depulse_vad written")
