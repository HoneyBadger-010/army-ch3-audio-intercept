import sys, time, wave
import numpy as np
from faster_whisper import WhisperModel

model_size = sys.argv[1] if len(sys.argv) > 1 else "medium"
audio = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\Admin\Desktop\army internship\challenge3_output\war-intercept_cleaned.wav"
task = sys.argv[3] if len(sys.argv) > 3 else "transcribe"
language = sys.argv[4] if len(sys.argv) > 4 else None  # None = auto-detect

w = wave.open(audio, 'rb')
sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
if ch > 1:
    x = x.reshape(-1, ch).mean(axis=1)
assert sr == 16000, f"expected 16kHz, got {sr}"
print(f"decoded {len(x)/sr:.1f}s @ {sr}Hz ch={ch}; model={model_size} lang={language} task={task}", flush=True)

t0 = time.time()
model = WhisperModel(model_size, device="cpu", compute_type="int8")
segments, info = model.transcribe(
    x, task=task, language=language,
    vad_filter=False,                  # process the whole file; VAD was discarding quiet speech
    condition_on_previous_text=False,  # avoid runaway repetition on noisy audio
    beam_size=5,
    no_speech_threshold=0.6,
)
print(f"=== MODEL={model_size} lang={info.language} prob={info.language_probability:.2f} dur={info.duration:.1f}s ===", flush=True)
full = []
for s in segments:
    line = f"[{s.start:7.2f} -> {s.end:7.2f}] {s.text.strip()}"
    print(line, flush=True)
    full.append(s.text.strip())
print("\n--- JOINED ---\n" + " ".join(full), flush=True)
print("\nELAPSED", round(time.time() - t0, 1), "s", flush=True)
