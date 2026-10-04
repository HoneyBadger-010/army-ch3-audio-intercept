# Challenge 3 — Audio Intercept: denoising & transcription

Pre-interview challenge for the Indian Army (Eastern Command) internship selection.
**Analysis & report: Omkar Prabhakar Gadekar.**

## 🔊 Hear the result
- **Cleaned audio (final):** [`audio/war-intercept_cleaned.wav`](audio/war-intercept_cleaned.wav)
- **Original (jammed) audio:** [`audio/war-intercept_original.wav`](audio/war-intercept_original.wav)

(Open the file, then click **Download** / the raw link to play it in your browser.)

## What this is
A "war intercept" buried under **seven layers of deliberate jamming** — five steady tones (700/1400/2600/2800/4200 Hz), 60 Hz mains hum, a periodic 1.5 s frequency sweep, an 8 Hz click train, and broadband hiss, all over speech below ~1 kHz. Each layer was removed with the technique matched to its structure (notch for tones, a phase-fold median tracker for the moving sweep, a voice-aware gate for the clicks, broadband reduction for hiss). **Whisper large-v3** then produced a full **English** transcript. The content is **not tactical traffic** — it is a **1942 radio dramatisation of a soldier's Christmas letter home** (a decoy). No hidden data was present in the spectrogram.

## Contents
- `Challenge3_Report.pdf` / `.html` — full report: jammer analysis, the mathematics, before/after measurements, the full transcript, the 10-method residual-beat tournament
- `audio/` — cleaned (final) and original WAV
- `code/` — denoise scripts (FFT notch, phase-fold sweep tracker, VAD pulse gate) + transcription + measurement
- `transcript_narration.txt` — the full transcript
- `figures/` — before/after spectrograms

*Practice / assessment exercise.*
