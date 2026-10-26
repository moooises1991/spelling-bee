# Transcribe cada audio con Whisper para encontrar palabras o letras mal pronunciadas
import json, glob, os, time, subprocess
import numpy as np
from faster_whisper import WhisperModel
m = WhisperModel("small.en", device="cpu", compute_type="int8")
exp = json.load(open("expected.json"))
only = os.environ.get("ONLY", "")
out, t0 = {}, time.time()
for i, k in enumerate(sorted(exp)):
    if only and not any(k.startswith(p) for p in only.split(",")): continue
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f"clips/{k}.mp3", "-f", "s16le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    segs, _ = m.transcribe(audio, language="en", beam_size=5, without_timestamps=True, vad_filter=False, condition_on_previous_text=False)
    out[k] = " ".join(s.text.strip() for s in segs)
    print("RES\t" + k + "\t" + out[k], flush=True)
json.dump(out, open("results.json", "w"), indent=0)
print("listo", len(out))
