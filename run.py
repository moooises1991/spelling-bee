# Transcribe cada audio con Whisper para encontrar palabras o letras mal pronunciadas
import json, glob, os, time
from faster_whisper import WhisperModel
m = WhisperModel("small.en", device="cpu", compute_type="int8")
exp = json.load(open("expected.json"))
only = os.environ.get("ONLY", "")
out, t0 = {}, time.time()
for i, k in enumerate(sorted(exp)):
    if only and not any(k.startswith(p) for p in only.split(",")): continue
    segs, _ = m.transcribe(f"clips/{k}.mp3", language="en", beam_size=5, without_timestamps=True, vad_filter=False, condition_on_previous_text=False)
    out[k] = " ".join(s.text.strip() for s in segs)
    if i % 200 == 0: print(i, k, out[k], round(time.time() - t0), flush=True)
json.dump(out, open("results.json", "w"), indent=0)
print("listo", len(out))
