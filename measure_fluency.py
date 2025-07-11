#!/usr/bin/env python
"""
Single-speaker fluency metrics (≥250 ms pause threshold).
Outputs:
  • transcript .txt (one line per segment)
  • output.csv with speech_rate, articulation_rate,
    mean_len_run, #pauses, mean_pause_ms
Usage:
  python measure_fluency.py input/ output.csv
"""

import argparse, re, sys
from pathlib import Path
import whisperx, parselmouth, pandas as pd

PAUSE = 0.25  # seconds

def syllables(path):
    snd = parselmouth.Sound(str(path))
    nuclei = snd.to_syllable_nuclei()
    return len(nuclei), snd.get_total_duration()           # (sylls, secs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir"), ap.add_argument("outfile")
    args = ap.parse_args()

    wavs = list(Path(args.indir).glob("*.wav")) + list(Path(args.indir).glob("*.mp3"))
    if not wavs:
        sys.exit("No audio files in " + args.indir)

    # small-int8 model → runs on CPU, no HF token, no diarisation
    model = whisperx.load_model("small", device="cpu", compute_type="int8")

    rows = []
    for wav in wavs:
        segs = model.transcribe(str(wav))["segments"]

        # save plain transcript
        with open(wav.with_suffix(".txt"), "w", encoding="utf-8") as t:
            for s in segs:
                t.write(s["text"].strip() + "\n")

        words = [w for s in segs for w in s["words"]]
        # pause list (between words)
        pauses = [b["start"] - a["end"] for a, b in zip(words, words[1:]) if b["start"] - a["end"] >= PAUSE]
        n_p   = len(pauses)
        mlr   = len(words) / n_p if n_p else len(words)

        sylls, tot_secs = syllables(wav)
        speak_secs = tot_secs - sum(pauses)
        speech_rt  = sylls / (tot_secs / 60)
        artic_rt   = sylls / (speak_secs / 60) if speak_secs else 0

        rows.append({
            "file": wav.name,
            "speech_rate_syll/min": round(speech_rt, 2),
            "artic_rate_syll/min": round(artic_rt, 2),
            "mean_len_run": round(mlr, 2),
            "pauses": n_p,
            "mean_pause_ms": round(sum(pauses)/n_p*1000, 1) if n_p else 0
        })

    pd.DataFrame(rows).to_csv(args.outfile, index=False)
    print("✅  Saved →", args.outfile)

if __name__ == "__main__":
    main()
