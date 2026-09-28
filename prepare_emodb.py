"""Create the project manifest from the official Berlin EmoDB WAV filenames."""
from __future__ import annotations

import csv
from pathlib import Path

EMOTION_CODES = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}

root = Path("data/emodb/wav")
files = sorted(root.glob("*.wav"))
if not files:
    raise SystemExit("Nenhum WAV encontrado em data/emodb/wav.")

with Path("data/manifest.csv").open("w", newline="", encoding="utf-8") as target:
    writer = csv.DictWriter(target, fieldnames=("path", "label", "speaker"))
    writer.writeheader()
    for file in files:
        # Example: 03a01Fa.wav: actor=03 and emotion=F (joy/happy).
        code = file.stem[5]
        if code not in EMOTION_CODES:
            raise ValueError(f"Código de emoção desconhecido em {file.name}")
        writer.writerow({"path": file.as_posix(), "label": EMOTION_CODES[code], "speaker": file.stem[:2]})

print(f"Manifesto criado com {len(files)} arquivos e {len(EMOTION_CODES)} emoções.")
