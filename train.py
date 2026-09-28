"""Train either required CNN using a speaker-disjoint CSV manifest.

Manifest columns: path,label,speaker. Paths are relative to the repository root.
"""
from __future__ import annotations

import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from app.audio import to_mel
from app.model import EmotionCNN, TransferEmotionCNN


class SpecAugment:
    """Mask time/frequency stripes and randomly shift time, only during training."""

    def __init__(self, max_frequency_mask: int = 8, max_time_mask: int = 24, max_shift: int = 16) -> None:
        self.max_frequency_mask = max_frequency_mask
        self.max_time_mask = max_time_mask
        self.max_shift = max_shift

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        augmented = batch.clone()
        for sample in augmented:
            frequency_width = int(torch.randint(self.max_frequency_mask + 1, ()).item())
            frequency_start = int(torch.randint(sample.shape[-2] - frequency_width + 1, ()).item())
            sample[:, frequency_start:frequency_start + frequency_width, :] = 0
            time_width = int(torch.randint(self.max_time_mask + 1, ()).item())
            time_start = int(torch.randint(sample.shape[-1] - time_width + 1, ()).item())
            sample[:, :, time_start:time_start + time_width] = 0
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, ()).item())
            sample.copy_(torch.roll(sample, shifts=shift, dims=-1))
        return augmented


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/manifest.csv")
parser.add_argument("--architecture", choices=("scratch", "transfer"), required=True)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
with Path(args.manifest).open(encoding="utf-8", newline="") as source: rows = list(csv.DictReader(source))
if not rows or set(rows[0]) != {"path", "label", "speaker"}: raise SystemExit("O CSV deve conter exatamente path,label,speaker.")
labels = sorted({row["label"] for row in rows})
if len(labels) < 2 or len({row["speaker"] for row in rows}) < 2: raise SystemExit("São necessárias pelo menos duas emoções e dois locutores.")
features = np.stack([to_mel(Path(row["path"])) for row in rows]); targets = np.array([labels.index(row["label"]) for row in rows]); speakers = np.array([row["speaker"] for row in rows])
train_idx, test_idx = next(GroupShuffleSplit(test_size=.2, n_splits=1, random_state=args.seed).split(features, targets, speakers))
if set(speakers[train_idx]) & set(speakers[test_idx]): raise RuntimeError("Falha: há locutores em ambos os conjuntos.")
train = TensorDataset(torch.tensor(features[train_idx]), torch.tensor(targets[train_idx])); test = TensorDataset(torch.tensor(features[test_idx]), torch.tensor(targets[test_idx]))
model = TransferEmotionCNN(len(labels)) if args.architecture == "transfer" else EmotionCNN(len(labels))
optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
augment = SpecAugment()
for epoch in range(args.epochs):
    model.train()
    # Frozen ImageNet BatchNorm statistics must remain unchanged for transfer learning.
    if args.architecture == "transfer": model.backbone.eval()
    for xb, yb in DataLoader(train, batch_size=args.batch_size, shuffle=True):
        if args.architecture == "scratch": xb = augment(xb)
        optimizer.zero_grad(); loss_fn(model(xb), yb).backward(); optimizer.step()
    model.eval(); actual, predicted = [], []
    with torch.inference_mode():
        for xb, yb in DataLoader(test, batch_size=args.batch_size): actual.extend(yb.tolist()); predicted.extend(model(xb).argmax(1).tolist())
    print(f"epoch={epoch + 1} accuracy={accuracy_score(actual, predicted):.3f} macro_f1={f1_score(actual, predicted, average='macro'):.3f}")
Path("artifacts").mkdir(exist_ok=True); torch.save(model.state_dict(), f"artifacts/model-{args.architecture}.pt")
Path(f"artifacts/results-{args.architecture}.json").write_text(json.dumps({"architecture": args.architecture, "seed": args.seed, "speakers_train": sorted(set(speakers[train_idx])), "speakers_test": sorted(set(speakers[test_idx])), "accuracy": accuracy_score(actual, predicted), "macro_f1": f1_score(actual, predicted, average="macro")}, indent=2), encoding="utf-8")
Path("artifacts/labels.json").write_text(json.dumps(labels), encoding="utf-8"); Path("artifacts/metadata.json").write_text(json.dumps({"architecture": args.architecture}), encoding="utf-8")
print("Artefatos salvos. Para a demo, copie model-<arquitetura>.pt para artifacts/model.pt.")
