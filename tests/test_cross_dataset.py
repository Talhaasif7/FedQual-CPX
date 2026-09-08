import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.data.loaders import FEMNISTDataset, ShakespeareDataset
from src.models.cnn import create_model
from src.models.lstm import ShakespeareLSTM


def test_femnist_pipeline():
    ds = FEMNISTDataset()
    assert len(ds) > 0
    img, lbl = ds[0]
    assert img.shape == (1, 28, 28)
    assert 0 <= int(lbl) < 62

    model = create_model("FEMNISTCNN", num_classes=62)
    out = model(img.unsqueeze(0))
    assert out.shape == (1, 62)
    print("FEMNIST pipeline test passed!")


def test_shakespeare_pipeline():
    ds = ShakespeareDataset()
    assert len(ds) > 0
    seq, lbl = ds[0]
    assert seq.shape == (80,)
    assert 0 <= int(lbl) < 90

    model = create_model("ShakespeareLSTM", num_classes=90)
    out = model(seq.unsqueeze(0))
    logits = out[0] if isinstance(out, tuple) else out
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    assert logits.shape == (1, 90)
    print("Shakespeare pipeline test passed!")


if __name__ == "__main__":
    test_femnist_pipeline()
    test_shakespeare_pipeline()
    print("All cross-dataset pipeline tests passed!")
