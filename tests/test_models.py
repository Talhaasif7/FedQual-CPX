"""
Unit tests for neural network architectures in FedQual-CPX.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.cnn import SmallCNN, FEMNISTCNN, create_model
from src.models.lstm import ShakespeareLSTM


def test_small_cnn_shape():
    model = SmallCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"
    print("[PASS] SmallCNN forward shape verified: (4, 10)")


def test_femnist_cnn_shape():
    model = FEMNISTCNN(num_classes=62)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 62), f"Expected (4, 62), got {out.shape}"
    print("[PASS] FEMNISTCNN forward shape verified: (4, 62)")


def test_create_model_factory():
    m1 = create_model("SmallCNN", 10)
    assert isinstance(m1, SmallCNN)
    m2 = create_model("FEMNISTCNN", 62)
    assert isinstance(m2, FEMNISTCNN)

    caught = False
    try:
        create_model("NonExistentModel")
    except ValueError:
        caught = True
    assert caught, "create_model failed to raise ValueError on unknown model"
    print("[PASS] Model factory verified for SmallCNN and FEMNISTCNN")


def test_shakespeare_lstm_shape():
    model = ShakespeareLSTM(vocab_size=90, embedding_dim=8, hidden_dim=64, num_layers=2)
    x = torch.randint(0, 90, (4, 50))  # Batch=4, seq_len=50
    logits, (h, c) = model(x)
    assert logits.shape == (4, 50, 90), f"Expected (4, 50, 90), got {logits.shape}"
    assert h.shape == (2, 4, 64), f"Expected (2, 4, 64), got {h.shape}"
    print("[PASS] ShakespeareLSTM forward shape verified: (4, 50, 90)")


if __name__ == "__main__":
    test_small_cnn_shape()
    test_femnist_cnn_shape()
    test_create_model_factory()
    test_shakespeare_lstm_shape()
    print("\nAll model architecture unit tests passed!")
