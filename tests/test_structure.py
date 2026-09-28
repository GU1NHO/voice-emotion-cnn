from pathlib import Path

import torch

from app.audio import to_mel
from app.main import app
from app.model import EmotionCNN, TransferEmotionCNN


def test_scratch_cnn_matches_assignment_shape_and_size() -> None:
    model = EmotionCNN(num_classes=7)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 1_100_000 < parameters < 1_250_000
    assert model(torch.zeros(2, 1, 64, 251)).shape == (2, 7)


def test_transfer_resnet_is_frozen_except_final_head() -> None:
    model = TransferEmotionCNN(num_classes=7, pretrained=False)
    assert all(not parameter.requires_grad for name, parameter in model.backbone.named_parameters() if not name.startswith("fc."))
    assert all(parameter.requires_grad for parameter in model.backbone.fc.parameters())
    assert model(torch.zeros(2, 1, 64, 251)).shape == (2, 7)


def test_emodb_audio_becomes_expected_log_mel_shape() -> None:
    sample = next(Path("data/emodb/wav").glob("*.wav"))
    assert to_mel(sample).shape == (1, 64, 251)


def test_health_route_reports_untrained_model() -> None:
    from fastapi.testclient import TestClient

    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"trained_model_available": False}
