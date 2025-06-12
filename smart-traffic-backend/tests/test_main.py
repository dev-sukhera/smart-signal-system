import importlib
import os

import numpy as np
from fastapi.testclient import TestClient


def create_client():
    # Use the lightweight dummy model during tests to avoid loading YOLO weights.
    os.environ["ITCS_USE_DUMMY_MODEL"] = "1"
    module = importlib.import_module("main")
    return TestClient(module.app), module


class DummyCapture:
    def __init__(self, frames):
        self._frames = frames
        self._index = 0

    def isOpened(self):
        return True

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self):
        pass


def _build_dummy_frames(count: int = 5):
    return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(count)]


def test_health_check():
    client, _ = create_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_ambulance_detected(monkeypatch, tmp_path):
    client, module = create_client()

    frames = _build_dummy_frames()

    def dummy_videocapture(path):
        return DummyCapture(frames)

    monkeypatch.setattr(module.cv2, "VideoCapture", dummy_videocapture)

    # Ensure the dummy model reports an ambulance.
    module.model.detect_ambulance = True

    video_file = tmp_path / "ambulance_video.mp4"
    video_file.write_bytes(b"dummy-video-content")

    with video_file.open("rb") as f:
        files = {"file": ("ambulance_video.mp4", f, "video/mp4")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["ambulance"] is True
    assert data["frame"] is not None
    assert data["confidence"] >= 0.75


def test_predict_no_ambulance_detected(monkeypatch, tmp_path):
    client, module = create_client()

    frames = _build_dummy_frames()

    def dummy_videocapture(path):
        return DummyCapture(frames)

    monkeypatch.setattr(module.cv2, "VideoCapture", dummy_videocapture)

    # Ensure the dummy model does not report an ambulance.
    module.model.detect_ambulance = False

    video_file = tmp_path / "traffic_video.mp4"
    video_file.write_bytes(b"dummy-video-content")

    with video_file.open("rb") as f:
        files = {"file": ("traffic_video.mp4", f, "video/mp4")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["ambulance"] is False
    assert data["frame"] is not None
    assert data["confidence"] == 0.0

