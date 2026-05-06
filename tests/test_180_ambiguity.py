import numpy as np
import cv2
import pytest

from cad_image_alignment.alignment import (
    _compute_coarse_transform,
    apply_transform,
    _compute_alignment_score,
)


def create_simple_shape(size=200):
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(img, (50, 50),  (80, 150),  255, -1)
    cv2.rectangle(img, (50, 120), (150, 150), 255, -1)
    return cv2.Canny(img, 50, 150)


def test_coarse_transform_identical_shapes():
    cad  = create_simple_shape(size=200)
    real = cad.copy()

    M = _compute_coarse_transform(cad, real)
    assert M is not None
    assert M.shape == (3, 3)

    aligned = apply_transform(cad, M, output_shape=real.shape)
    score   = _compute_alignment_score(aligned, real)
    assert score > 0.7, f"Expected high score for identical shapes, got {score}"


def test_coarse_transform_180_rotation():
    cad = create_simple_shape(size=200)
    rot = cv2.getRotationMatrix2D((100, 100), 180, 1.0)
    real = cv2.warpAffine(cad, rot, (200, 200), flags=cv2.INTER_NEAREST)

    M = _compute_coarse_transform(cad, real)
    assert M is not None

    aligned = apply_transform(cad, M, output_shape=real.shape)
    score   = _compute_alignment_score(aligned, real)
    assert score > 0.5, f"Expected reasonable score after 180° recovery, got {score}"


def test_coarse_transform_no_contour_returns_none():
    cad  = create_simple_shape(size=200)
    real = np.zeros((200, 200), dtype=np.uint8)
    assert _compute_coarse_transform(cad, real) is None


def test_apply_transform_preserves_shape_and_dtype():
    edge_map = np.zeros((100, 100), dtype=np.uint8)
    edge_map[40:60, 40:60] = 255
    result = apply_transform(edge_map, np.eye(3, dtype=np.float64))
    assert result.shape == edge_map.shape
    assert result.dtype == np.uint8


def test_apply_transform_custom_output_shape():
    edge_map = np.zeros((100, 100), dtype=np.uint8)
    edge_map[40:60, 40:60] = 255
    result = apply_transform(edge_map, np.eye(3, dtype=np.float64), output_shape=(150, 150))
    assert result.shape == (150, 150)


def test_compute_alignment_score_identical():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), 255, 2)
    score = _compute_alignment_score(img, img.copy())
    assert score > 0.9


def test_compute_alignment_score_no_overlap():
    a = np.zeros((100, 100), dtype=np.uint8); a[10:30, 10:30] = 255
    b = np.zeros((100, 100), dtype=np.uint8); b[70:90, 70:90] = 255
    score = _compute_alignment_score(a, b)
    assert score == 0.0


def test_compute_alignment_score_empty():
    a = np.zeros((100, 100), dtype=np.uint8)
    b = np.zeros((100, 100), dtype=np.uint8)
    assert _compute_alignment_score(a, b) == 0.0
