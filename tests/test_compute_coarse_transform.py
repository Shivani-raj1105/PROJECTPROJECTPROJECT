import numpy as np
import pytest
import cv2

from cad_image_alignment.alignment import _compute_coarse_transform


def create_simple_shape(size=200, offset=(0, 0)):
    img = np.zeros((size, size), dtype=np.uint8)
    x_off, y_off = offset
    cv2.rectangle(img, (50 + x_off, 50 + y_off), (150 + x_off, 150 + y_off), 255, 2)
    return img


def test_compute_coarse_transform_basic():
    cad_edge_map = create_simple_shape(size=200)

    real_edge_map = create_simple_shape(size=200)

    result = _compute_coarse_transform(cad_edge_map, real_edge_map)

    assert result is not None
    assert result.shape == (3, 3)
    assert result.dtype == np.float64

    np.testing.assert_array_almost_equal(result[2, :], [0.0, 0.0, 1.0])


def test_compute_coarse_transform_with_translation():
    cad_edge_map = create_simple_shape(size=200, offset=(0, 0))

    real_edge_map = create_simple_shape(size=200, offset=(20, 30))

    result = _compute_coarse_transform(cad_edge_map, real_edge_map)

    assert result is not None
    assert result.shape == (3, 3)
    assert result.dtype == np.float64


def test_compute_coarse_transform_no_contour_in_real():
    cad_edge_map = create_simple_shape(size=200)

    real_edge_map = np.zeros((200, 200), dtype=np.uint8)

    result = _compute_coarse_transform(cad_edge_map, real_edge_map)

    assert result is None


def test_compute_coarse_transform_no_contour_in_cad():
    cad_edge_map = np.zeros((200, 200), dtype=np.uint8)

    real_edge_map = create_simple_shape(size=200)

    result = _compute_coarse_transform(cad_edge_map, real_edge_map)

    assert result is None


def test_compute_coarse_transform_with_scale():
    cad_edge_map = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(cad_edge_map, (50, 50), (250, 250), 255, 2)

    real_edge_map = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(real_edge_map, (100, 100), (200, 200), 255, 2)

    result = _compute_coarse_transform(cad_edge_map, real_edge_map)

    assert result is not None
    assert result.shape == (3, 3)
    assert result.dtype == np.float64

    np.testing.assert_array_almost_equal(result[2, :], [0.0, 0.0, 1.0])


def test_compute_coarse_transform_with_rotation():
    cad_edge_map = create_simple_shape(size=200)

    real_edge_map = np.zeros((200, 200), dtype=np.uint8)
    center = (100, 100)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    temp = create_simple_shape(size=200)
    real_edge_map = cv2.warpAffine(temp, M, (200, 200))

    result = _compute_coarse_transform(cad_edge_map, real_edge_map)

    assert result is not None
    assert result.shape == (3, 3)
    assert result.dtype == np.float64

    np.testing.assert_array_almost_equal(result[2, :], [0.0, 0.0, 1.0])
