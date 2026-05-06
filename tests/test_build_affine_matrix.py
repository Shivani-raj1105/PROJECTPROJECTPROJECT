import numpy as np
import pytest

from cad_image_alignment.alignment import _build_affine_matrix


def test_build_affine_matrix_identity():
    scale = 1.0
    angle_deg = 0.0
    real_centroid = (100.0, 100.0)
    cad_centroid = (100.0, 100.0)

    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)

    assert M.shape == (3, 3)
    assert M.dtype == np.float64

    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])

    np.testing.assert_array_almost_equal(M, np.eye(3))


def test_build_affine_matrix_scale_only():
    scale = 2.0
    angle_deg = 0.0
    real_centroid = (50.0, 50.0)
    cad_centroid = (50.0, 50.0)

    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)

    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])

    point = np.array([60.0, 60.0, 1.0])
    transformed = M @ point

    expected = np.array([70.0, 70.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected)


def test_build_affine_matrix_rotation_only():
    scale = 1.0
    angle_deg = 90.0
    real_centroid = (100.0, 100.0)
    cad_centroid = (100.0, 100.0)

    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)

    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])

    point = np.array([110.0, 100.0, 1.0])
    transformed = M @ point

    expected = np.array([100.0, 110.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected, decimal=5)


def test_build_affine_matrix_translation_only():
    scale = 1.0
    angle_deg = 0.0
    real_centroid = (50.0, 50.0)
    cad_centroid = (150.0, 200.0)

    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)

    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])

    point = np.array([50.0, 50.0, 1.0])
    transformed = M @ point

    expected = np.array([150.0, 200.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected)


def test_build_affine_matrix_combined():
    scale = 2.0
    angle_deg = 45.0
    real_centroid = (100.0, 100.0)
    cad_centroid = (200.0, 200.0)

    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)

    assert M.shape == (3, 3)
    assert M.dtype == np.float64
    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])

    point = np.array([100.0, 100.0, 1.0])
    transformed = M @ point
    expected = np.array([200.0, 200.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected, decimal=5)


def test_build_affine_matrix_180_degree_rotation():
    scale = 1.0
    angle_deg = 180.0
    real_centroid = (100.0, 100.0)
    cad_centroid = (100.0, 100.0)

    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)

    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])

    point = np.array([110.0, 110.0, 1.0])
    transformed = M @ point

    expected = np.array([90.0, 90.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected, decimal=5)


def test_build_affine_matrix_arbitrary_angle():
    scale = 1.5
    angle_deg = 237.5
    real_centroid = (75.0, 125.0)
    cad_centroid = (300.0, 400.0)

    M = _build_affine_matrix(scale, angle_deg, real_centroid, cad_centroid)

    assert M.shape == (3, 3)
    assert M.dtype == np.float64
    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])

    point = np.array([75.0, 125.0, 1.0])
    transformed = M @ point
    expected = np.array([300.0, 400.0, 1.0])
    np.testing.assert_array_almost_equal(transformed, expected, decimal=5)
