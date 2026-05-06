import numpy as np
import pytest

from tests.fixtures import (
    generate_bracket_shape,
    generate_rotated_variant,
    generate_noisy_variant,
)


def test_generate_bracket_shape_basic():
    edge_map = generate_bracket_shape()

    assert isinstance(edge_map, np.ndarray)
    assert edge_map.dtype == np.uint8
    assert edge_map.ndim == 2
    assert edge_map.shape == (512, 512)

    assert np.any(edge_map > 0), "Edge map should contain edge pixels"

    unique_values = np.unique(edge_map)
    assert all(v in [0, 255] for v in unique_values), "Edge map should be binary"


def test_generate_bracket_shape_custom_size():
    edge_map = generate_bracket_shape(image_size=(256, 256))

    assert edge_map.shape == (256, 256)
    assert edge_map.dtype == np.uint8
    assert np.any(edge_map > 0)


def test_generate_rotated_variant_basic():
    edge_map = generate_bracket_shape(image_size=(200, 200))

    rotated, transform = generate_rotated_variant(edge_map, angle_deg=45.0)

    assert isinstance(rotated, np.ndarray)
    assert rotated.dtype == np.uint8
    assert rotated.ndim == 2
    assert rotated.shape == edge_map.shape
    assert np.any(rotated > 0), "Rotated edge map should contain edge pixels"

    assert isinstance(transform, np.ndarray)
    assert transform.dtype == np.float64
    assert transform.shape == (3, 3)

    assert np.allclose(transform[2, :], [0, 0, 1])


def test_generate_rotated_variant_zero_rotation():
    edge_map = generate_bracket_shape(image_size=(200, 200))

    rotated, transform = generate_rotated_variant(edge_map, angle_deg=0.0)

    assert rotated.shape == edge_map.shape

    expected = np.eye(3, dtype=np.float64)
    assert np.allclose(transform, expected, atol=1e-6)


def test_generate_rotated_variant_multiple_angles():
    edge_map = generate_bracket_shape(image_size=(200, 200))

    for angle in [0, 45, 90, 135, 180, 270]:
        rotated, transform = generate_rotated_variant(edge_map, angle_deg=float(angle))

        assert rotated.shape == edge_map.shape
        assert rotated.dtype == np.uint8
        assert transform.shape == (3, 3)
        assert transform.dtype == np.float64


def test_generate_noisy_variant_basic():
    edge_map = generate_bracket_shape(image_size=(200, 200))

    noisy = generate_noisy_variant(edge_map, noise_level=0.02)

    assert isinstance(noisy, np.ndarray)
    assert noisy.dtype == np.uint8
    assert noisy.ndim == 2
    assert noisy.shape == edge_map.shape

    assert not np.array_equal(noisy, edge_map), "Noisy image should differ from original"


def test_generate_noisy_variant_zero_noise():
    edge_map = generate_bracket_shape(image_size=(200, 200))

    noisy = generate_noisy_variant(edge_map, noise_level=0.0)

    assert np.array_equal(noisy, edge_map)


def test_generate_noisy_variant_high_noise():
    edge_map = generate_bracket_shape(image_size=(200, 200))

    noisy = generate_noisy_variant(edge_map, noise_level=0.10)

    diff_pixels = np.sum(noisy != edge_map)
    total_pixels = edge_map.shape[0] * edge_map.shape[1]

    assert diff_pixels > 0.02 * total_pixels, \
        f"Expected at least {0.02 * total_pixels:.0f} different pixels, got {diff_pixels}"

    assert diff_pixels < 0.15 * total_pixels, \
        f"Expected less than {0.15 * total_pixels:.0f} different pixels, got {diff_pixels}"


def test_generate_noisy_variant_preserves_binary():
    edge_map = generate_bracket_shape(image_size=(200, 200))

    noisy = generate_noisy_variant(edge_map, noise_level=0.05)

    unique_values = np.unique(noisy)
    assert all(v in [0, 255] for v in unique_values), "Noisy edge map should remain binary"


def test_fixtures_integration():
    bracket = generate_bracket_shape(image_size=(300, 300))
    assert np.any(bracket > 0)

    rotated, transform = generate_rotated_variant(bracket, angle_deg=90.0)
    assert rotated.shape == bracket.shape
    assert transform.shape == (3, 3)

    noisy = generate_noisy_variant(rotated, noise_level=0.03)
    assert noisy.shape == rotated.shape
    assert not np.array_equal(noisy, rotated)

    for img in [bracket, rotated, noisy]:
        assert img.dtype == np.uint8
        assert img.ndim == 2
        assert np.any(img > 0)
