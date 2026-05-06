import logging
import numpy as np
import pytest

from cad_image_alignment.alignment import _validate_inputs


def test_valid_inputs():
    cad = np.ones((100, 100), dtype=np.uint8) * 255
    real = np.ones((100, 100), dtype=np.uint8) * 255

    result = _validate_inputs(cad, real)
    assert result.shape == cad.shape
    assert result.dtype == np.uint8


def test_invalid_dtype_cad():
    cad = np.ones((100, 100), dtype=np.float32)
    real = np.ones((100, 100), dtype=np.uint8) * 255

    with pytest.raises(ValueError, match="cad_edge_map has invalid dtype"):
        _validate_inputs(cad, real)


def test_invalid_dtype_real():
    cad = np.ones((100, 100), dtype=np.uint8) * 255
    real = np.ones((100, 100), dtype=np.int32)

    with pytest.raises(ValueError, match="real_edge_map has invalid dtype"):
        _validate_inputs(cad, real)


def test_invalid_ndim_cad():
    cad = np.ones((100, 100, 3), dtype=np.uint8)
    real = np.ones((100, 100), dtype=np.uint8) * 255

    with pytest.raises(ValueError, match="cad_edge_map has invalid ndim"):
        _validate_inputs(cad, real)


def test_invalid_ndim_real():
    cad = np.ones((100, 100), dtype=np.uint8) * 255
    real = np.ones((100,), dtype=np.uint8)

    with pytest.raises(ValueError, match="real_edge_map has invalid ndim"):
        _validate_inputs(cad, real)


def test_empty_cad():
    cad = np.zeros((100, 100), dtype=np.uint8)
    real = np.ones((100, 100), dtype=np.uint8) * 255

    with pytest.raises(ValueError, match="cad_edge_map is empty"):
        _validate_inputs(cad, real)


def test_empty_real():
    cad = np.ones((100, 100), dtype=np.uint8) * 255
    real = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError, match="real_edge_map is empty"):
        _validate_inputs(cad, real)


def test_both_invalid():
    cad = np.zeros((100, 100), dtype=np.float32)
    real = np.zeros((100, 100), dtype=np.int32)

    with pytest.raises(ValueError, match="cad_edge_map has invalid dtype"):
        _validate_inputs(cad, real)


def test_resolution_mismatch_resize():
    real = np.ones((100, 100), dtype=np.uint8) * 255

    cad = np.ones((50, 50), dtype=np.uint8) * 255

    result = _validate_inputs(cad, real)

    assert result.shape == real.shape
    assert result.dtype == np.uint8
    assert result.shape == (100, 100)


def test_resolution_mismatch_logs_debug(caplog):
    real = np.ones((100, 100), dtype=np.uint8) * 255

    cad = np.ones((50, 50), dtype=np.uint8) * 255

    with caplog.at_level(logging.DEBUG, logger='cad_image_alignment.alignment'):
        result = _validate_inputs(cad, real)

    assert len(caplog.records) >= 1
    assert caplog.records[0].levelname == 'DEBUG'
    assert 'Resolution mismatch' in caplog.records[0].message
    assert '(50, 50)' in caplog.records[0].message
    assert '(100, 100)' in caplog.records[0].message
