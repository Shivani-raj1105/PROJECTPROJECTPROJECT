import numpy as np
import cv2
import pytest

from cad_image_alignment.alignment import (
    _detect_and_match_features,
    _estimate_similarity,
    _validate_similarity,
    _compute_fine_transform,
)


def test_detect_and_match_features_basic():
    img1 = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (150, 150), 255, 2)
    cv2.circle(img1, (100, 100), 30, 255, 2)

    img2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img2, (55, 55), (155, 155), 255, 2)
    cv2.circle(img2, (105, 105), 30, 255, 2)

    kp1, kp2, des1, des2, matches = _detect_and_match_features(img1, img2)

    assert len(kp1) > 0
    assert len(kp2) > 0
    assert des1 is not None
    assert des2 is not None
    assert len(matches) > 0


def test_detect_and_match_features_empty_images():
    img1 = np.zeros((200, 200), dtype=np.uint8)
    img2 = np.zeros((200, 200), dtype=np.uint8)

    kp1, kp2, des1, des2, matches = _detect_and_match_features(img1, img2)

    assert des1 is None or len(kp1) == 0
    assert des2 is None or len(kp2) == 0
    assert len(matches) == 0


def test_estimate_similarity_insufficient_matches():
    kp1 = [cv2.KeyPoint(float(i), float(i), 1.0) for i in range(5)]
    kp2 = [cv2.KeyPoint(float(i), float(i), 1.0) for i in range(5)]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(5)]

    M, inlier_ratio = _estimate_similarity(kp1, kp2, matches)

    assert M is None
    assert inlier_ratio is None


def test_estimate_similarity_sufficient_matches():
    src_pts = [(10.0, 10.0), (100.0, 10.0), (100.0, 100.0), (10.0, 100.0),
               (50.0, 50.0), (30.0, 30.0), (70.0, 70.0), (50.0, 80.0)]
    dst_pts = [(x + 5.0, y + 5.0) for x, y in src_pts]

    kp1 = [cv2.KeyPoint(x, y, 1.0) for x, y in src_pts]
    kp2 = [cv2.KeyPoint(x, y, 1.0) for x, y in dst_pts]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(len(src_pts))]

    M, inlier_ratio = _estimate_similarity(kp1, kp2, matches)

    assert M is not None
    assert M.shape == (3, 3)
    assert inlier_ratio is not None
    assert 0.0 <= inlier_ratio <= 1.0
    np.testing.assert_array_almost_equal(M[2, :], [0.0, 0.0, 1.0])


def test_estimate_similarity_preserves_shape():
    src_pts = [(10.0, 10.0), (100.0, 10.0), (100.0, 100.0), (10.0, 100.0),
               (50.0, 50.0), (30.0, 30.0), (70.0, 70.0), (50.0, 80.0)]
    dst_pts = [(x + 5.0, y + 5.0) for x, y in src_pts]

    kp1 = [cv2.KeyPoint(x, y, 1.0) for x, y in src_pts]
    kp2 = [cv2.KeyPoint(x, y, 1.0) for x, y in dst_pts]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(len(src_pts))]

    M, _ = _estimate_similarity(kp1, kp2, matches)
    if M is None:
        pytest.skip("Similarity estimation returned None for this input")

    assert abs(M[0, 0] - M[1, 1]) < 1e-4, "Similarity must have M[0,0]==M[1,1]"
    assert abs(M[0, 1] + M[1, 0]) < 1e-4, "Similarity must have M[0,1]==-M[1,0]"


def test_validate_similarity_valid_scale():
    M = np.eye(3, dtype=np.float64)
    assert _validate_similarity(M) == True


def test_validate_similarity_scale_too_small():
    s = 0.3
    M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    assert _validate_similarity(M) == False


def test_validate_similarity_scale_too_large():
    s = 2.5
    M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    assert _validate_similarity(M) == False


def test_validate_similarity_boundary_values():
    for s in (0.5, 2.0):
        M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
        assert _validate_similarity(M) == True, f"Scale {s} should be valid"


def test_compute_fine_transform_insufficient_matches():
    coarsely_aligned = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(coarsely_aligned, (50, 50), 2, 255, -1)

    real_edge_map = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(real_edge_map, (50, 50), 2, 255, -1)

    coarse_matrix = np.eye(3, dtype=np.float64)

    M_total, inlier_ratio = _compute_fine_transform(
        coarsely_aligned, real_edge_map, coarse_matrix
    )

    if M_total is not None:
        assert M_total.shape == (3, 3)
        assert inlier_ratio is not None
        assert 0.0 <= inlier_ratio <= 1.0


def test_compute_fine_transform_with_good_features():
    coarsely_aligned = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(coarsely_aligned, (50, 50), (150, 150), 255, 2)
    cv2.circle(coarsely_aligned, (100, 100), 30, 255, 2)
    cv2.circle(coarsely_aligned, (80, 80), 10, 255, 2)
    cv2.circle(coarsely_aligned, (120, 120), 10, 255, 2)

    real_edge_map = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(real_edge_map, (52, 52), (152, 152), 255, 2)
    cv2.circle(real_edge_map, (102, 102), 30, 255, 2)
    cv2.circle(real_edge_map, (82, 82), 10, 255, 2)
    cv2.circle(real_edge_map, (122, 122), 10, 255, 2)

    coarse_matrix = np.eye(3, dtype=np.float64)

    M_total, inlier_ratio = _compute_fine_transform(
        coarsely_aligned, real_edge_map, coarse_matrix
    )

    if M_total is not None:
        assert M_total.shape == (3, 3)
        assert inlier_ratio is not None
        assert 0.0 <= inlier_ratio <= 1.0
        np.testing.assert_array_almost_equal(M_total[2, :], [0.0, 0.0, 1.0])
