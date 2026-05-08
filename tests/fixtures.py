import numpy as np
import cv2
from typing import Tuple


def generate_bracket_shape(
    image_size: Tuple[int, int] = (512, 512),
    outer_rect_size: Tuple[int, int] = (300, 200),
    cutout_size: int = 60,
    circle_radius: int = 15,
) -> np.ndarray:
    height, width = image_size
    canvas = np.zeros((height, width), dtype=np.uint8)

    center_x = width // 2
    center_y = height // 2

    rect_width, rect_height = outer_rect_size
    top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
    bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
    corner_radius = 20

    temp_canvas = canvas.copy()
    cv2.rectangle(
        temp_canvas,
        top_left,
        bottom_right,
        255,
        thickness=-1
    )

    cv2.circle(
        temp_canvas,
        (top_left[0] + corner_radius, top_left[1] + corner_radius),
        corner_radius,
        255,
        thickness=-1
    )
    cv2.circle(
        temp_canvas,
        (bottom_right[0] - corner_radius, top_left[1] + corner_radius),
        corner_radius,
        255,
        thickness=-1
    )
    cv2.circle(
        temp_canvas,
        (top_left[0] + corner_radius, bottom_right[1] - corner_radius),
        corner_radius,
        255,
        thickness=-1
    )
    cv2.circle(
        temp_canvas,
        (bottom_right[0] - corner_radius, bottom_right[1] - corner_radius),
        corner_radius,
        255,
        thickness=-1
    )

    cutout_x = center_x + rect_width // 2 - cutout_size // 2
    cutout_y = center_y - cutout_size // 2
    cv2.rectangle(
        temp_canvas,
        (cutout_x, cutout_y),
        (cutout_x + cutout_size, cutout_y + cutout_size),
        0,
        thickness=-1
    )

    circle1_pos = (center_x - rect_width // 4, center_y - rect_height // 4)
    cv2.circle(temp_canvas, circle1_pos, circle_radius, 0, thickness=-1)

    circle2_pos = (center_x - rect_width // 4, center_y + rect_height // 4)
    cv2.circle(temp_canvas, circle2_pos, circle_radius, 0, thickness=-1)

    circle3_pos = (center_x, center_y)
    cv2.circle(temp_canvas, circle3_pos, circle_radius, 0, thickness=-1)

    blurred = cv2.GaussianBlur(temp_canvas, (5, 5), 0)

    edge_map = cv2.Canny(blurred, threshold1=50, threshold2=150)

    return edge_map


def generate_rotated_variant(
    edge_map: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = edge_map.shape
    center = (width / 2.0, height / 2.0)

    rotation_matrix_2x3 = cv2.getRotationMatrix2D(
        center=center,
        angle=angle_deg,
        scale=1.0
    )

    transform_matrix = np.eye(3, dtype=np.float64)
    transform_matrix[:2, :] = rotation_matrix_2x3

    rotated_edge_map = cv2.warpAffine(
        edge_map,
        rotation_matrix_2x3,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated_edge_map, transform_matrix


def generate_noisy_variant(
    edge_map: np.ndarray,
    noise_level: float = 0.02,
) -> np.ndarray:
    noisy_edge_map = edge_map.copy()

    total_pixels = edge_map.shape[0] * edge_map.shape[1]

    num_noise_pixels = int(total_pixels * noise_level)

    salt_coords_y = np.random.randint(0, edge_map.shape[0], num_noise_pixels // 2)
    salt_coords_x = np.random.randint(0, edge_map.shape[1], num_noise_pixels // 2)
    noisy_edge_map[salt_coords_y, salt_coords_x] = 255

    pepper_coords_y = np.random.randint(0, edge_map.shape[0], num_noise_pixels // 2)
    pepper_coords_x = np.random.randint(0, edge_map.shape[1], num_noise_pixels // 2)
    noisy_edge_map[pepper_coords_y, pepper_coords_x] = 0

    return noisy_edge_map
#sakshi is amazing