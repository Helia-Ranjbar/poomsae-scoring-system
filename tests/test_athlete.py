import numpy as np

from poomsae_scoring.preprocessing.athlete import (
    box_bottom_contact_points,
    box_iou,
    interpolate_box_gaps,
    isolate_person,
    select_front_candidate,
    select_tracking_candidate,
)


def test_select_front_candidate_prefers_lower_larger_center_person() -> None:
    boxes = np.array(
        [
            [80, 80, 160, 300],
            [260, 70, 390, 390],
            [500, 80, 580, 300],
        ],
        dtype=float,
    )

    selected = select_front_candidate(boxes, frame_width=640, frame_height=480)

    assert selected == 1


def test_tracking_candidate_prefers_previous_location() -> None:
    previous = np.array([250, 80, 390, 400], dtype=float)
    boxes = np.array(
        [
            [70, 70, 180, 360],
            [258, 85, 398, 405],
            [470, 75, 590, 365],
        ],
        dtype=float,
    )

    selected = select_tracking_candidate(boxes, previous, 640, 480)

    assert selected == 1


def test_isolate_person_blacks_out_background() -> None:
    frame = np.full((4, 4, 3), 255, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1

    isolated = isolate_person(frame, mask)

    assert np.all(isolated[1:3, 1:3] == 255)
    assert np.all(isolated[0] == 0)


def test_box_iou_for_identical_boxes_is_one() -> None:
    box = np.array([10, 20, 30, 50], dtype=float)

    assert box_iou(box, box) == 1.0


def test_box_bottom_contact_points_include_both_possible_stance_feet() -> None:
    points = box_bottom_contact_points(
        np.array([100, 50, 300, 400], dtype=float),
        inset_ratio=0.10,
    )

    np.testing.assert_allclose(
        points,
        np.array([[120, 400], [200, 400], [280, 400]], dtype=float),
    )


def test_interpolate_box_gaps_bridges_short_tracking_dropout() -> None:
    boxes = [
        np.array([0, 0, 10, 20], dtype=float),
        None,
        None,
        np.array([6, 3, 16, 23], dtype=float),
    ]

    interpolated = interpolate_box_gaps(boxes, max_gap=2)

    np.testing.assert_allclose(interpolated[1], [2, 1, 12, 21])
    np.testing.assert_allclose(interpolated[2], [4, 2, 14, 22])


def test_interpolate_box_gaps_leaves_long_dropout_missing() -> None:
    boxes = [
        np.array([0, 0, 10, 20], dtype=float),
        None,
        None,
        None,
        np.array([8, 4, 18, 24], dtype=float),
    ]

    interpolated = interpolate_box_gaps(boxes, max_gap=2)

    assert interpolated[1:4] == [None, None, None]
