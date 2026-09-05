# coding: utf-8
"""Degenerate OCR token boxes must not abort table structure recognition."""

import random

from pix2text.table_postprocess import slot_into_containers

CONTAINERS = [
    {'bbox': [0, 0, 200, 500], 'score': 1.0, 'label': 'table row'},
    {'bbox': [0, 500, 200, 900], 'score': 0.9, 'label': 'table row'},
]


def _token(bbox):
    return [{'bbox': bbox, 'text': 'x', 'span_num': 0, 'line_num': 0, 'block_num': 0}]


def test_inverted_box_does_not_raise():
    """A box written with y0 > y1 has zero Rect area; it must be normalised.

    Reported upstream as microsoft/table-transformer#135, where the offending
    token was Rect(95.99, 383.89, 98.00, 374.89) -- note y0 > y1.
    """
    bbox = [95.99999451586913, 383.89018184350584, 98.0000028239502, 374.8903775869141]
    _containers, packages, scores = slot_into_containers(CONTAINERS, _token(bbox))
    assert packages == [[0]], 'normalised box should slot into the first container'
    assert scores[0] > 0.999, 'the recovered box lies wholly inside that container'


def test_zero_height_box_is_left_unassigned():
    """A box with zero height covers no area, so it belongs to no container."""
    _containers, packages, scores = slot_into_containers(
        CONTAINERS, _token([10.0, 20.0, 50.0, 20.0])
    )
    assert packages == [[]]
    assert scores == [0]


def test_zero_width_box_is_left_unassigned():
    _containers, packages, scores = slot_into_containers(
        CONTAINERS, _token([10.0, 20.0, 10.0, 60.0])
    )
    assert packages == [[]]
    assert scores == [0]


def test_valid_boxes_are_unaffected():
    """The guard must not change behaviour for ordinary boxes."""
    rng = random.Random(0)
    for _ in range(200):
        x0 = rng.uniform(0, 150)
        y0 = rng.uniform(0, 800)
        bbox = [x0, y0, x0 + rng.uniform(1, 40), y0 + rng.uniform(1, 40)]
        _c, packages, scores = slot_into_containers(CONTAINERS, _token(bbox))
        assert len(scores) == 1
        # Rect intersection can exceed the package area by a float epsilon, which
        # is pre-existing behaviour for valid boxes and not what this guard changes.
        assert 0.0 <= scores[0] <= 1.0 + 1e-5
        assert packages[0] in ([0], [1])


def test_mixed_batch_keeps_scores_aligned_with_packages():
    """One bad token must not shift the score list relative to the inputs."""
    tokens = [
        {'bbox': [10, 10, 50, 40], 'text': 'a', 'span_num': 0, 'line_num': 0, 'block_num': 0},
        {'bbox': [10, 30, 50, 30], 'text': 'b', 'span_num': 1, 'line_num': 0, 'block_num': 0},
        {'bbox': [10, 600, 50, 640], 'text': 'c', 'span_num': 2, 'line_num': 0, 'block_num': 0},
    ]
    _c, packages, scores = slot_into_containers(CONTAINERS, tokens)
    assert len(packages) == len(tokens)
    assert len(scores) == len(tokens)
    assert packages[0] == [0]
    assert packages[1] == []
    assert packages[2] == [1]
