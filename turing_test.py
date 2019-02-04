import math
import numpy
from assertpy import assert_that
from turing import *


def test_unrotated_ellipse_contains_point():
    ellipse = Ellipse(
        center_x=1, center_y=1, radius_x=2, radius_y=3,
        rotation=0, r=0, g=0, b=0, alpha=0
    )
    assert_that(ellipse_contains_point(ellipse, 1, 1)).is_true()
    assert_that(ellipse_contains_point(ellipse, 1, 2)).is_true()
    assert_that(ellipse_contains_point(ellipse, 1, 3)).is_true()
    assert_that(ellipse_contains_point(ellipse, 1, 4)).is_true()
    assert_that(ellipse_contains_point(ellipse, 1, 5)).is_false()
    assert_that(ellipse_contains_point(ellipse, 3, 4)).is_false()

    assert_that(ellipse_contains_point(ellipse, 1, 1)).is_true()
    assert_that(ellipse_contains_point(ellipse, 2, 1)).is_true()
    assert_that(ellipse_contains_point(ellipse, 3, 1)).is_true()
    assert_that(ellipse_contains_point(ellipse, 4, 1)).is_false()
    assert_that(ellipse_contains_point(ellipse, 3, 3)).is_false()


def test_rotated_ellipse_contains_point():
    theta = math.pi / 8
    ellipse = Ellipse(
        center_x=1, center_y=1, radius_x=3, radius_y=2,
        rotation=theta, r=0, g=0, b=0, alpha=0
    )

    assert_that(ellipse_contains_point(ellipse, 4, 1)).is_false()

    # (4,1) rotated by theta around (1,1) == (3,0) rotated then shift
    rotation_matrix = [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ]
    rotated_pt = numpy.dot(rotation_matrix, [3, 0])
    x = ellipse.center_x + rotated_pt[0]
    y = ellipse.center_y + rotated_pt[1]
    assert_that(ellipse_contains_point(ellipse, x - 0.01, y - 0.01)).is_true()
    assert_that(ellipse_contains_point(ellipse, x + 0.01, y + 0.01)).is_false()

    assert_that(ellipse_contains_point(ellipse, 1, 3.01)).is_true()

    '''
    Bounding box:
    x_min, x_max (-1.7711244512803932, 3.771120782167797)
    y_min, y_max (-1.3066910373449676, 3.3066773423503184)
    '''
    assert_that(ellipse_contains_point(ellipse, 3.78, 3.31)).is_false()


def test_bounding_box():
    theta = math.pi / 8
    ellipse = Ellipse(
        center_x=1, center_y=1, radius_x=2, radius_y=3,
        rotation=theta, r=0, g=0, b=0, alpha=0
    )

    bounding_box = ellipse_bounding_box(ellipse)
    x_min, x_max = bounding_box[0]
    y_min, y_max = bounding_box[1]

    assert_that(x_min).is_close_to(-1.77, 0.01)
    assert_that(x_max).is_close_to(3.77, 0.01)
    assert_that(y_min).is_close_to(-1.30, 0.01)
    assert_that(y_max).is_close_to(3.30, 0.01)
