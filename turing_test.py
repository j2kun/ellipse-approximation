import math
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
    ellipse = Ellipse(
        center_x=1, center_y=1, radius_x=3, radius_y=2,
        rotation=math.pi/4, r=0, g=0, b=0, alpha=0
    )

    assert_that(ellipse_contains_point(ellipse, 4, 1)).is_false()

    # (4,1) rotated by pi/4
    assert_that(ellipse_contains_point(ellipse, 2.1, 3.5)).is_true()
