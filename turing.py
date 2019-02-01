''' Generate ellipse art

Outline:

The input is a single numpy array of dimensions
[n, n, 4], where the last dimension is an RGBA channel.

A hypothesis is a collection of ellipses of the form

(center_x, center_y, x_radius, y_radius, rotation)

Where center_x and center_y are from the top right of the image,
x_radius and y_radius are the two radii of the ellipse along the two axes, and
rotation is the rotation of the ellipse from the x axis. Rotation is in units
of 2pi / 100

Core operations:

    - Convert a collection of ellipses to an image of the right dimensions
    - Compare two such images (pixelwise sum of squared differences)
    - Generate a set of neighbor ellipses for any given ellipse

Then do randomized gradient descent from a large number of starting positions
to greedily find the next best ellipse to add.

'''

import numpy
import numba
from numba import jit, jitclass
from numba import int32, float32


'''
Data structures and generating neighbors
'''

spec = [
    ('center_x', int32),
    ('center_y', int32),
    ('radius_x', int32),
    ('radius_y', int32),
    ('rotation', int32),
    ('r', int32),
    ('g', int32),
    ('b', int32),
    ('alpha', int32),
]


@jitclass(spec)
class Ellipse:
    def __init__(self, center_x, center_y, radius_x, radius_y, rotation, r, g, b, alpha):
        self.center_x = center_x
        self.center_y = center_y
        self.radius_x = radius_x
        self.radius_x = radius_x
        self.rotation = rotation
        self.r = r
        self.g = g
        self.b = b
        self.alpha = alpha


# neighbors are in +/- window
center_x_window = 1
center_y_window = 1
radius_x_window = 1
radius_y_window = 1
rotation_window = 1
r_window = 1
g_window = 1
b_window = 1
alpha_window = 1


@jit(nopython=True)
def neighbors(ellipse):
    nbrs = []

    for center_x in range(ellipse.center_x - center_x_window, ellipse.center_x + center_x_window + 1):
        for center_y in range(ellipse.center_y - center_y_window, ellipse.center_y + center_y_window + 1):
            for radius_x in range(ellipse.radius_x - radius_x_window, ellipse.radius_x + radius_x_window + 1):
                for radius_y in range(ellipse.radius_y - radius_y_window, ellipse.radius_y + radius_y_window + 1):
                    for rotation in range(ellipse.rotation - rotation_window, ellipse.rotation + rotation_window + 1):
                        for r in range(ellipse.r - r_window, ellipse.r + r_window + 1):
                            for g in range(ellipse.g - g_window, ellipse.g + g_window + 1):
                                for b in range(ellipse.b - b_window, ellipse.b + b_window + 1):
                                    for alpha in range(ellipse.alpha - alpha_window, ellipse.alpha + alpha_window + 1):
                                        nbrs.append(Ellipse(center_x, center_y, radius_x, radius_y, rotation, r, g, b, alpha))

    return nbrs


image_x = 256
image_y = 256


@jit(nopython=True)
def ellipse_contains_point(ellipse, x, y):
    cosa = math.cos(ellipse.rotation)
    sina = math.sin(ellipse.rotation)
    dd = ellipse.radius_x / 2 ** 2
    DD = ellipse.radius_y / 2 ** 2

    a = math.pow(cosa * (x - ellipse.center_x) + sina * (y - ellipse_center_y), 2)
    b = math.pow(sina * (x - ellipse.center_x) - cosa * (y - ellipse.center_y), 2)
    ellipse=(a/dd)+(b/DD)

    if ellipse <= 1:
        return True
    else:
        return False


@jit(nopython=True)
def ellipse_bounding_box(ellipse):
    pass


@jit(nopython=True)
def image_plus_ellipse(image, ellipse):
    pass


@jit(nopython=True)
def image_distance(image, reference_image):
    pass
