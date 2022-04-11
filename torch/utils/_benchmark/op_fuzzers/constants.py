import enum

import numpy as np


class Fuzzers(enum.Enum):
    UNARY = 0
    BINARY = 1


class Scale(enum.Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    LARGER = 3


MIN_DIM_SIZE = 8
MAX_DIM_SIZE = {
    Scale.SMALL: 128,
    Scale.MEDIUM: 1024,
    Scale.LARGE: 16 * 1024 ** 2,
    Scale.LARGER: 64 * 1024 ** 2,
}


def pow_2_values(lower, upper):
    r = range(int(np.log2(lower)), int(np.log2(upper)) + 1)
    values = tuple(2 ** i for i in r)
    return {v: 1 / len(values) for v in values}


POINTWISE_MIN_ELEMENTS = {
    Scale.SMALL: 0,
    Scale.MEDIUM: 128,
    Scale.LARGE: 128 * 1024,
    Scale.LARGER: 1024 ** 2,
}

POINTWISE_MAX_ELEMENTS = {
    Scale.SMALL: 16 * 1024,
    Scale.MEDIUM: 4 * 1024 ** 2,
    Scale.LARGE: 32 * 1024 ** 2,
    Scale.LARGER: 256 * 1024 ** 2,
}
