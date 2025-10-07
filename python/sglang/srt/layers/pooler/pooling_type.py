from enum import IntEnum


class PoolingType(IntEnum):
    LAST = 0
    CLS = 1
    MEAN = 2
