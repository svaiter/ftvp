from enum import Enum

class OptMethod(Enum):
    SPLIT_NEWTON = 1
    SPLIT_DESCENT = 2

class UpdateStrategy(Enum):
    UPDATE_STD = 0
    UPDATE_CONSTANT = 1
    UPDATE_VARYING = 2
