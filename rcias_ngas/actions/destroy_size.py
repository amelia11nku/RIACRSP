"""A1 preregistered conservative sizes, with explicit tiny-instance collisions."""
import math

SIZE_FRACTIONS = {"small": .08, "medium": .15, "large": .22}


def destroy_count(num_operations, size):
    if num_operations < 1 or size not in SIZE_FRACTIONS:
        raise ValueError("Invalid operation count or size")
    return min(num_operations, max(2, math.floor(num_operations * SIZE_FRACTIONS[size] + .5)))
