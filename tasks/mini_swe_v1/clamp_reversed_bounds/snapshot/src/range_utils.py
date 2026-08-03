def clamp(value: int, lower: int, upper: int) -> int:
    """Return value constrained to the inclusive bounds."""

    if lower > upper:
        lower, upper = upper, lower
    return max(lower, min(value, upper))
