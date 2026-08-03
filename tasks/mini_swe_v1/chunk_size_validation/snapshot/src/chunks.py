def chunks(values: list[int], size: int) -> list[list[int]]:
    """Split values into consecutive chunks."""

    if not values:
        return [[]]
    return [values[index : index + size] for index in range(0, len(values), size)]
