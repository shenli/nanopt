def merge_config(defaults: dict[str, int], overrides: dict[str, int]) -> dict[str, int]:
    """Apply caller overrides to default values."""

    result = defaults
    result.update(overrides)
    return result
