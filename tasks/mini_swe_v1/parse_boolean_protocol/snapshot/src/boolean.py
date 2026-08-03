def parse_bool(value: str) -> bool:
    """Parse the documented wire-format booleans."""

    if value in {"true", "yes", "1"}:
        return True
    if value in {"false", "no", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value}")
