def slugify(text: str) -> str:
    """Convert a title into a URL-style slug."""

    return text.lower().replace(" ", "-")
