"""Pipeline exception classes."""


class VideoNotFoundError(FileNotFoundError):
    """Raised when the specified video file does not exist."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Video file not found: {path}")


class UnsupportedFormatError(ValueError):
    """Raised when the video file format is not supported."""

    def __init__(self, path: str, supported: list[str]):
        self.path = path
        self.supported = supported
        super().__init__(
            f"Unsupported format for '{path}'. "
            f"Supported: {', '.join(supported)}"
        )


class VideoReadError(RuntimeError):
    """Raised when OpenCV cannot read the video file."""

    def __init__(self, path: str, reason: str = ""):
        self.path = path
        msg = f"Failed to read video: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)
