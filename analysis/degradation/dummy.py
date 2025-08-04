from typing import Generator, Tuple


def apply(input_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    Applies no degradation and simply returns the original video 3 times for testing.

    Yields:
        Tuple of (output_path, label) for 3 dummy videos.
    """
    for i in range(1, 4):
        yield input_path, f"Dummy {i}"
