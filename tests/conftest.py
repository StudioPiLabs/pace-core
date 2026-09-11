"""Shared pytest fixtures."""
import pytest


@pytest.fixture
def fails():
    """Collect failure messages; the test fails at teardown if any were added."""
    bucket: list[str] = []
    yield bucket
    if bucket:
        pytest.fail("\n".join(bucket))
