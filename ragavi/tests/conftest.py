import os
import pytest

HOME = os.getenv("HOME")


@pytest.fixture(scope="session")
def in_dir():
    return os.path.join(HOME, "Documents", "test_gaintables")


@pytest.fixture(scope="session")
def out_dir(in_dir):
    return os.path.join(in_dir, "outputs")


@pytest.fixture(scope="session")
def test_ms(in_dir):
    return os.path.join(in_dir, "ms", "1491291289.1ghz.1.1ghz.4hrs.ms")

