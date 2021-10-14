import os
import pytest

HOME = os.getenv("HOME")


@pytest.fixture(scope="session", autouse=True)
def in_dir():
    return os.path.join(HOME, "Documents", "test_gaintables")


@pytest.fixture(scope="session", autouse=True)
def out_dir(in_dir):
    return os.path.join(in_dir, "outputs")


@pytest.fixture(scope="session")
def test_ms(in_dir):
    return os.path.join(in_dir, "ms", "1491291289.1ghz.1.1ghz.4hrs.ms")


@pytest.mark.parametrize("ext", ["html", "png", "pdf", "svg", "log"])
def test_cleanup(ext, in_dir, out_dir):
    [os.remove(out) for out in glob(os.path.join(in_dir, f"*.{ext}"))]
    [os.remove(out) for out in glob(os.path.join(in_dir, "ms", f"*.{ext}"))]
    [os.remove(out) for out in glob(os.path.join(out_dir, f"*.{ext}"))]
    assert True
