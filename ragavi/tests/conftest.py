import os
import pytest

@pytest.fixture
def ms():
    return ("/home/lexya/Documents/test_gaintables/ms/1491291289.1ghz.1.1ghz.4hrs.ms")

@pytest.fixture
def in_dir():
    return "/home/lexya/Documents/test_gaintables"


@pytest.fixture
def out_dir(in_dir):
    return os.path.join(in_dir, "outputs")

# @pytest.fixture
# def plotting_data():
#     x = range(10)
#     y = [_**2 for _ in x]
#     return dict(x=x, y=y)