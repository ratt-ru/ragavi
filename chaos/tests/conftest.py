import pytest

@pytest.fixture
def plotting_data():
    x = range(10)
    y = [_**2 for _ in x]
    return dict(x=x, y=y)