import pytest
import numpy as np
import dask.array as da
import xarray as xr

from holy_chaos.chaos.processing import Processor, Chooser

@pytest.fixture
def complex_values():
    re = np.random.randint(0,20, 10)
    im = 1j * np.random.randint(0, 50, 10)
    return re + im

@pytest.fixture
def complex_values_d(complex_values):
    return da.from_array(complex_values, name="test-dat")

@pytest.fixture
def complex_values_x(complex_values_d):
    return xr.DataArray(data = complex_values_d,
                        coords={"x": np.arange(10)}, dims=("x",))

@pytest.fixture
def pro(complex_values_x):
    return Processor(complex_values_x)

@pytest.fixture
def uvw():
    return xr.DataArray(
        data=1+da.array([[np.arange(5)], [np.arange(5)],
                      [np.arange(5)]]).squeeze().T,
        coords={"rows": da.arange(5)},
        dims = ("rows", "uvw"),
        name="UVW"
    )

@pytest.fixture
def freqs():
    return 8+ 10 * np.arange(5)


##### START OF TESTS
def test_amplitude(complex_values, pro):
    assert np.all(np.abs(complex_values) == pro.calculate("amplitude").values)

def test_real(complex_values, pro):
    assert np.all(np.real(complex_values) == pro.calculate("real").values)

def test_imag(complex_values, pro):
    assert np.all(np.imag(complex_values) == pro.calculate("imaginary").values)


def test_phase(complex_values, pro):
    assert np.all(np.rad2deg(np.unwrap(np.angle(complex_values, deg=False))) 
                  == pro.calculate("phase").values)

def test_wrapped_phase(complex_values, pro):
    assert np.all(np.angle(complex_values, deg=True)
                  == pro.phase(unwrap=False).values)


def test_uvdistance(uvw):
    assert np.all(np.sqrt(np.square(uvw.values[:, :-1]).sum(axis=1))
                     == Processor.uv_distance(uvw))


def test_uvwavelength(uvw, freqs):
    assert np.all((np.sqrt(np.square(
        uvw.values[:, :-1, np.newaxis]).sum(axis=1)) / (3e8/freqs))
         == Processor.uv_wavelength(uvw, freqs).values)


@pytest.mark.parametrize(
    "mjdtime, hrtime", [(
        np.array([4998008435.988423, 4998008437.987587, 4998008443.985041]),
        ['2017-04-04T07:40:35', '2017-04-04T07:40:37', '2017-04-04T07:40:43']
    )])
def test_unix_timestamp(mjdtime, hrtime):
    """MJD TO UNIX converter
    from datetime import datetime, timezone
    [datetime.fromtimestamp(x).astimezone(
        timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") for x in uns]
    """
    print(Processor.unix_timestamp(mjdtime))
    assert np.datetime_as_string(Processor.unix_timestamp(mjdtime)).tolist() == hrtime

@pytest.mark.parametrize(
        "sel, knife",
        [(None, slice(None)),
        ("", slice(None)),
        ("4", 4),
        ("5,6,7", np.array([5,6,7])),
        ("5~7", slice(5, 8)),
        ("5:8:2", slice(5,8,2)),
        ("5:8", slice(5,8)),
        ("5:", slice(None, 5, None)),
        ("::10", slice(None, None, 10))
    ])
def test_chooser_knife(sel, knife):
    assert np.all(Chooser.get_knife(sel) == knife)


@pytest.fixture
def data_map():
    return {
        "m003": 3,
        "m005": 5,
        "m012": 12
    }

@pytest.mark.parametrize("ants, outs", [("m005","5"), ("3", "3"), ("m010", None)])
def test_nametoid(ants, outs, data_map):
    assert Chooser.nametoid(ants, data_map) == outs


"""
antenna = "m003, m012, m056, m010, 5, 9, 0"
baseline = "m003-m012, 0-10, m048-m045, m029-m077, 5-9"
field = "0, DEEP_2, 0252-712, 2, 0408-65"
spw = "0"
scan = "4, 10, 12, 67"
taql = None
time = "60, 3500"
uv_range = "8430_m"
# uv_range = "14000_l"

corr = Chooser.get_knife("0")
chan = Chooser.get_knife("::5")

ms_name = "/home/lexya/Documents/test_stimela_dir/msdir/1491291289.1ghz.1.1ghz.4hrs.ms"
msdata = MsData(ms_name)
msdata.initialise_data()

tstring = Chooser.form_taql_string(msdata, antenna="m003, m012, m056, m010, 5, 9, 0",
                                    baseline="m003-m012, 0-10, m048-m045, m029-m077, 5-9",
                                    field="0, DEEP_2, 0252-712, 2, 0408-65",
                                    spw="0",
                                    scan="4, 10, 12, 67",
                                    taql=None,
                                    time="60, 3500",
                                    uv_range="8430_m")
"""
