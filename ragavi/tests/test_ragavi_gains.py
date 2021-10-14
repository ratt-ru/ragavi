"""
params in fixures are given in iterable form and it means that the the fixture
will iterate over the params list and create 'a new fixture'. Notice the param
request the fixture now takes. It is a requrement by pytest if we parametrize
a fixture.

By parametrizing a fixure, we can use it in a test and the fixture will be executed
for each param. This way, we INDIRECTLY PARAMETRISE A TEST FUNCTION WITH A FIXTURE.
Note fixtures use a parametrisation value by setting indirect=True

See: 
 * https://medium.com/opsops/deepdive-into-pytest-parametrization-cb21665c05b9
 * https://docs.pytest.org/en/6.2.x/example/parametrize.html#paramexamples
 * https://stackoverflow.com/questions/35413134/what-does-indirect-true-false-in-pytest-mark-parametrize-do-mean
 * https://docs.pytest.org/en/latest/reference/reference.html#pytest-fixture-api
 * https://stackoverflow.com/questions/42014484/pytest-using-fixtures-as-arguments-in-parametrize#42599627
 * https://stackoverflow.com/questions/44047224/pytest-getting-the-value-of-fixture-parameter
"""

import pytest
import os

from glob import glob
from ragavi.gains import main
from ragavi.arguments import gains_argparser

table_names = [
    "1491291289.B0",
    "1491291289.F0",
    "1491291289.K0",
    "1491291289.G0",
    "3c391_ctm_mosaic_10s_spw0.B0",
    "4k_pcal3_1-1576599977_fullpol_1318to1418MHZ-single-3pcal_1.Df",
    "4k_pcal3_1-1576599977_fullpol_1318to1418MHZ-single-3pcal_1.Kcrs",
    "4k_pcal3_1-1576599977_fullpol_1318to1418MHZ-single-3pcal_1.Xf",
    "4k_pcal3_1-1576599977_fullpol_1318to1418MHZ-single-3pcal_1.Xref",
    "J0538_floi-beck-1pcal.Xfparang",
    "short-gps1-1gc1_primary_cal.G0",
    "VLBA_0402_3C84.B0",
    # "workflow2-1576687564_sdp_l0-1gc1_primary.B0"
]


@pytest.fixture(params=table_names)
def tables(in_dir, request):
    return os.path.join(in_dir, request.param)


@pytest.fixture(params=table_names)
def outputs(out_dir, request):
    return os.path.join(out_dir, f"{request.param}_ap.html")


def test_basic_out(tables):
    if table_names[0] not in tables:
        pytest.skip()
    assert main(gains_argparser, ["-t", tables]) == 0


def test_custom_htmlname(tables, out_dir):
    if table_names[0] not in tables:
        pytest.skip()
    out_name = os.path.join(out_dir, "wapi.html")
    assert main(gains_argparser, ["-t", tables, "--htmlname", out_name]) == 0
    assert os.path.isfile(out_name) == True


def test_default_outputs(tables, outputs):
    assert main(gains_argparser, ["-t", tables, "--htmlname", outputs]) == 0
    assert os.path.isfile(outputs) == True


def test_multiple_inputs(tables, in_dir, out_dir):
    out_name = os.path.join(out_dir, "combined.html")
    select_tabs = [os.path.join(in_dir, table_name) for table_name in table_names[:4]]
    assert main(gains_argparser, ["-t"] + select_tabs + ["--htmlname", out_name]) == 0
    assert os.path.isfile(out_name) == True


@pytest.mark.parametrize("ext", ["png", "svg", "pdf"])
def test_custom_image_name(tables, ext, out_dir):
    if table_names[0] not in tables:
        pytest.skip()
    out_name = os.path.join(out_dir, "wapi")
    assert main(gains_argparser, ["-t", tables, "--plotname", f"{out_name}.{ext}"]) == 0
    assert len(glob(f"{out_name}*.{ext}")) > 0
