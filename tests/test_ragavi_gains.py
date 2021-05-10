import pytest
import os

from glob import glob
from holy_chaos.chaos.gains import main
from holy_chaos.chaos.arguments import gains_argparser


PATH = "/home/lexya/Documents/chaos_project/holy_chaos/tests/gain_tables"
def tables():
    tabs =  [
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
        "workflow2-1576687564_sdp_l0-1gc1_primary.B0"
        ]
    return [os.path.join(PATH, table) for table in tabs]

def outputs():
    return [f"{table}_ap.html" for table in tables()]

def test_cleanup():
    [os.remove(out) for out in glob(f"{PATH}/*.html")]
    [os.remove(out) for out in glob(f"{PATH}/*.png")]
    [os.remove(out) for out in glob(f"{PATH}/*.pdf")]
    [os.remove(out) for out in glob(f"{PATH}/*.svg")]
    assert True


# @pytest.mark.skip
@pytest.mark.parametrize(
    "ins, outs",
    [(table, output) for table, output in zip(tables(), outputs())]
)
def test_default_outputs(ins, outs):
    assert main(gains_argparser, ["-t", ins]) == 0
    assert os.path.isfile(os.path.join(PATH, outs)) == True


# @pytest.mark.skip
def test_multiple_inputs():
    assert main(gains_argparser, ["-t"]+tables()[:4]) == 0
    assert all([os.path.isfile(out) for out in outputs()[:4]]) == True

def test_custom_htmlname():
    assert main(gains_argparser, ["-t", tables()[0], "--htmlname", f"{PATH}/wapi.html"]) == 0
    assert len(glob(f"{PATH}/wapi*.html")) == True


@pytest.mark.parametrize("tab, ext",
    [(tables()[0], ext)for ext in ".png .svg .pdf".split()])
def test_custom_image_name(tab, ext):
    assert main(gains_argparser, ["-t", tab, "--plotname", f"{PATH}/wapi{ext}"]) == 0
    assert len(glob(f"{PATH}/wapi*{ext}")) > 0

