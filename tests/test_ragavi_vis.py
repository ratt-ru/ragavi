import os
import pytest
from chaos.vis import get_ms, main, vis_argparser

@pytest.fixture
def ms():
    return ("/home/lexya/Downloads/radioastro/tutorial_docs/"+
            "clueless_calibration/1491291289.1ghz.1.1ghz.4hrs.ms")

# class TestGetMs:
        
#     def test_get_ms(self):
#         assert True

#     def test_corr_iter(self):
#         assert True

#     def test_antenna_iter(self):
#         assert True

#     def test_baseline_iter(self):
#         assert True


# assert main(gains_argparser, ["-t", tab, "--plotname", f"{PATH}/wapi{ext}"]) == 0

@pytest.mark.usefixtures("ms")
class TestAxes:
    """Test that the various x, y, iteration and colouration axes work well"""
    def cat_axes():
        return ("ant", "antenna", "ant1", "antenna1", "ant2", "antenna2", "bl",
        "baseline", "corr", "field", "scan", "spw")
    
    def xaxes():
        return "ant1", "antenna1", "ant2", "antenna2", "amp", "amplitude", "chan",
        "channel", "freq", "frequency", "imag", "imaginary", "phase", "real",
        "scan", "time", "uvdist", "UVdist", "uvdistance", "uvdistl",
        "uvdist_l", "UVwave", "uvwave"
    
    def yaxes():
        return "amp", "ampl", "amplitude", "freq", "chan", "frequency", "real",
        "imaginary", "imag", "im", "re", "ph", "phase"

    @pytest.mark.parametrize("xaxis", xaxes())
    def test_xaxes(self, xaxis, ms):
        # amp axis will fail coz yaxis is also amp
        ins = f"--ms {ms} -x {xaxis} -y amp".split()
        assert main(vis_argparser, ins) == 0

    @pytest.mark.parametrize("yaxis", yaxes())
    def test_yaxes(self, yaxis, ms):
        ins = f"--ms {ms} -x time -y {yaxis}".split()
        assert main(vis_argparser, ins) == 0

    @pytest.mark.parametrize("iaxis", cat_axes())
    def test_iaxes(self, iaxis, ms):
        ins = f"--ms {ms} -x time -y amp --iter-axis {iaxis}".split()
        assert main(vis_argparser, ins) == 0
    
    @pytest.mark.parametrize("caxis", cat_axes())
    @pytest.mark.caxes
    def test_caxes(self, caxis, ms):
        ins = f"--ms {ms} -x time -y amp --colour-axis {caxis}".split()
        assert main(vis_argparser, ins) == 0

#  pytest -q test_ragavi_vis.py
#  pytest -q --lf test_ragavi_vis.py li
#  pytest -m caxes test_ragavi_vis.py run specific test