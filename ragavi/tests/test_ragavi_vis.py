import os
import glob
import pytest
from ragavi.vis import get_ms, main, vis_argparser

cat_axes = (
    "ant",
    "antenna",
    "ant1",
    "antenna1",
    "ant2",
    "antenna2",
    "bl",
    "baseline",
    "corr",
    "field",
    "scan",
    "spw",
)

xaxes = (
    "ant1",
    "antenna1",
    "ant2",
    "antenna2",
    "amp",
    "amplitude",
    "chan",
    "channel",
    "freq",
    "frequency",
    "imag",
    "imaginary",
    "phase",
    "real",
    "scan",
    "time",
    "uvdist",
    "UVdist",
    "uvdistance",
    "uvdistl",
    "uvdist_l",
    "UVwave",
    "uvwave",
)

yaxes = (
    "amp",
    "ampl",
    "amplitude",
    "freq",
    "chan",
    "frequency",
    "real",
    "imaginary",
    "imag",
    "im",
    "re",
    "ph",
    "phase",
)


class TestAxes:
    """Test that the various x, y, iteration and colouration axes work well"""

    @pytest.mark.parametrize("xaxis", xaxes)
    def test_xaxes(self, xaxis, test_ms):
        if xaxis.startswith("amp"):
            pytest.skip("Skipping because both x and y are amp")
        ins = f"--ms {test_ms} -x {xaxis} -y amp".split()
        assert main(vis_argparser, ins) == 0

    @pytest.mark.parametrize("yaxis", yaxes)
    def test_yaxes(self, yaxis, test_ms):
        ins = f"--ms {test_ms} -x time -y {yaxis}".split()
        assert main(vis_argparser, ins) == 0

    @pytest.mark.parametrize("iaxis", cat_axes)
    def test_iaxes(self, iaxis, test_ms):
        ins = f"--ms {test_ms} -x time -y amp --iter-axis {iaxis}".split()
        assert main(vis_argparser, ins) == 0

    @pytest.mark.parametrize("caxis", cat_axes)
    def test_caxes(self, caxis, test_ms):
        ins = f"--ms {test_ms} -x time -y amp --colour-axis {caxis}".split()
        assert main(vis_argparser, ins) == 0


class TestPlotArgs:
    @pytest.mark.parametrize("caxis", cat_axes)
    def test_cosmetics(self, caxis, out_dir, test_ms):
        fname = os.path.join(out_dir, f"{caxis}_test_post.html")
        ins = (
            f"--ms {test_ms} -x time -y amp --colour-axis {caxis}"
            + " -x time -y amp "
            + "-ch 100 -cw 200 --cmap glasbey_dark "
            + f"--cols 3 -o {fname} --ymin 40 --ymax 50"
        ).split()
        assert main(vis_argparser, ins) == 0
        assert len(glob.glob(fname)) == 1

    def test_logfile_creation(self):
        pass

    def test_cachefile_creation(self):
        pass
