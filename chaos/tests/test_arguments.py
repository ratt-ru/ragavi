from chaos.arguments import *

from ipdb import set_trace

def test_gains_argparser():
    # check if items are lists
    pass

ga = gains_argparser()
gargs = ["-t", "a.ms", "b.ms", "c.ms",
         "-a", "0,1,10,13", "m000,m10,m063", "7",
         "-c", "0", "1", "0,1",
         "--ddid", "0",
         "-f", "0", "DEEP2", "J342-342,X034-123",
         "--t0", "500", "1000", "2000",
         "--t1", "6000", "7000", "7500",
         "-y", "a,p,r", "r,i", "i,a",
         "-x", "chan", "time", "time"]
gh = ga.parse_args(gargs)
set_trace()
