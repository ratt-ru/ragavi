from casacore.tables import table
import numpy as np
from ipdb import set_trace

class MsData:
    def __init__(self, ms_name):
        self.ms_name = ms_name
        self._start_time = None
        self._end_time = None
        self._telescope = None
        self._ant_names = None
        self._ant_names = None
        self._ant_ids = None
        self._field_names = None
        self._field_ids = None
        self._freqs = None
        self._num_spws = None
        self._num_chans = None
        self._corr_types = None
        self._corr_product = None
        self._num_corrs = None
        self._scans = None

    def initialise_data(self):
        with table(self.ms_name, ack=False) as self._ms:
            self._process_antenna_table()
            self._process_field_table()
            self._process_frequency_table()
            self._process_observation_table()
            self._process_polarisation_table()
            self._get_scan_table()
            self._ms.close()
    
    def _process_observation_table(self):
        with table(self._ms.getkeyword("OBSERVATION"), ack=False) as sub:
            self._start_time, self._end_time = sub.getcell("TIME_RANGE",0)
            self._telescope = sub.getcell("TELESCOPE_NAME", 0)

    def _process_antenna_table(self):
        with table(self._ms.getkeyword("ANTENNA"), ack=False) as sub:
            self._ant_names = sub.getcol("NAME")
            self._ant_ids = list(range(len(self._ant_names)))

    def _process_field_table(self):
        with table(self._ms.getkeyword("FIELD"), ack=False) as sub:
            self._field_names = sub.getcol("NAME")
            self._field_ids = sub.getcol("SOURCE_ID")
    
    def _process_frequency_table(self):
        with table(self._ms.getkeyword("SPECTRAL_WINDOW"), ack=False) as sub:
            self._freqs = sub.getcol("CHAN_FREQ")
            self._num_spws = self._freqs.shape[0]
            self._num_chans = sub.getcell("NUM_CHAN", 0)

    def _process_polarisation_table(self):
        stokes_types = np.array(["I", "Q", "U", "V", "RR", "RL", "LR", "LL", "XX", "XY",
                        "YX", "YY", "RX", "RY", "LX", "LY", "XR", "XL", "YR",
                        "YL", "PP", "PQ", "QP", "QQ", "RCircular", "LCircular",
                        "Linear", "Ptotal", "Plinear", "PFtotal", "PFlinear",
                        "Pangle"])
        with table(self._ms.getkeyword("POLARIZATION"), ack=False) as sub:
            self._corr_types = sub.getcell("CORR_TYPE", 0)
            self._num_corrs = sub.getcell("NUM_CORR", 0)
            self._corr_product = sub.getcol("CORR_PRODUCT")

        self._corr_types = stokes_types[self._corr_types-1]
        if self._corr_types.size == 2 and self._corr_types.ndim==2:
            self._corr_types = self._corr_types[0]

    def _get_scan_table(self):
        self._scans = np.unique(self._ms.getcol("SCAN_NUMBER"))
    
    def calc_baselines(self):
        # unique baselines
        self._num_bl = (self.num_ants * (self.num_ants-1))/2

        ant2 = self._ms.getcol("ANTENNA1")
        ant1 = self._ms.getcol("ANTENNA2")
        #TODO: find an algorithm to caluclate the unique baselines



    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self): 
        return self._end_time

    @property
    def table_type(self):
        pass

    @property
    def num_chans(self):
        return self._num_chans

    @property
    def num_corrs(self):
        return self._num_corrs

    @property
    def num_spws(self):
        return self._num_spws

    @property
    def num_ants(self):
        "Number of unique antennas"
        return len(self._ant_names)
    
    @property
    def num_fields(self):
        "Number of Unique fields"
        return len(self._field_names)

    @property
    def num_scans(self):
        "Number of Unique scans"
        return len(self._scans)
    
    @property
    def num_baselines(self):
        pass
    
    @property
    def telescope(self):
        return self._telescope

    @property
    def ant_names(self):
        return self._ant_names
    
    @property
    def ant_ids(self):
        return self._ant_ids
    
    @property
    def field_names(self):
        return self._field_names
    
    @property
    def field_ids(self):
        return self._field_ids

    @property
    def freqs(self):
        return self._freqs
    
    @property
    def corr_types(self):
        return self._corr_types
    
    @property
    def corr_product(self):
        return self._corr_product

    @property
    def scans(self):
        return self._scans




"""
if __name__ == "__main__":
    ms_name = "/home/lexya/Documents/test_stimela_dir/msdir/1491291289.1ghz.1.1ghz.4hrs.ms"
    msdata = MsData(ms_name)
    msdata.initialise_data()
    set_trace()
"""