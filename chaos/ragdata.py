import re
import numpy as np
import dask.array as da

from casacore.tables import table
from dataclasses import dataclass, field
from typing import Any

#for testing purposes
from ipdb import set_trace
import daskms as xm
import xarray as xr


class MsData:
    """
    Get some general data that will be required in the Plotter.
    This does not include user selected data and actual data columns
    More like the passive data only.
    """
    def __init__(self, ms_name):
        self.ms_name = ms_name
        self._colnames = None
        self._start_time = None
        self._end_time = None
        self._telescope = None
        self._ant_names = None
        self._ant_ids = None
        self._ant_map = None
        self._field_names = None
        self._field_ids = None
        self._field_map = None
        self._freqs = None
        self._spws = None
        self._num_spws = None
        self._num_chans = None
        self._corr_types = None
        self._corr_product = None
        self._num_corrs = None
        self._corr_map = None
        self._scans = None

        """
        TODO: Add in
            - x-data column name
            - y-data column name
            - x-data
            - y-data
            - iter-data column name
            - color-data column name
            - maybe a util function that converts input name to actual data column name
        """
    def initialise_data(self):
        with table(self.ms_name, ack=False) as self._ms:
            self._process_antenna_table()
            self._process_field_table()
            self._process_frequency_table()
            self._process_observation_table()
            self._process_polarisation_table()
            self._get_scan_table()
            self._colnames = self._ms.colnames()
            self._ms.close()
    
    def _process_observation_table(self):
        try:
            with table(self._ms.getkeyword("OBSERVATION"), ack=False) as sub:
                self._start_time, self._end_time = sub.getcell("TIME_RANGE",0)
                self._telescope = sub.getcell("TELESCOPE_NAME", 0)
        except RuntimeError:
            pass

    def _process_antenna_table(self):
        try:
            with table(self._ms.getkeyword("ANTENNA"), ack=False) as sub:
                self._ant_names = sub.getcol("NAME")
                self._ant_ids = sub.rownumbers()
                self._ant_map = {name: ids for ids, name in zip(
                    self._ant_ids, self._ant_names)}
        except RuntimeError:
            pass

    def _process_field_table(self):
        try:
            with table(self._ms.getkeyword("FIELD"), ack=False) as sub:
                self._field_names = sub.getcol("NAME")
                self._field_ids = sub.getcol("SOURCE_ID")
                self._field_map = {name: ids for ids, name in zip(
                    self._field_ids, self._field_names)}
        except RuntimeError:
            pass
    
    def _process_frequency_table(self):
        try:
            with table(self._ms.getkeyword("SPECTRAL_WINDOW"), ack=False) as sub:
                self._freqs = sub.getcol("CHAN_FREQ")
                self._spws = sub.rownumbers()
                self._num_spws = len(self._spws)
                self._num_chans = sub.getcell("NUM_CHAN", 0)
        except RuntimeError:
            pass

    def _process_polarisation_table(self):
        stokes_types = np.array(["I", "Q", "U", "V", "RR", "RL", "LR", "LL", "XX", "XY",
                        "YX", "YY", "RX", "RY", "LX", "LY", "XR", "XL", "YR",
                        "YL", "PP", "PQ", "QP", "QQ", "RCircular", "LCircular",
                        "Linear", "Ptotal", "Plinear", "PFtotal", "PFlinear",
                        "Pangle"])
        try:                        
            with table(self._ms.getkeyword("POLARIZATION"), ack=False) as sub:
                self._corr_types = sub.getcell("CORR_TYPE", 0)
                self._num_corrs = sub.getcell("NUM_CORR", 0)
                self._corr_product = sub.getcol("CORR_PRODUCT")

            self._corr_types = stokes_types[self._corr_types-1]
            if self._corr_types.size == 2 and self._corr_types.ndim==2:
                self._corr_types = self._corr_types[0]
            self._corr_map = {name: ids for ids, name in enumerate(self._corr_types)}
        except RuntimeError:
            pass

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
    def spws(self):
        return self._spws

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
    def ant_map(self):
        return self._ant_map
    
    @property
    def reverse_ant_map(self):
        return {idx: name for name, idx in self._ant_map.items()}
    
    @property
    def field_names(self):
        return self._field_names
    
    @property
    def field_ids(self):
        return self._field_ids

    @property
    def field_map(self):
        return self._field_map

    @property
    def reverse_field_map(self):
        return {idx: name for name, idx in self._field_map.items()}

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
    def corr_map(self):
        return self._corr_map
    
    @property
    def reverse_corr_map(self):
        return {idx: name for name, idx in self._corr_map.items()}

    @property
    def scans(self):
        return self._scans
    
    @property
    def colnames(self):
        return self._colnames




@dataclass
class Genargs:
    version: str
    msname: str
    chunks: str = None
    mem_limit: str = None
    ncores: str = None


@dataclass
class Axargs:
    xaxis: str
    yaxis: str
    data_column: str
    ms_obj: Any
    msdata: Any
    xdata_col: str = field(init=False)
    ydata_col: str = field(init=False)
    xdata: da.array = None
    ydata: da.array = None
    flags: da.array = None
    errors: da.array = None

    def __post_init__(self):
        #Get the proper name for the data column first before getting other names
        self.yaxis = self.translate_y(self.yaxis)
        self.data_column = self.get_colname(self.data_column, self.data_column)
        self.xdata_col = self.get_colname(self.xaxis, self.data_column)
        self.ydata_col = self.get_colname(self.yaxis, self.data_column)
        self.ydata = self.ms_obj[self.ydata_col]
        if self.xaxis in ["channel", "frequency"]:
            self.xdata = self.msdata.active_channels
        else:
            self.xdata = self.ms_obj[self.xdata_col]
        
    def translate_y(self, ax):
        axes = {}
        axes["a"] = axes["amp"] = "amplitude"
        axes["i"] = axes["imag"] = "imaginary"
        axes["p"] = "phase"
        axes["r"] = "real"
        return axes.get(ax) or ax
    
    
    def update_data(self, kwargs):
        """
        kwargs: :obj:`dict` containing the name of the data to add and its 
        value
        """
        for key, value in kwargs.items():
            self.__setattr__(key, value)


    def colname_map(self, data_column):
        axes = dict()
        axes["time"] = "TIME"
        axes["spw"] = "DATA_DESC_ID"
        axes["corr"] = "corr"
        axes.update({key: data_column for key in ("a", "amp", "amplitude",
                        "i", "imag", "imaginary", "p", "phase", "r", "real")})
        axes.update({key: "UVW" for key in ("uvdist", "UVdist", "uvdistance",
                        "uvdistl", "uvdist_l", "UVwave", "uvwave")})
        axes.update({key: "ANTENNA1" for key in ("ant1", "antenna1")})
        axes.update({key: "ANTENNA2" for key in ("ant2", "antenna2")})
        axes.update({key: ("ANTENNA1", "ANTENNA2") for key in ("bl", "baseline")})
        axes.update({key: "chan" for key in ("chan", "channel", "freq",
                        "frequency")})
        return axes

    def get_colname(self, axis, data_column):
        cols = ["ANTENNA1", "ANTENNA2", "ARRAY_ID", "CPARAM", "CORRECTED_DATA",
                "DATA", "DATA_DESC_ID", "EXPOSURE", "FEED1", "FEED2", "FIELD_ID",
                "FLAG_ROW", "FLAG", "FLAG_CATEGORY", "INTERVAL", "MODEL_DATA",
                "OBSERVATION_ID", "CPARAM", "PROCESSOR_ID", "SCAN_NUMBER", "SIGMA",
                "SIGMA_SPECTRUM", "SPECTRAL_WINDOW_ID", "STATE_ID", "TIME",
                "TIME_CENTROID", "UVW", "WEIGHT", "WEIGHT_SPECTRUM"]
        
        col_maps = self.colname_map(data_column)

        if axis.upper() in cols:
            colname = axis.upper()
        elif re.search(r"\w*{}\w*".format(axis), ", ".join(cols), re.IGNORECASE) and len(axis)>1:
            colname = re.search(r"\w*{}\w*".format(axis),
                                ", ".join(cols), re.IGNORECASE).group()
        elif axis in col_maps:
            colname = col_maps[axis]
        else:
            colname = None
            print(f"{axis} column not found")
        return colname


@dataclass
class Selargs:
    antennas: str
    baselines: str
    corrs: str
    channels: str
    ddids: str
    fields: str
    taql: str
    t0: str
    t1: str


@dataclass
class Plotargs:
    cmap: str
    html_name: str = None
