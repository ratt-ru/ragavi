import re
import numpy as np
import dask.array as da
import daskms as xm
import xarray as xr

from dataclasses import dataclass, field
from difflib import get_close_matches
from itertools import combinations
from psutil import cpu_count, virtual_memory
from typing import Any
from casacore.tables import table

from chaos.exceptions import TableNotFound
from chaos.lograg import logging
from chaos.utils import pair

snitch = logging.getLogger(__name__)

stokes_types = np.array([
    "I", "Q", "U", "V", "RR", "RL", "LR", "LL", "XX", "XY",
    "YX", "YY", "RX", "RY", "LX", "LY", "XR", "XL", "YR",
    "YL", "PP", "PQ", "QP", "QQ", "RCircular", "LCircular",
    "Linear", "Ptotal", "Plinear", "PFtotal", "PFlinear", "Pangle"])

class MsData:
    """
    Get some general data that will be required in the Plotter.
    This does not include user selected data and actual data columns
    More like the passive data only.
    """
    def __init__(self, ms_name):
        self.ms_name = ms_name
        self.active_channels = None
        self.active_antennas = []
        self.active_fields = []
        self.active_corrs = None
        self.active_spws = []
        self.taql_selector = None
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
        self._table_type = "ms"
        self._num_rows = None
        self.initialise_data()

    def initialise_data(self):
        with table(self.ms_name, ack=False) as self._ms:
            self._process_antenna_table()
            self._process_field_table()
            self._process_frequency_table()
            self._process_observation_table()
            self._process_polarisation_table()
            self._get_scan_table()
            self._colnames = self._ms.colnames()
            self._num_rows = self._ms.nrows()
            if "VisCal" in self._ms.keywordnames():
                self._table_type = self._ms.getkeyword("VisCal").split()[0]
            self._ms.close()
    
    def _process_observation_table(self):
        try:
            with table(self._ms.getkeyword("OBSERVATION"), ack=False) as sub:
                self._start_time, self._end_time = sub.getcell("TIME_RANGE",0)
                if self._start_time == self._end_time:
                    self._start_time = self._ms.getcell("TIME", 0)
                    self._end_time = self._ms.getcell("TIME", self._ms.nrows()-1)
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
        """Uses daskms to get frequency data as xarray"""
        try:
            sub = xm.xds_from_table(
                self.ms_name + "::SPECTRAL_WINDOW", columns="CHAN_FREQ")[0]
            self._freqs = sub.CHAN_FREQ
            self._spws = self._freqs.row
            self._num_spws = self._freqs.row.size
            self._num_chans = self._freqs.chan.size
        except RuntimeError:
            pass

    def _process_polarisation_table(self):
        try:                        
            with table(self._ms.getkeyword("POLARIZATION"), ack=False) as sub:
                self._corr_types = sub.getcell("CORR_TYPE", 0)
                self._num_corrs = sub.getcell("NUM_CORR", 0)
                self._corr_product = sub.getcol("CORR_PRODUCT")

            self._corr_types = stokes_types[self._corr_types-1]
            if self._corr_types.size == 2 and self._corr_types.ndim==2:
                self._corr_types = self._corr_types[0]

            self._corr_map = {name: ids for ids, name in enumerate(self._corr_types)}
            #update here for custom names
            self._corr_map.update({_: f"0,{self.num_corrs-1}" for _ in 
                                    ["DIAGONAL", "DIAG"]})
            self._corr_map.update({_: "1,2" for _ in ["OFF-DIAGONAL", "OFF-DIAG"]})
        except RuntimeError:
            # raise TableNotFound(f"No Polarization table for {self.ms_name}")
            self._num_corrs = self._ms.getcell("FLAG",0).shape[-1]

    def _get_scan_table(self):
        self._scans = np.unique(self._ms.getcol("SCAN_NUMBER"))
    
    def _make_bl_map(self):
        # unique baselines        
        bl_combis = list(set(combinations(range(self.num_ants), 2)))
        bl_combis.sort()
        return {f"{self.reverse_ant_map[a1]}-{self.reverse_ant_map[a2]}": 
                    pair(a1, a2) for a1, a2 in bl_combis}
    
    @property
    def bl_map(self):
        return self._make_bl_map()
    
    @property
    def reverse_bl_map(self):
        return {v: k for k, v in self.bl_map.items()}

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self): 
        return self._end_time

    @property
    def table_type(self):
        return self._table_type

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
    def num_rows(self):
        return self._num_rows

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
        return len(self.bl_map)
    
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
    def scan_map(self):
        return {s: s for s in self._scans}
    
    @property
    def reverse_scan_map(self):
        return self.scan_map
    
    @property
    def colnames(self):
        return self._colnames


class CubicalTableData:
    def __init__(self, ms_name, ants=None, fields=None, corr1s=None):
        self.ms_name = ms_name
        self.ant_names = ants
        self.field_names = [str(f) for f in fields]
        self.corr1s = [c.upper() for c in corr1s]
        self.corrs = None
        self.active_antennas = None
        self.active_corrs = []
        self.active_corr1s = None
        self.active_corr2s = None
        self.active_fields = None
        self.active_spws = [0]

    @property
    def ant_map(self):
        return {a: self.ant_names.index(a) for a in self.ant_names}

    @property
    def reverse_ant_map(self):
        return {v: k for k, v in self.ant_map.items()}

    @property
    def field_map(self):
        return {a: self.field_names.index(a) for a in self.field_names}

    @property
    def reverse_field_map(self):
        return {v: k for k, v in self.field_map.items()}

    @property
    def corr_names(self):
        return [f"{c1}{c2}".upper() for c1, c2 in product(self.corr1s, self.corr1s)]

    @property
    def corr_map(self):
        return {a: self.corr_names.index(a) for a in self.corr_names}

    @property
    def reverse_corr_map(self):
        return {v: k for k, v in self.corr_map.items()}

    @property
    def num_corr1s(self):
        return len(self.corr1s)

    @property
    def num_ants(self):
        return len(self.ant_names)

    @property
    def num_fields(self):
        return len(self.fields)


class QuarticalTableData:
    def __init__(self, ms_name, ant_names=None, field_names=None,
                 corr_names=None, scans=None, spws=None):
        self.ms_name = ms_name
        self.ant_names = ant_names.tolist()
        self.field_names = field_names
        self.corr_names = corr_names.tolist()
        self.scans = scans
        self.spws = xr.DataArray(spws)
        self.active_corrs = []
        self.active_spws = [0]
        self.active_fields = []

    @property
    def ant_map(self):
        return {a: self.ant_names.index(a) for a in self.ant_names}

    @property
    def reverse_ant_map(self):
        return {v: k for k, v in self.ant_map.items()}

    @property
    def field_map(self):
        return {a: self.field_names.index(a) for a in self.field_names}

    @property
    def reverse_field_map(self):
        return {v: k for k, v in self.field_map.items()}

    @property
    def corr_map(self):
        return {a: self.corr_names.index(a) for a in self.corr_names}

    @property
    def reverse_corr_map(self):
        return {v: k for k, v in self.corr_map.items()}

    @property
    def num_corrs(self):
        return len(self.corr_names)

    @property
    def num_ants(self):
        return len(self.ant_names)

    @property
    def num_fields(self):
        return len(self.field_names)

    @property
    def num_spws(self):
        return len(self.spws)

    @property
    def num_scans(self):
        return len(self. scans)


@dataclass
class Genargs:
    version: str = "0.0.1"
    msname: str = None
    chunks: str = None
    mem_limit: str = None
    ncores: str = None

    def __post_init__(self):
        self.ncores, self.mem_limit = self.resource_defaults(
            self.mem_limit, self.ncores)
        self.chunks = int(self.chunks) if self.chunks is not None else self.chunks

    def get_mem(self):
        """Get 90% of the memory available"""
        GB = 2**30
        mem = virtual_memory().total
        snitch.info(f"Total RAM size: ~{(mem / GB):.2f} GB")
        mem = int((mem*0.9)//GB)
        return mem

    def get_cores(self):
        """Get half the number of cores available. Max at 10"""
        snitch.info(f"Total number of Cores: {cpu_count()}")
        cores = cpu_count() // 2
        cores = 10 if cores > 10 else cores
        return cores

    def resource_defaults(self, ml, nc):
        cores = self.get_cores() - 1
        mem_size = self.get_mem()

        ml = 2 if ml is None else ml
        nc = cores if nc is None else nc

        snitch.info(f"Current performance config: {nc} cores and {ml} GB per core")
        snitch.info(f"Expecting to use {nc*ml} GB of RAM")
        # bigger than mem_size reduce the number of cores
        if nc*ml > mem_size:
            nc = int(mem_size//ml)
            snitch.warn(f"Reducing cores from {self.ncores} -> {nc}")
            snitch.info(f"Expecting to use {nc * ml} GB of RAM")
        return nc, f"{ml}GB"


@dataclass
class Axargs:
    """
    Dataclass containing axis arguments. ie. x and y axis, respective
    column names, and data corresponding to those columns

    - *axis: Name of the axis chosen
    - *data_col: actual data column name for axis in the MS
    - *data: data contained in the column name for the axis

    """
    xaxis: str
    yaxis: str
    data_column: str
    msdata: Any
    iaxis: str = None
    caxis: str = None
    xdata_col: str = field(init=False, default=None)
    ydata_col: str = field(init=False, default=None)
    cdata_col: str = field(init=False, default=None)
    idata_col: str = field(init=False, default=None)
    xdata: da.array = None
    ydata: da.array = None
    cdata: da.array = None
    # idata: da.array = None # This is not necessary here, iterating over ms
    flags: da.array = None
    errors: da.array = None
  

    def __post_init__(self):
        #Get the proper name for the data column first before getting other names
        self.yaxis = self.get_proper_axis_names(self.yaxis)
        self.xaxis = self.get_proper_axis_names(self.xaxis)
        self.data_column = self.get_colname(self.data_column, self.data_column)
        self.xdata_col = self.get_colname(self.xaxis, self.data_column)
        self.ydata_col = self.get_colname(self.yaxis, self.data_column)

        if self.iaxis is not None:
            self.iaxis = self.get_proper_axis_names(self.iaxis)
            if self.iaxis == "baseline":
                self.idata_col = "ANTENNA1 ANTENNA2"
            elif self.iaxis in ["corr", "antenna"]:
                self.idata_col = None
            else:
                self.idata_col = self.get_colname(self.iaxis, self.data_column)
        if self.caxis is not None:
            self.caxis = self.get_proper_axis_names(self.caxis)
            if self.caxis == "baseline":
                self.cdata_col = "ANTENNA1 ANTENNA2"
            elif self.caxis == "antenna":
                self.cdata_col = "ANTENNA"
            elif self.caxis == "corr":
                self.cdata_col = None
            else:
                self.cdata_col = self.get_colname(self.caxis, self.data_column)
        
    def get_proper_axis_names(self, name):
        """
        Translate inpu axis name to name known and accepted by ragavi
        name: obj:`str`
            Input axis name
        
        Returns
        -------
        name: :obj:`str`
            Proper name of the given alias. Otherwise, returns the input
        """
        names, name = {}, name.lower()
        names["a"] = names["amp"] = names["ampl"] = \
            names["amplitude"] = "amplitude"
        names["ant"] = names["antenna"] = "antenna"
        names["ant1"] = names["antenna1"] = "antenna1"
        names["ant2"] = names["antenna2"] = "antenna2"
        names["bl"] = names["baseline"] = "baseline"
        names["chan"] = names["freq"] = names["frequency"] = \
            names["channel"] = "chan"
        names["correlation"] = names["corr"] =  "corr"
        names["ddid"] = names["spw"] = "spw"
        names["field"] = "field"
        names["i"] = names["im"] = names["imag"] = \
            names["imaginary"] = "imaginary"
        names["r"] = names["re"]=names["real"] = "real"    
        names["p"] = names["ph"] = names["phase"] = "phase"
        names["scan"] = "scan"
        names["time"] = "time"
        names["uvdist"] = names["uvdist_l"] = names["uvdistl"] = \
            names["uvdistance"] = "uvdistance"
        names["uvwave"] = names["uvwavelength"] = "uvwavelength"
        
        if name in names:
            name = names.get(name)
        elif len(get_close_matches(name, names, 1))>0:
            name = names[get_close_matches(name, names, 1)[0]]
            snitch.warn(f"'{name}' is not a valid name in ragavi")
            snitch.debug(f"Switching to '{names[name]}' as a close match")
        return name


    def translate_y(ax):
        axes = {}
        axes["a"] = axes["amp"] = "amplitude"
        axes["i"] = axes["imag"] = "imaginary"
        axes["p"] = "phase"
        axes["r"] = "real"
        return axes.get(ax) or ax
    
    def update_data(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: :obj:`dict` containing the name of the data to add and its 
        value
        """
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def colname_map(self, data_column):
        """Map some custom axis names to actual column names
        
        Note that: "chan" and "corr" axes are defaulted to `data_column`
        """
        axes = dict()
        axes["time"] = "TIME"
        axes["spw"] = "DATA_DESC_ID"
        axes["corr"] = data_column
        axes.update({key: data_column for key in ("a", "amp", "amplitude",
                "delay", "i", "imag", "imaginary", "p", "phase", "r", "real")})
        axes.update({key: "UVW" for key in ("uvdist", "uvdistance",
                        "uvdistl", "uvdist_l", "uvwave", "uvwavelength")})
        axes.update({key: "ANTENNA1" for key in ("ant1", "antenna1")})
        axes.update({key: "ANTENNA2" for key in ("ant2", "antenna2")})
        axes.update({key: ("ANTENNA1", "ANTENNA2") for key in ("bl", "baseline")})
        axes.update({key: data_column for key in ("chan", "channel", "freq",
                        "frequency")})
        return axes

    def get_colname(self, axis, data_column):
        """
        Get the appropriate column name for axis from MS
        Parameters
        ----------
        axis: : obj:`str`

        data_column: :obj:`str`
            The main DATA column to be used for this run
        """        
        col_maps = self.colname_map(data_column)

        if axis.upper() in self.msdata.colnames:
            colname = axis.upper()
        elif re.search(r"\w*{}\w*".format(axis), 
                       ", ".join(self.msdata.colnames),
                       re.IGNORECASE) and len(axis) > 1:
            colname = re.search(r"\w*{}\w*".format(axis),
                                ", ".join(self.msdata.colnames),
                                re.IGNORECASE).group()
        elif axis in col_maps:
            colname = col_maps[axis]
        elif len(get_close_matches(axis.upper(), self.msdata.colnames,
                               n=1)) > 0:
            colname = get_close_matches(axis.upper(), self.msdata.colnames,
                                        n=1)[0]
            snitch.debug(f"'{axis}' column not found, using closest '{colname}'")
        else:
            colname = None
            snitch.debug(f"column for: {axis} not found. Setting Column to {colname}")
        return colname
    
    def set_axis_data(self, axis, ms_obj):
        """
        Automatically Set axis data for a certain column

        Parameters
        ----------
        axis: str
            Name of selected axis: {yaxis, xaxis, iaxis, caxis}
        ms_obj:
            MS xarray object
        """
        axis_name = getattr(self, f"{axis[0]}axis")
        _axis = f"{axis[0]}data"
        column = getattr(self, f"{_axis}_col")
        if len(re.findall(rf"{axis_name}\w*", "channel frequency")) > 0:
            snitch.debug(f"{axis[0]}axis column is channel/frequency")
            # setattr(self, _axis, ms_obj[column].chan)
            setattr(self, _axis, self.msdata.active_channels/1e9)
        else:
            snitch.debug(f"{axis[0]}axis data column is {column}")
            if column is not None:
                if column == "ANTENNA1 ANTENNA2":
                    ms_obj = ms_obj.assign(BASELINE=pair(ms_obj.ANTENNA1,
                                                        ms_obj.ANTENNA2))
                    setattr(self, _axis, ms_obj["BASELINE"])
                else:
                    if column in ms_obj:
                        setattr(self, _axis, ms_obj.get(column))
                    else:
                        setattr(self, _axis, ms_obj.attrs[column])
        return

    @property
    def active_columns(self):
        actives = {self.xdata_col, self.ydata_col, self.data_column}
        if self.cdata_col and self.cdata_col!="ANTENNA":
            actives.update({*self.cdata_col.split()})
        if self.idata_col:
            actives.update({*self.idata_col.split()})
        return list(actives)


@dataclass
class Selargs:
    antennas: str = None
    baselines: str = None
    corrs: str = None
    channels: str = None
    ddids: str = None
    fields: str = None
    taql: str = None
    t0: str = None
    t1: str = None
    scans: str = None

@dataclass
class Plotargs:
    cmap: str = None
    html_name: str = None
    c_height: int = None
    c_width: int = None
    grid_cols: int = None
    link_plots: bool = None
    # user mins and maxes
    x_min: float = None
    x_max: float = None
    y_min: float = None
    y_max: float = None
    # Grouped mins and maxes
    xmin: float = None
    xmax: float = None
    ymin: float = None
    ymax: float = None

    #Default maximum screen size on my computer
    plot_width: int = 1920
    plot_height: int = 1080
    partitions: int =  None #Number or partions in the dataset
    grid_rows: int = None # number of rows in the grid. To be calculated
    title: str = None
    n_categories: int = None
    # categories: str = None
    cat_map: dict = None
    i_title: str = ""
    i_ttips: dict = None

    def __post_init__(self):
        self.i_ttips = {}
        if self.grid_cols is None:
            # No grid is needed therefore
            self.c_height = 720 if not self.c_height else self.c_height
            self.c_width = 1080 if not self.c_width else self.c_width
        else:
            # meaning the iteration axis is active
            self.c_height = 200 if not self.c_height else self.c_height
            self.c_width = 200 if not self.c_width else self.c_width
        
        self.xmin, self.xmax, self.ymin, self.ymax = (self.x_min, self.x_max,
            self.y_min, self.y_max)
        snitch.debug(f"Canvas (w x h): {self.c_width} x {self.c_height}")


    def set_grid_cols_and_rows(self):
        """Set the number of rows and columns to be in subploting grid
        This function is called in the main plotting script
        """
        # set columns depending on the numer of partitions
        if self.partitions is None:
            snitch.info("No partition size. Please set 'partitions' attribute")
            return

        if self.grid_cols:
            if self.partitions < self.grid_cols: 
                self.grid_cols = self.partitions
            self.grid_rows = int(np.ceil(self.partitions // self.grid_cols))
            self.plot_width = int((self.plot_width * 0.961) // self.grid_cols)
            # set plot height same as width for iter mode. I want a square box
            # Default number of grid columns is 5
            if self.grid_rows > 1:
                self.plot_height = int(self.plot_width *0.9)
            else:
                self.plot_height = int(self.plot_height * 0.72)
            snitch.debug(f"Plot grid (r x c): {self.grid_rows} x {self.grid_cols}")
        else:
            self.plot_width = int(self.plot_width * 0.961)
            self.plot_height = int(self.plot_height * 0.8)
            #set grid cols to 1 in this case for colour bar purpose
            self.grid_cols = 1

            

    def form_plot_title(self, axargs):
        self.title = f"{axargs.yaxis} vs {axargs.xaxis}"
        self.title += f" coloured by {axargs.caxis}" if axargs.caxis else ""
        self.title += f" iterated by {axargs.iaxis}" if axargs.iaxis else ""
        self.title = self.title.title()

    def get_category_maps(self, cat, msd):
        """
        Get attributes to be gotten from msdata for a specific axis
        cat: str
            Category name
        msd: MsData objectt
            Object containing some data for the ms
        """
        cat_maps = {
            "rant": "reverse_ant_map",
            "ant": "ant_map", "rantenna": "reverse_ant_map",
            "antenna": "ant_map", "rantenna": "reverse_ant_map",
            "antenna1": "ant_map", "rantenna1": "reverse_ant_map",
            "antenna2": "ant_map", "rantenna2": "reverse_ant_map",
            "baseline": "bl_map", "rbaseline": "reverse_bl_map",
            "corr": "corr_map", "rcorr": "reverse_corr_map",
            "field": "field_map", "rfield": "reverse_field_map",
            "scan": "scan_map", "rscan": "reverse_scan_map",
            "chan": "", "ddid": "", "spw": "", "rspw": ""
        }
        return getattr(msd, cat_maps[cat], None)
    
    def get_category_sizes(self, cat, msd):
        cat_sizes = {
            "antenna": getattr(msd, "num_ants"),
            "antenna1": getattr(msd, "num_ants")-1,
            "antenna2": getattr(msd, "num_ants")-1,
            "baseline": getattr(msd, "num_baselines"),
            "corr": getattr(msd, "num_corrs"),
            "field": getattr(msd, "num_fields"),
            "scan": getattr(msd, "num_scans"),
            "chan": getattr(msd, "num_chans"),
            "ddid": getattr(msd, "num_spws"),
            "spw": getattr(msd, "num_spws"),
        }
        return cat_sizes[cat]

    def set_category_ids_and_sizes(self, axis, msd):
        """
        Set the number of categories and their maps for plotting purposes.
        One of the valid category axes: antenna, baseline, corr, field, chan, 
        ddid, scan

        Parameters
        ----------
        axis: str
            Name of the category axis
        msd: msdata
            Data container for MS 
        """
        self.cat_map = self.get_category_maps(f"r{axis}", msd)
        self.n_categories = self.get_category_sizes(axis, msd)

    def set_iter_title(self, axes, sub, msd):
        title, i_ttips = "", {}
        if axes.iaxis == "baseline":
            cat_map = self.get_category_maps(f"rantenna", msd)
        else:
            cat_map = self.get_category_maps(f"r{axes.iaxis}", msd)
        for name, nid in sub.attrs.items():
            if name == "__daskms_partition_schema__":
                continue
            elif name == "DATA_DESC_ID":
                title += f"{name} {nid} "
                i_ttips[name] = nid
            elif "ANTENNA" in name and axes.iaxis == "baseline":
                title += f"{cat_map[nid]}-"
                if "baseline" in i_ttips:
                    i_ttips["baseline"] += f"-{cat_map[nid]}"
                else:
                    i_ttips["baseline"] = f"{cat_map[nid]}"
            else:
                title += f"{name} {cat_map[nid]} "
                i_ttips[name] = cat_map[nid]
        self.i_title = title.upper()
        self.i_ttips = {k.title() : [v] for k, v in i_ttips.items()}
