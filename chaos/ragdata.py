import numpy as np
import xarray as xr
import dask.array as da
import daskms as xm

from casacore.tables import table

#for testing purposes
import datetime
from ipdb import set_trace

class MsData:
    """
    Get some general data that will be required in the Plotter.
    This does not include user selected data and actual data columns
    More like the passive data only.
    """
    def __init__(self, ms_name):
        self.ms_name = ms_name
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
            self._ms.close()
    
    def _process_observation_table(self):
        with table(self._ms.getkeyword("OBSERVATION"), ack=False) as sub:
            self._start_time, self._end_time = sub.getcell("TIME_RANGE",0)
            self._telescope = sub.getcell("TELESCOPE_NAME", 0)

    def _process_antenna_table(self):
        with table(self._ms.getkeyword("ANTENNA"), ack=False) as sub:
            self._ant_names = sub.getcol("NAME")
            self._ant_ids = sub.rownumbers()
            self._ant_map = {name: ids for ids, name in zip(
                self._ant_ids, self._ant_names)}

    def _process_field_table(self):
        with table(self._ms.getkeyword("FIELD"), ack=False) as sub:
            self._field_names = sub.getcol("NAME")
            self._field_ids = sub.getcol("SOURCE_ID")
            self._field_map = {name: ids for ids, name in zip(
                self._field_ids, self._field_names)}
    
    def _process_frequency_table(self):
        with table(self._ms.getkeyword("SPECTRAL_WINDOW"), ack=False) as sub:
            self._freqs = sub.getcol("CHAN_FREQ")
            self._spws = sub.rownumbers()
            self._num_spws = len(self._spws)
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
        self._corr_map = {name: ids for ids, name in enumerate(self._corr_types)}

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
    def field_names(self):
        return self._field_names
    
    @property
    def field_ids(self):
        return self._field_ids

    @property
    def field_map(self):
        return self._field_map

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
    def scans(self):
        return self._scans


class RagProcessor:
    def __init__(self, data):
        self.data = data

    def amplitude(self):
        return da.absolute(self.data)

    def phase(self, unwrap=True):
        phase = xr.apply_ufunc(da.angle, self.data,
                               dask="allowed", kwargs=dict(deg=True))
        if unwrap:
            return phase.reduce(np.unwrap)
        else:
            return phase

    def real(self):
        return self.data.real

    def imaginary(self):
        return self.data.imag

    @staticmethod
    def uv_distance(uvw):
        return da.sqrt(da.square(uvw.isel({'uvw': 0})) +
                       da.square(uvw.isel({'uvw': 1})))

    @staticmethod
    def uv_wavelength(uvw, freqs):
        return RagProcessor.uv_distance(uvw).expand_dims({"chan": 1}, axis=1) / (3e8/freqs)

    @staticmethod
    def unix_timestamp(in_time):
        """
        The difference between MJD and unix time i.e. munix = MJD - unix_time
        so unix_time = MJD - munix => munix = 3506716800.0 = (40857 * 86400)
        The value 40587 is the number of days between the MJD epoch (1858-11-17)
        and the Unix epoch (1970-01-01), and 86400 is the number of seconds in a
        day
        """
        munix = 3506716800.0
        return in_time - munix


class ActiveData:
    def __init__(self):
        pass

class DataSelector:
    """
    Select subsets of data and form data selection strings for
    case a:     4 (only this)
    case b:     5, 6, 7 (all of these)
    case c:     5~7 (inclusive range)
    case d:     5:8 (exclusive range)
    case e:     5: (to_last)
    case f:     ::10 (every 10th channel)

    TAQL
    """
    @staticmethod
    def nametoid(selection, dmap):
        notnumber = lambda x: not x.isdecimal()
        if notnumber(selection):
            try:
                return str(dmap[selection])
            except KeyError:
                return None
        return selection
    
    @staticmethod
    def form_taql_string(msdata, antenna=None, baseline=None, field=None, spw=None,
                         scan=None, taql=None, time=None, uv_range=None):
        """
        Can be searched by name and id:
        ALL WILL BE A comma separated string
         - antenna
         - baseline: this will be a list of comma separated antennas which will be divided by a    
            dash. e.g. m001-m003, m001, m10-m056
         - field

         time will be in seconds and a string: start, end
        """
        super_taql = []

        if antenna and baseline is None:
            antenna = antenna.replace(" ", "").split(",")
            for i, selection in enumerate(antenna):
                antenna[i] = DataSelector.nametoid(selection, msdata.ant_map)
            
            antenna = f"ANTENNA1 IN [{','.join(set([_ for _ in antenna if _]))}]"
            super_taql.append(antenna)

        if baseline:
            baseline = baseline.replace(" ", "").split(",")
            ants = {"ANTENNA1": [], "ANTENNA2": []}
            for bl in baseline:
                bl = bl.split("-")
                if not all([DataSelector.nametoid(_, msdata.ant_map) for _ in bl]):
                    continue
                for i, selection in enumerate(bl, 1):
                    ants[f"ANTENNA{i}"].append(DataSelector.nametoid(selection, msdata.ant_map))
            baseline = " && ".join([f"{key}==[{','.join(value)}]" for key, value in ants.items()])
            super_taql.append(f"any({baseline})")

        if field:
            field = field.replace(" ", "").split(",")
            for i, selection in enumerate(field):
                field[i] = DataSelector.nametoid(selection, msdata.field_map)
            field = f"FIELD_ID IN [{','.join(set([_ for _ in field if _]))}]"
            super_taql.append(field)
        if scan:
            scan = scan.replace(" ", "")
            scan = f"SCAN_NUMBER IN [{scan}]"
            super_taql.append(scan)
        if spw:
            spw = spw.replace(" ", "")
            spw = f"DATA_DESC_ID IN [{spw}]"
            super_taql.append(spw)
        if taql:
            super_taql.append(taql)
        if time:
            time = [int (_) for _ in time.replace(" ", "").split(",")]
            if len(time) < 2:
                time.append(None)
            time = f"TIME IN [{msdata.start_time + time[0]} =:= {msdata.start_time + time[-1] or msdata.end_time}]"
            super_taql.append(time)
        if uv_range:
            # TODO sELECT an spw here in the CHANG FREQ
            uv_range, unit = uv_range.replace(" ", "").split("_")
            if unit == "lambda":
                uv_range = f"""any(sqrt(sumsqr(UVW[:2])) / c() *  
                                [select CHAN_FREQ[0] from ::SPECTRAL_WINDOW][DATA_DESC_ID,] < {uv_range}) 
                            """
            else:
                uv_range = f"any(sqrt(sumsqr(UVW[:2])) < {uv_range})"
            super_taql.append(uv_range)

        set_trace()
        return " && ".join(super_taql)

    
    @staticmethod
    def get_knife(data):
        """
        a.k.a knife
        Format the data to addvalues where they need t be added
        for selecting channels and corrs
        """
        if data.isdecimal():
            return np.array([int(data)])
        if "," in data:
            return np.array([int(_) for _ in data])
        if "~" in data:
            data = data.replace("~", ",1+")
        if "::" in data:
            if data.startswith("::"):
                data = data.replace(":", "None,")
            elif data.endswith("::"):
                data = data.replace(":", ":None:None")
            else:
                raise SyntaxError(f"Invalid String {data}")
        data = eval(data.replace(":", ","))
        return slice(*data)



#SLICE SELECTION FOR
#channels, corrs, spws
cases = {
    "a" :  "4",
    "b" :  "5, 6, 7",
    "c" :  "5~7",
    "g":   "5:8:2",
    "d" :  "5:8",
    "e" :  "5:",
    "f" : "::10"
}

antenna = "m003, m012, m056, m010, 5, 9, 0"
baseline = "m003-m012, 0-10, m048-m045, m029-m077, 5-9"
field = "0, DEEP_2, 0252-712, 2, 0408-65"
spw = "0"
scan = "4, 10, 12, 67"
taql = None
time = "60, 3500"
uv_range = "8430_m"
# uv_range = "14000_l"
    



if __name__ == "__main__":
    ms_name = "/home/lexya/Documents/test_stimela_dir/msdir/1491291289.1ghz.1.1ghz.4hrs.ms"
    corr = "0"
    chan = "::5"
    msdata = MsData(ms_name)
    msdata.initialise_data()
    msdata.sel_freq = msdata.freqs[DataSelector.get_knife(chan)]

    tstring = DataSelector.form_taql_string(msdata, antenna="m003, m012, m056, m010, 5, 9, 0",
                                            baseline="m003-m012, 0-10, m048-m045, m029-m077, 5-9",
                                            field="0, DEEP_2, 0252-712, 2, 0408-65",
                                            spw="0",
                                            scan="4, 10, 12, 67",
                                            taql=None,
                                            time="60, 3500",
                                            uv_range="8430_m")
    set_trace()                                            
    for ms in xm.xds_from_ms(ms_name, taql_where=tstring):
        ms = ms.sel(chan=DataSelector.get_knife(chan), corr=DataSelector.get_knife(corr))
        msdata.data = ms.DATA
        msdata.data_column = ms.DATA.name

        process = RagProcessor(msdata.data)
        set_trace()
