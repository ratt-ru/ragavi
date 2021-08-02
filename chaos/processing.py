import re
import dask.array as da
import numpy as np
import xarray as xr

from warnings import warn

class Processor:
    """Contains functions for:
    amplitude
    phase
    imaginary
    real
    uv_distance (metres)
    uv_distance (wavelengths)
    mjd -> unix time
    """
    def __init__(self, data):
        self.data = data

    def amplitude(self, in_array):
        if isinstance(in_array, np.ndarray):
            return np.abs(in_array)
        else:
            return da.absolute(in_array)

    def phase(self, in_array, unwrap=False):
        if isinstance(in_array, np.ndarray):
            data = np.angle(in_array)
            if unwrap:
                data = np.unwrap(data)
            return np.rad2deg(data)
        else:
            #unwrap radians before converting to degrees
            data = in_array.copy(deep=True)
            data.data = data.data.map_blocks(da.angle)
            if unwrap:
                data.data = data.data.map_blocks(np.unwrap)
            data.data = data.data.map_blocks(da.rad2deg)
            return data

    def real(self, in_array):
        return in_array.real

    def imaginary(self, in_array):
        return in_array.imag
    
    def calculate(self, axis=None, freqs=None):
        if axis is None:
            return self.data
        elif axis.startswith("a"):
            return self.amplitude(self.data)
        elif axis.startswith("i"):
            return self.imaginary(self.data)
        elif axis.startswith("r"):
            return self.real(self.data)
        elif axis.startswith("p"):
            return self.phase(self.data)
        elif axis == "time":
            return Processor.unix_timestamp(self.data)
        elif "wave" in axis.lower():
            return Processor.uv_wavelength(self.data, freqs)
        elif "dist" in axis.lower():
            return Processor.uv_distance(self.data)
        elif len(re.findall(f"{axis}\w*", "channel frequency")) > 0:
            return self.data
        else:
            return self.data

    @staticmethod
    def uv_distance(uvw):
        if isinstance(uvw, np.ndarray):
            return np.sqrt(np.square(uvw[:,:2]).sum(axis=1))
        else:
            return da.sqrt(da.square(uvw.sel(uvw=slice(0, 2))).sum(axis=1))

    @staticmethod
    def uv_wavelength(uvw, freqs):
        if isinstance(uvw, np.ndarray):
            return Processor.uv_distance(uvw)[:, np.newaxis] / (3e8/freqs)
            
        else:    
            return Processor.uv_distance(uvw) / (3e8/freqs)

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
        in_time = (in_time - munix).astype("datetime64[s]")
        return in_time


class Chooser:
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
        if not selection.isdecimal():
            try:
                return str(dmap[selection])
            except KeyError:
                return None
        return selection

    @staticmethod
    def form_taql_string(msdata, antennas=None, baselines=None, fields=None,
                        spws=None, scans=None, taql=None, time=None,
                        uv_range=None, return_ids=False):
        """
        Create TAQL used for preprocessing data selection in the MS

        Parameters
        ----------
        msdata: :obj:`MsData`
        antennas: :obj:`str`
            Comma seprated string containing antennas to be selection
        baselines: :obj:`str`
            Comma seprated string containing baselines to be selected.
            The baseline is of the form 'm000-m001'
        fields: :obj:`str`
            Comma seprated string containing fields to be selected
        spws: :obj:`str`
            Comma seprated string containing spws to be selected
        scans: :obj:`str`
            Comma seprated string containing scans to be selected
        taql: :obj:`str`
            TAQL string to be included in the main selected
        time: :obj:`tuple`
            Tuple containing time to start and time to end time
        uv_range: :obj:`str`
            Comma seprated string containing antenas to be selection
        return_ids: :ob:`bool`
            Return number ids for antennas, baselines, fields
        
        Returns
        -------
        super_taql: :obj:`str`
            String containing a complete TAQL selection string

        Can be searched by name and id:
        ALL WILL BE A comma separated string
         - antennas
         - baselines: this will be a list of comma separated antennas which will be divided by a    
            dash. e.g. m001-m003, m001, m10-m056
         - fields

         time will be in seconds and a string: start, end

        """
        super_taql = []
        super_dict = {}

        if antennas and baselines is None:
            antennas = antennas.replace(" ", "").split(",")
            for i, selection in enumerate(antennas):
                antennas[i] = Chooser.nametoid(selection, msdata.ant_map)
            
            super_dict["antennas"] = [int(_) for _ in antennas]
            antennas = f"ANTENNA1 IN [{','.join(set([_ for _ in antennas if _]))}]"
            super_taql.append(antennas)

        if baselines:
            baselines = baselines.replace(" ", "").split(",")
            ants = {"ANTENNA1": [], "ANTENNA2": []}
            for bl in baselines:
                bl = bl.split("-")
                if not all([Chooser.nametoid(_, msdata.ant_map) for _ in bl]):
                    warn(f"Baseline: {bl} is not available")
                    continue
                for i, selection in enumerate(bl, 1):
                    ants[f"ANTENNA{i}"].append(
                        Chooser.nametoid(selection, msdata.ant_map))
            
            super_dict["baselines"] = ants
            baselines = " && ".join(
                [f"{key}==[{','.join(value)}]" for key, value in ants.items()])
            super_taql.append(f"any({baselines})")

        if fields:
            fields = fields.replace(" ", "").split(",")
            for i, selection in enumerate(fields):
                fields[i] = Chooser.nametoid(selection, msdata.field_map)
            
            super_dict["fields"] = [int(_) for _ in fields]
            fields = f"FIELD_ID IN [{','.join(set([_ for _ in fields if _]))}]"
            super_taql.append(fields)
        if scans:
            scans = scans.replace(" ", "")
            super_dict["scans"] = [int(_) for _ in scans.split(",")]
            scans = f"SCAN_NUMBER IN [{scans}]"
            super_taql.append(scans)
        if spws:
            spws = spws.replace(" ", "")
            super_dict["spws"] = [int(_) for _ in spws.split(",")]
            spws = "DATA_DESC_ID" if "DATA_DESC_ID" in msdata.colnames \
                        else "SPECTRAL_WINDOW_ID" + f" IN [{spws}]"
            super_taql.append(spws)
        if taql:
            super_taql.append(taql)
        if time:
            #default for t0 and t1 is None, given as a tuple
            time = [float(_) if _ is not None else 0 for _ in time]
            time[0] = msdata.start_time + time[0]
            if time[1] > 0:
                time[1] = msdata.start_time + time[1]
            else:
                time[1] = msdata.end_time
            time = f"TIME IN [{time[0]} =:= {time[1]}]"
            super_taql.append(time)
        if uv_range:
            # TODO sELECT an spws here in the CHANG FREQ
            uv_range, unit = uv_range.replace(" ", "").split("_")
            if unit == "lambda":
                uv_range = f"any(sqrt(sumsqr(UVW[:2])) / c() *  " \
                    + f"[select CHAN_FREQ[0] from ::SPECTRAL_WINDOW]" \
                    + f"[DATA_DESC_ID,] < {uv_range}) "
            else:
                uv_range = f"any(sqrt(sumsqr(UVW[:2])) < {uv_range})"
            super_taql.append(uv_range)
        if return_ids:
            # Return the super selector string and a dict containing ids
            " && ".join(super_taql), super_dict
        return " && ".join(super_taql)

    @staticmethod
    def get_knife(data):
        """
        a.k.a knife
        Format the data to addvalues where they need t be added
        for selecting channels and corrs
        """
        #remove any spaces in data
        if data in [None, ""]:
            return slice(None)
        else:
            data = data.replace(" ", "")
        if data.isdecimal():
            return int(data)
        if "," in data:
            return np.array([int(_) for _ in data.split(",")])
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
