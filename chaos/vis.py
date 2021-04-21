from .ragdata import dataclass, field, MsData, Axargs, Genargs, Selargs, Plotargs

@dataclass
class Axes(Axargs):
    i_axis: str
    c_axis: str
    cdata_col: str = field(init=False)
    idata_col: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.idata_col = super().get_colname(self.i_axis, self.data_column)
        self.cdata_col = super().get_colname(self.c_axis, self.data_column)


@dataclass
class Pargs(Plotargs):
    c_height: int
    c_width: int
    grid_cols: int
    link_plots: bool
    x_min: float
    x_max: float
    y_min: float
    y_max: float

#do this first immediately after getting ms name
msdata = MsData().initialise_data()
#set user input arguments
general = Genargs()
axes = Axes()
selection = Selargs()
plargs = Pargs()

