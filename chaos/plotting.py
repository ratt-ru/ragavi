import os
import math

from ipdb import set_trace

from bokeh.models import (BasicTicker, DatetimeAxis, DataRange1d, Grid, Legend,
                          LinearAxis, LinearScale, LogAxis, LogScale,
                          Toolbar, Plot, Range1d,
                          Title, ColumnDataSource, Whisker, Scatter, Line, Circle)
from bokeh.models.tools import (BoxSelectTool, BoxZoomTool,
                                HoverTool, LassoSelectTool, PanTool,
                                ResetTool, SaveTool, UndoTool,
                                WheelZoomTool)

from bokeh.io import save

from overrides import rdict


"""
REMEMBER
========
fig.select_one("selection string, e.g items name")
fig.select(name="selection string")
"""

class BaseFigure:
    f_num = -1
    def __init__(self, width, height, x_scale, y_scale, add_grid, add_toolbar, add_xaxis,
                add_yaxis, plot_args, axis_args, tick_args):

        self.__update_fnum__()
        self.f_num = self.get_fnum()
      
        self.width = width
        self.height = height
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.add_grid = add_grid
        self.add_toolbar = add_toolbar
        self.add_xaxis = add_xaxis
        self.add_yaxis = add_yaxis
        # set to empty dict if not provided
        self.plot_args = plot_args or {}
        self.axis_args = axis_args or {}
        self.tick_args = tick_args or {}

    def __update_fnum__(self):
        BaseFigure.f_num += 1
    
    def get_fnum(self):
        return BaseFigure.f_num

    def create_figure(self):
        self.plot_args = rdict(self.plot_args)

        # Set these defaults only if they haven't been set by user explicitly
        self.plot_args.set_multiple_defaults(
            background="white", border_fill_alpha=0.1, border_fill_color="white",
            min_border=3, outline_line_dash="solid",
            outline_line_width=2, outline_line_color="#017afe", outline_line_alpha=0.4, 
            output_backend="canvas", sizing_mode="stretch_width", title_location="above",
            toolbar_location="above", plot_width=self.width,
            plot_height=self.height, frame_height=int(0.93 * self.height),
            frame_width=int(0.98*self.width), name=f"fig{self.f_num}_plot")

        fig = Plot()

        for axis, scale in [("x", self.x_scale), ("y", self.y_scale)]:
            self.plot_args[f"{axis}_range"] = self.make_range(dim=axis)
            self.plot_args[f"{axis}_scale"] = self.make_scale(dim=axis, scale=scale)

        if self.add_toolbar:
            self.plot_args["toolbar"] = self.make_toolbar()
        
        fig.update(**self.plot_args)
        
        if self.add_grid:
            for axis in ["x", "y"]:
                fig.add_layout(self.make_grid(dim=axis))
        
        if self.add_yaxis:
            fig.add_layout(self.make_axis(dim="y", scale=self.y_scale), "below")
        if self.add_xaxis:
            fig.add_layout(self.make_axis(dim="x", scale=self.x_scale), "left")

        return fig
        

    def make_scale(self, dim, scale):
        scales = {
            "linear": LinearScale,
            "log": LogScale,
            "datetime": LinearScale
        }
        return scales[scale](name=f"fig{self.f_num}_{dim}_scale")


    def make_ticker(self, dim):
        return BasicTicker(name=f"fig{self.f_num}_{dim}_ticker", desired_num_ticks=5, 
        ** self.tick_args)


    def make_axis(self, dim, scale):
        axes = {
            "linear": LinearAxis,
            "log": LogAxis,
            "datetime": DatetimeAxis
        }

        if self.axis_args is None:
            self.axis_args = rdict()
        else:
            self.axis_args = rdict(self.axis_args)
        
        self.axis_args.set_multiple_defaults(minor_tick_line_alpha=0, axis_label_text_align="center",
                                             axis_label_text_font="monospace",
                                             axis_label_text_font_size="10px",
                                             axis_label_text_font_style="normal",
                                             major_label_orientation="horizontal",
                                             name=f"fig{self.f_num}_{dim}_axis")
        return axes[scale]( ticker=self.make_ticker(dim=dim), **self.axis_args)


    def make_grid(self, dim):
        dims = {"x": 0, "y": 1}
        return Grid(dimension=dims[dim], ticker=self.make_ticker(dim))


    def make_range(self, dim, r_min=None, r_max=None, visible=True):
        if r_min is None and r_max is None:
            return DataRange1d(name=f"fig{self.f_num}_{dim}_range", only_visible=visible)
        else:
            return Range1d(name=f"fig{self.f_num}_{dim}_range", start=r_min, end=r_max)


    def make_toolbar(self):
        return Toolbar(name=f"fig{self.f_num}_toolbar",
                        tools=[
                            HoverTool(tooltips=[("x", "$x"), ("y", "$y")],
                                      name=f"fig{self.f_num}_htool", point_policy="snap_to_data"),
                            BoxSelectTool(), BoxZoomTool(),
                            # EditTool(), # BoxEditTool(), # RangeTool(),
                            LassoSelectTool(), PanTool(), ResetTool(),
                            SaveTool(), UndoTool(), WheelZoomTool()
                            ]
                    )
    

class FigRag(BaseFigure):
    def __init__(self, width=1080,  height=720, x_scale="linear",
                y_scale="linear", add_grid=True, add_toolbar=False,
                add_xaxis=True, add_yaxis=True, plot_args=None, axis_args=None,
                tick_args=None):
        super().__init__( width, height, x_scale, y_scale, add_grid, add_toolbar, add_xaxis, 
                         add_yaxis, plot_args, axis_args, tick_args)

        self._fig = super().create_figure()
        self.rend_idx = 0
        self.legend_items = []
    
    
    def update_xlabel(self, label):
        self._fig.xaxis.axis_label = label


    def update_ylabel(self, label):
        self._fig.yaxis.axis_label = label


    def update_title(self, title, location="above", **kwargs):
        
        kwargs = rdict(kwargs)

        kwargs.set_multiple_defaults(align="center", name=f"fig{self.f_num}_title", text=title,
                      text_font_size="24px", text_font="monospace", text_font_style="bold")

        self._fig.add_layout(Title(**kwargs), location)
    
    def add_axis(self, label, dim, scale, location="right"):
        # TODO: fIX CHANGGE FROM EXTRA X_RANGE TO NON  SPECIFIC DIM TO MAKE IT MORE GENERAL
        # CALL UPPON SUPER MAKE_RANGE METHOD TO ACHIVER
        if self._fig.extra_x_ranges is None:
            self._fig.extra_x_ranges = {}

        self._fig.extra_x_ranges["fig{self.f_num}_extra_{dim}range"] = super().make_range
        new_axis = super().make_axis(dim, scale, name=f"fig{self.f_num}_extra_{dim}axis")
        self._fig.add_layout(new_axis, location)

    def format_axes(self, **kwargs):
        for axis in self._fig.axes:
            axis.update(**kwargs)

    def create_data_source(self, data, **kwargs):
        return ColumnDataSource(data=data, **kwargs)

    def hide_glyphs(self, exclude=0):
        if exclude<0:
            #subtract exclude to get index of the desired last number in reverse
            exclude = len(self._fig.renderers) + exclude
        for idx, renderer in enumerate(self._fig.renderers):
            if idx!=exclude:
                renderer.visible = False

    def show_glyphs(self):
        for renderer in self._fig.renderers:
            renderer.visible = True

    def add_glyphs(self, glyph, data, errors=None, legend=None, **kwargs):
        kwargs = rdict(kwargs)
        
        data_src = self.create_data_source(data,
                        name=f"fig{self.f_num}_gl{self.rend_idx}_ds")


        rend = self._fig.add_glyph(data_src, glyph(name=f"fig{self.f_num}_gl{self.rend_idx}",
                         **kwargs))
        rend.name = f"fig{self.f_num}_ren{self.rend_idx}"
        if self.rend_idx > 0:
            rend.visible = False

        if errors is not None:
            self.add_errors(data_src, errors)

        if legend is not None:
            self.legend_items.append((legend, [rend]))

        self.rend_idx += 1

    def add_errors(self, data, errors, base="x", dim="y", **kwargs):
        """
        data: :obj:`dict` or :obj:`ColumnDataSource`
            All data in a dictionary format or a column data format
        errors: :obj:`np.array` or :obj:`da.array`
            Errors in an numpy or dask array
        base: :obj:`str`
            The name of the data column for the x-axis
        dim: :obj:`str`
            Name of the axis on which to add the errors
        """

        if  type(data) != ColumnDataSource:
            data = self.create_data_source(data)
        
        data.add(data.data[dim] + errors, name="upper")
        data.add(data.data[dim] - errors, name="lower")

        ebar = Whisker(source=data, base=base, lower="lower", upper="upper",
                name=f"fig{self.f_num}_er{self.rend_idx}", visible=False, upper_head=None,
                       lower_head=None, line_color="red", line_cap="round", ** kwargs)

        if self.rend_idx < 1:
            ebar.visible = True

        #link the visible properties of this error bars and its corresponding glyph
        self._fig.select_one(f"fig{self.f_num}_ren{self.rend_idx}").js_link("visible", ebar, "visible")
     
        self._fig.add_layout(ebar)

    def write_out(self, filename="oer.html"):
        if ".html" not in filename:
            filename += ".html"
        save(self._fig, filename=filename, title=os.path.splitext(filename)[0])

    
    def add_legends(self, group_size=16, **kwargs):
        """
        Group legend items into 16 each,
        create legend objects and,
        attach them to figure
        """
        kwargs = rdict(kwargs)
        kwargs.set_multiple_defaults(click_policy="hide", glyph_height=20,
             glyph_width=20, label_text_font_size="8pt",label_text_font="monospace",
             location="top_left", margin=1, orientation="horizontal", 
             level="annotation", spacing=1, padding=2, visible=False)

        legends = []
        n_groups = math.ceil(len(self.legend_items) / group_size)
        for idx in range(n_groups):
            # push the legends into the stack
            self._fig.above.insert(0,
                Legend(items=self.legend_items[idx*group_size: group_size*(idx+1)],
                        name=f"fig{self.f_num}_leg{idx}", **kwargs))
            legends.append(Legend(items=self.legend_items[idx*group_size: group_size*(idx+1)],
                                name=f"fig{self.f_num}_leg{idx}", **kwargs))
       
    def link_figures(self, *others):
        """
        Link to or more items of this class
        others: tuple
        """
        for idx, renderer in enumerate(self._fig.renderers):
            for other_fig in others:
                # Link their renderer's visible properties
                renderer.js_link("visible", other_fig.fig.renderers[idx], "visible")
                other_fig.fig.renderers[idx].js_link("visible", renderer, "visible")

    @property
    def fig(self):
        return self._fig
