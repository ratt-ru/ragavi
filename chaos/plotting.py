import os
import math

import numpy as np
from bokeh.models import (BasicTicker, Circle, Line, Scatter,
                          CDSView, BooleanFilter, DatetimeAxis, DataRange1d, Grid, Legend,
                          LinearAxis, LinearScale, LogAxis, LogScale,
                          Toolbar, Plot, Range1d,
                          Title, ColumnDataSource, Whisker, Scatter, Line, Circle)
from bokeh.models.tools import (BoxSelectTool, BoxZoomTool,
                                HoverTool, LassoSelectTool, PanTool,
                                ResetTool, SaveTool, UndoTool,
                                WheelZoomTool)
from bokeh.models.renderers import GlyphRenderer

from bokeh.io import save

from overrides import set_multiple_defaults

# for testing
from ipdb import set_trace


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
        # Set these defaults only if they haven't been set by user explicitly
        self.plot_args = set_multiple_defaults(self.plot_args, dict(
                background="white", border_fill_alpha=0.1, 
                border_fill_color="white",
                min_border=3, outline_line_dash="solid",
                outline_line_width=2, outline_line_color="#017afe", outline_line_alpha=0.4, 
                output_backend="canvas", sizing_mode="stretch_width", title_location="above",
                toolbar_location="above", plot_width=self.width,
                plot_height=self.height, frame_height=int(0.93 * self.height),
                frame_width=int(0.98*self.width), name=f"fig{self.f_num}_plot"
                )
        )

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
            fig.add_layout(self.make_axis(dim="y", scale=self.y_scale), "left")
        if self.add_xaxis:
            fig.add_layout(self.make_axis(dim="x", scale=self.x_scale), "below")

        return fig
        

    def make_scale(self, dim, scale):
        scales = {
            "linear": LinearScale,
            "log": LogScale,
            "datetime": LinearScale,
            "time": LinearScale
        }
        return scales[scale](tags=[f"{dim}scale"])


    def make_ticker(self, dim):
        return BasicTicker(tags=[f"{dim}ticker"], desired_num_ticks=5, 
        ** self.tick_args)


    def make_axis(self, dim, scale):
        axes = {
            "linear": LinearAxis,
            "log": LogAxis,
            "datetime": DatetimeAxis,
            "time": DatetimeAxis
        }

        if self.axis_args is None:
            self.axis_args = dict()
        else:
            self.axis_args = dict(self.axis_args)
        
        self.axis_args = set_multiple_defaults(self.axis_args, dict(
            minor_tick_line_alpha=0, axis_label_text_align="center",
            axis_label_text_font="monospace", 
            axis_label_text_font_size="10px",
            axis_label_text_font_style="normal",
            major_label_orientation="horizontal"))
        self.axis_args["tags"] = [f"{dim}axis"]
        return axes[scale]( ticker=self.make_ticker(dim=dim), **self.axis_args)


    def make_grid(self, dim):
        dims = {"x": 0, "y": 1}
        return Grid(dimension=dims[dim], ticker=self.make_ticker(dim))


    def make_range(self, dim, r_min=None, r_max=None, visible=True):
        if r_min is None and r_max is None:
            return DataRange1d(tags=[f"{dim}range"], only_visible=visible)
        else:
            return Range1d(tags=[f"{dim}range"], start=r_min, end=r_max)


    def make_toolbar(self):
        return Toolbar(name=f"fig{self.f_num}_toolbar",
                        tools=[
                            HoverTool(tooltips=[("x", "$x"), ("y", "$y")],
                                      name=f"fig{self.f_num}_htool", 
                                      point_policy="snap_to_data"),
                            BoxSelectTool(), BoxZoomTool(),
                            # EditTool(), # BoxEditTool(), # RangeTool(),
                            LassoSelectTool(), PanTool(), ResetTool(),
                            SaveTool(), UndoTool(), WheelZoomTool()
                    ])
    

class FigRag(BaseFigure):
    def __init__(self, width=1080,  height=720, x_scale="linear",
                y_scale="linear", add_grid=True, add_toolbar=False,
                add_xaxis=True, add_yaxis=True, plot_args=None,
                axis_args=None, tick_args=None):
        super().__init__( width, height, x_scale, y_scale, add_grid,
                         add_toolbar, add_xaxis, add_yaxis, plot_args,
                         axis_args, tick_args)

        self._fig = super().create_figure()
        self.rend_idx = 0
        self.legend_items = {}
    
    
    def update_xlabel(self, label):
        self._fig.xaxis.axis_label = label.capitalize()


    def update_ylabel(self, label):
        self._fig.yaxis.axis_label = label.capitalize()


    def update_title(self, title, location="above", **kwargs):
        kwargs = set_multiple_defaults(kwargs, dict(
            align="center", text=title.title(),
            text_font_size="24px", text_font="monospace",
            text_font_style="bold"))
        kwargs["tags"] = ["title"]

        self._fig.add_layout(Title(**kwargs), location)
    
    def add_axis(self, lo, hi, dim, scale, label, location="right"):
        """
        Add an extra x or y axis to the plot
        
        Parameters
        ----------
        lo: :obj:`float`
            Minimum value for this axis' range
        hi: :obj:`float`
            Maximum vlaue for this axis' range
        dim: :obj:`str`
            Dimension on which to add axis
        scale: :obj:`str`
            Scale of the new axis
        label: :obj:`str`
            Title for the new axis
        location: :obj:`str`
            Where to place the axis
        
        Returns
        -------
        Nothing
        """
        extra_ranges = getattr(self._fig, f"extra_{dim}_ranges")
        if extra_ranges is None:
            extra_ranges = {}
        
        #use nx to number the next range. Starts from 0
        nx = len(extra_ranges)

        extra_ranges.update({f"extra_{dim}range{nx}":
            self.make_range(dim, r_min=lo, r_max=hi, visible=True)})
        
        getattr(self._fig, f"extra_{dim}_ranges").update(extra_ranges)

        new_axis = self.make_axis(dim, scale)
        new_axis.update(**{f"{dim}_range_name": f"extra_{dim}range{nx}", 
                        "axis_label": label, "tags": [f"extra_{dim}axis"]})
        new_axis.ticker.desired_num_ticks = 10
        
        #update the extra ranges on the figure
        self._fig.add_layout(new_axis, location)

    def format_axes(self, **kwargs):
        for axis in self._fig.axes:
            axis.update(**kwargs)

    def create_data_source(self, data, **kwargs):
        return ColumnDataSource(data=data, **kwargs)
    
    def create_view(self, cds, view_data, **kwargs):
        return CDSView(filters=[BooleanFilter(view_data)], source=cds, **kwargs)

    def hide_glyphs(self, selection=None):
        """
        Make some glyphs invisible
        Parameters
        ----------
        selection: :obj:`str`
            comma separated strings of tags to hide
        """
        if selection is not None:
            selection = selection.replace(" ", "").split(",")
            for sel in selection:
                for rend in self._fig.select(tags=sel, type=GlyphRenderer):
                    rend.visible = False
        else:
            for rend in self._fig.renderers:
                rend.visible = False


    def show_glyphs(self, selection=None):
        if selection is not None:
            selection = selection.replace(" ", "").split(",")
            # hide all glyphs to begin with
            self.hide_glyphs()
            for sel in selection:
                for rend in self._fig.select(tags=sel, type=GlyphRenderer):
                    rend.visible = True
        else:
            for rend in self._fig.renderers:
                rend.visible = True

    def add_glyphs(self, glyph, data, legend=None, **kwargs):
        #allow passing actual data objects or AxInfo objects
        if type(data) != dict:
            pdata = dict(x=data.xdata, y=data.ydata)
        else:
            pdata = data

        data_src = self.create_data_source(pdata,
                        name=f"fig{self.f_num}_gl{self.rend_idx}_ds",
                                           tags=[self.rend_idx])
        if data.flags is not None:
            markers = np.full_like(data.flags, glyph, dtype="U17")
            markers[np.where(data.flags == False)] = "inverted_triangle"
            data_src.add(markers, name="markers")
            data_src.add(np.logical_or(data.flags, True), name="noflags")
            data_src.add(data.flags, name="flags")
            glyph = "markers"
            data_view = self.create_view(data_src, data.flags,
                tags=[self.rend_idx],
                name=f"fig{self.f_num}_gl{self.rend_idx}_view")
            
        tags = kwargs.pop("tags") or []
        rend = self._fig.add_glyph(data_src, Scatter(x="x", y="y",
            marker=glyph, tags=["glyph"], size=10, **kwargs))
        rend.update(name=f"fig{self.f_num}_ren_{self.rend_idx}", tags=tags,
            view=data_view
            )

        if data.errors is not None:
            self.add_errors(data_src, data.errors)

        if legend is not None:
            rend.tags.append(legend)
            self.legend_items[legend] = self._fig.select(tags=[legend], type=GlyphRenderer)

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
                visible=False, upper_head=None, lower_head=None,
                line_color="red",line_cap="round",
                tags=["ebar", f"{self.rend_idx}"], **kwargs)
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
        kwargs = set_multiple_defaults(kwargs, dict(
            click_policy="hide", glyph_height=20,
            glyph_width=20, label_text_font_size="8pt",
            label_text_font="monospace",
            location="top_left", margin=1, orientation="horizontal", 
            level="annotation", spacing=1, padding=2, visible=False))

        self.legend_items = list(self.legend_items.items())
        legends = []

        n_groups = math.ceil(len(self.legend_items) / group_size)
        for idx in range(n_groups):
            items = self.legend_items[idx*group_size: group_size*(idx+1)]
            for leg, rend_list in items:
                for rend in rend_list:
                    # add batch tag to the renderers
                    rend.tags.append(f"b{idx}")
            # push the legends into the stack
            legends.append(Legend(items=items, tags=["legend"], **kwargs))
        legends.reverse()
        for item in legends:
            self._fig.add_layout(item, "above")
       
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
