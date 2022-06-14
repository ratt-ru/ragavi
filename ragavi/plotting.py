import os
import math

import numpy as np
import matplotlib.pyplot as plt
from bokeh.io import save, output_file
from bokeh.models import (BasicTicker, BooleanFilter, LinearAxis, Plot, Scatter,
                          CDSView, Circle, ColumnDataSource, DataRange1d,
                          DatetimeAxis, Grid, Legend, Line, LinearScale,
                          LogAxis, LogScale, Range1d, Scatter, Title, Toolbar,
                          Whisker)
from bokeh.models.renderers import GlyphRenderer
from bokeh.models.tools import (BoxSelectTool, BoxZoomTool, HoverTool,
                                LassoSelectTool, PanTool, ResetTool, SaveTool,
                                UndoTool, WheelZoomTool)

from ragavi.lograg import logging
from ragavi.overrides import set_multiple_defaults

snitch = logging.getLogger(__name__)
plt.switch_backend('agg')

class BaseFigure:
    """
    A base for ragavi plots with some default specs

    Parameters
    ----------
    width: :obj:`float`
        Plot width
    height: :obj:`float`
        Plot height
    x_scale: :obj:`str`
        Scale for the x axis. Can be linear, datetime or log
    y_scale: :obj:`str`
        Scale for y axis. Same as x-axis
    add_grid: :obj:`bool`
        Whether or not to add an x and y grid to the plot
    add_toolbar: :obj:`bool`
        Whether or not to add a toolbar
    add_xaxis: :obj:`bool`
        Whether or not to an x-axis, including tick marks.
        Useful for disabling when the plots are  in a grid
        and no x-axis is required for each plot
    add_yaxis: :obj:`bool`
        Whether or not to add a y-axis. Reasons same as x-axis
    plot_args: :obj:`dict`
        A dictionary containing specs to pass to the bokeh plot.
        These are passed as they are and are not validated.
    axis_args: :obj:`dict`
        Dictionary containing specs to pass to the axes. Added as they are.
    tick_args: :obj:`dict`
        Dictionary containing specs for ticks
    """
    f_num = -1
    def __init__(self, width, height, x_scale, y_scale, add_grid,
                add_toolbar, add_xaxis, add_yaxis, plot_args, axis_args,
                tick_args):

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
        """Initalise a bokeh figure from scratch i.e."""
        # Set these defaults only if they haven't been set by user explicitly
        self.plot_args = set_multiple_defaults(self.plot_args, dict(
            background="white", border_fill_alpha=0.1, 
            border_fill_color="white", min_border=3, 
            outline_line_dash="solid", outline_line_width=2,
            outline_line_color="#017afe", outline_line_alpha=0.4, 
            output_backend="canvas", sizing_mode="stretch_width",
            title_location="above", toolbar_location="above",
            plot_width=self.width, plot_height=self.height,
            frame_height=int(0.93 * self.height),
            frame_width=int(0.98*self.width), name=f"fig{self.f_num}_plot"
            )
        )

        fig = Plot()

        for axis, scale in [("x", self.x_scale), ("y", self.y_scale)]:
            self.plot_args[f"{axis}_range"] = self.make_range(dim=axis)
            self.plot_args[f"{axis}_scale"] = self.make_scale(dim=axis,
                scale=scale)

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
        """
        Create a scale for the plot
        Parameters
        ----------
        dim: :obj:`str`
            Dimension for which the scale wil be. Can be x or y
        scale: obj:`str`
            Type of scale to make. Can be linear,
            log or datetime
        
        Returns
        -------
        A bokeh scale object
        """
        scales = {
            "linear": LinearScale,
            "log": LogScale,
            "datetime": LinearScale,
            "time": LinearScale
        }
        return scales[scale](tags=[f"{dim}scale"])


    def make_ticker(self, dim):
        """
        Create a ticker that can be attached to an axis
        
        Parameters
        ----------
        dim: :obj:`str`
            Dimension on which the ticker is to be added. Can be x or y
        
        Returns
        -------
            Bokeh Ticker object
        """
        return BasicTicker(tags=[f"{dim}ticker"], desired_num_ticks=5, 
        ** self.tick_args)


    def make_axis(self, dim, scale):
        """
        Create a axis for the plot
        Parameters
        ----------
        dim: :obj:`str`
            Dimension which the axis is for. Can be x or y
        scale: obj:`str`
            Type of scale to make. Can be linear,
            log or datetime
        
        Returns
        -------
        A bokeh axis object with its attached ticker
        """
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
        """Create a grid object for a dimension"""
        dims = {"x": 0, "y": 1}
        return Grid(dimension=dims[dim], ticker=self.make_ticker(dim))


    def make_range(self, dim, r_min=None, r_max=None, visible=True):
        """
        Create an axis' range
        
        Parameters
        ----------
        dim: :obj:`str`
            Dimension for which the range is attached
        r_min: :obj:`float`
            Minimum range value
        r_max: :obj:`float`
            Maximum range value
        visible: :obj`bool`
            Select whether the range will will be adjusted depending on the
            glyphs made visible on the plot, or all the available glyphs. This
            property is only available if r_min or r_max have not been
            specified. 
        
        Returns
        -------
        Bokeh range object
        """
        if r_min is None and r_max is None:
            return DataRange1d(tags=[f"{dim}range"], only_visible=visible)
        else:
            return Range1d(tags=[f"{dim}range"], start=r_min, end=r_max)


    def make_toolbar(self):
        """ Add tools to the toolbar """
        return Toolbar(name=f"fig{self.f_num}_toolbar",
                        tools=[
                            HoverTool(tooltips=[("x", "$x"), ("y", "$y")],
                                      name=f"fig{self.f_num}_htool",
                                      tags=["hover"],
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
        Make some/all glyphs invisible
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
        """Make some/all glyphs visible"""
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
        """
        Add glyphs to an already existing bokeh figure to create
        a renderer
        
        Parameters
        ----------
        glyph: :obj:`str`
            Glyph type. Could be circle, hex, diamond etc.
            see https://docs.bokeh.org/en/latest/docs/\
                reference/models/glyphs/scatter.html
            for a complete list
        data: :obj:`dict` or :obj:`AxArgs`
            A dictionary or ragavi axis args object containing data for
            the specified glyph. Note that there must be a key 'data' in
            this dictionary that contains an AxArgs object, or any object
            for which the following attributes can be obtained:
                flags: To capture the data's flags
                errors: To capture the data's error 
        legend: :obj:`str`
            Label to be associated with the glyph to be made
        **kwargs
            Other key word arguments that will be passed directly
            to the Glyph object.
        
        Returns
        -------
        Nothing
        """
        #allow passing actual data dict objects or AxInfo objects
        if type(data) != dict:
            pdata = dict(x=data.xdata, y=data.ydata)
        else:
            pdata = data
            data = pdata.pop("data", None)
        data_src = self.create_data_source(pdata,
                        name=f"fig{self.f_num}_gl{self.rend_idx}_ds",
                                           tags=[self.rend_idx])
        data_view = None
        if (data is not None and hasattr(data, "flags") and 
            data.flags is not None):
            markers = np.full_like(data.flags, glyph, dtype="U17")
            markers[np.where(data.flags == False)] = "inverted_triangle"
            data_src.add(markers, name="markers")
            data_src.add(np.logical_or(data.flags, True), name="noflags")
            data_src.add(data.flags, name="flags")
            glyph = "markers"
            data_view = self.create_view(data_src, data.flags,
                tags=[self.rend_idx],
                name=f"fig{self.f_num}_gl{self.rend_idx}_view")
            
        tags = kwargs.pop("tags", [])
        
        # Assume that an actual glyph object has been passed if glyph is not
        #  a string
        if type(glyph) == str:
            rend = self._fig.add_glyph(data_src, Scatter(x="x", y="y",
                marker=glyph, tags=["glyph"], size=4, **kwargs))
        else:
            rend = self._fig.add_glyph(data_src, glyph(**kwargs))

        rend.update(name=f"fig{self.f_num}_ren_{self.rend_idx}", tags=tags)
        if data_view is not None:
            rend.update(view=data_view)

        if (data is not None and hasattr(data, "errors") and 
            data.errors is not None):
            self.add_errors(data_src, data.errors)

        if legend is not None:
            rend.tags.append(legend)
            self.legend_items[legend] = self._fig.select(tags=[legend],
                                                        type=GlyphRenderer)

        self.rend_idx += 1

    def add_errors(self, data, errors, base="x", dim="y", **kwargs):
        """
        Add Error a bars

        Parameters
        ----------
        data: :obj:`dict` or :obj:`ColumnDataSource`
            All data in a dictionary format or a column data format
        errors: :obj:`np.array` or :obj:`da.array`
            Errors in an numpy or dask array
        base: :obj:`str`
            The name of the data column for the x-axis
        dim: :obj:`str`
            Name of the axis on which to add the errors

        Returns
        -------
        Nothing
        """

        if  type(data) != ColumnDataSource:
            data = self.create_data_source(data)
        
        data.add(data.data[dim] + errors, name="upper")
        data.add(data.data[dim] - errors, name="lower")

        ebar = Whisker(source=data, base=base, lower="lower", upper="upper",
                visible=False,# upper_head=None, lower_head=None,
                line_color="red",line_cap="round",
                tags=["ebar", f"{self.rend_idx}"], **kwargs)
        ebar.upper_head.line_color = "red"
        ebar.lower_head.line_color = "red"
        self._fig.add_layout(ebar)

    def write_out(self, filename=None):
        if filename is None:
            return

        if ".html" not in filename:
            filename += ".html"
        output_file(filename)
        save(self._fig, filename=filename, title=os.path.splitext(filename)[1])

    def add_categorical_colourbar(self, caxis, plargs, visible=False,
        position="right"):
        """Add a colourbar for categorical data
        Parameters
        ----------
       
        caxis: :obj:`str`
            Name of the categorizing axis
        plargs: :obj:`ragavi.ragdata.PlotArgs`
            An object containing cmap-list with colours, n_categories-number
            of categories, cat_map- category id to name mapping dictionary
        """
        self._fig.plot_width = int(self._fig.plot_width * 0.98)
        snitch.debug("Adding colour bar")
        caxis, labels = caxis.capitalize(), list(plargs.cat_map.values())

        rends = []

        # legend height
        lh = int((self._fig.plot_height / plargs.n_categories) * 0.85)

        for c, cidx in enumerate(plargs.cat_map.keys()):
            # ignore other cats. This was added because of corr mapings
            if c >= plargs.n_categories:
                continue
            ssq = dict(
                x=[plargs.xmin, plargs.xmin], y=[plargs.ymin, plargs.ymin])

            self.add_glyphs(Line, ssq, legend=str(plargs.cat_map[cidx]),
                            line_color=plargs.cmap[c], line_width=lh)

        self.add_legends(
            group_size=60, position=position, visible=visible,
            name="cb_legend", title_standoff=2, border_line_width=1,
            title=caxis, glyph_height=lh, glyph_width=25,
            title_text_line_height=0.3, padding=4,
            title_text_font="monospace", title_text_font_style="normal",
            title_text_font_size="10pt", title_text_align="left",
            label_text_font_style="bold", spacing=0, margin=0,
            label_height=5, label_width=10, label_text_font_size="8pt",
            orientation="vertical"
        )
        snitch.debug("Colour bar Added")

    
    def add_legends(self, group_size=16, position="above", **kwargs):
        """
        Group legend items into group_size each, create legend objects and
        attach them to figure. This function also adds a batch number to
        the renderer indicating its group id. It is specified as b#.

        Parameters
        ----------
        group_size: :obj:`int`
            Number of antennas that each legend group will have
        kwargs:
            A bunch of key word arguments that will be passed directly
            to the Legend object
        """
        kwargs = set_multiple_defaults(kwargs, dict(
            click_policy="hide", glyph_height=20,
            glyph_width=20, label_text_font_size="8pt",
            label_text_font="monospace",
            location="top_left", margin=1, orientation="horizontal", 
            level="annotation", spacing=1, padding=2, visible=False))

        self.legend_items = list(self.legend_items.items())
        legends = []

        # add n_groups to class this is the number of batches avail
        self.n_groups = math.ceil(len(self.legend_items) / group_size)
        for idx in range(self.n_groups):
            items = self.legend_items[idx*group_size: group_size*(idx+1)]
            for leg, rend_list in items:
                for rend in rend_list:
                    # add batch tag to the renderers
                    rend.tags.append(f"b{idx}")
            # push the legends into the stack
            legends.append(Legend(items=items, tags=["legend"], **kwargs))
        legends.reverse()
        for item in legends:
            self._fig.add_layout(item, position)
       
    def link_figures(self, *others):
        """
        Link to or more items of this class
        others: tuple
        """
        for idx, renderer in enumerate(self._fig.renderers):
            for other_fig in others:
                # Link their renderer's visible properties
                renderer.js_link("visible", other_fig.fig.renderers[idx], 
                    "visible")
                other_fig.fig.renderers[idx].js_link("visible", renderer,
                    "visible")

    @property
    def fig(self):
        return self._fig

    def write_out_static(self, mdata, filename=None, dpi=None, group_size=16):
        """
        Save plots in png,ps, pdf and svg format for all the fields
        """
        if filename is None:
            return

        snitch.info("Setting up static image")
        name, ext = os.path.splitext(filename)
        ext = ext.lower() if ext else ".png"

        if dpi is None and ext==".png":
            dpi = 300
        else:
            dpi = 72

        # set up renderer sorting function
        skey = lambda x: int(x.id)

        plt.close("all")
        fig, ax = plt.subplots(nrows=1, ncols=len(mdata.active_fields),
                              sharex="row", squeeze=True,
                              gridspec_kw=dict(wspace=0.2, hspace=0.3),
                              figsize=(20, 8), dpi=dpi)

        # ensure ax is an array because squeezing is on        
        if not hasattr(ax, "len"):
            ax = np.array([ax])

        for idx, fid in enumerate(mdata.active_fields):
            rends = sorted(self._fig.select(tags=f"f{fid}"), key=skey)
            if len(rends) == 0:
                snitch.warning(f"No renderers found tagged f{fid}")
            for rend in rends:
                src = dict(rend.data_source.data)
                if "flags" in src:
                    src["flags"] = ~src["flags"] 
                else:
                    src["flags"] = None
                msize = 5 / (src["x"].size / 2000)
                msize = 4 if msize > 4 else msize
                mscale = 10 // msize

                ax[idx].plot(
                    src["x"],
                    np.ma.masked_array(data=src["y"], mask=src["flags"]),
                    "o", color=rend.glyph.fill_color, label=src["ant"][0],
                    markersize=msize)
        
            ax[idx].set_xlabel(self._fig.xaxis.axis_label)

            ax[idx].set_ylabel(self._fig.yaxis.axis_label)
            ax[idx].set_title(mdata.reverse_field_map[fid] + " "
                              + self._fig.select_one({"tags": "title"}).text)

        ax_handles, labels = ax[0].get_legend_handles_labels()
        labels, indices = np.unique(labels, return_index=True)
        ax_handles = np.array(ax_handles)[indices]
        ax[0].legend(ax_handles, labels, loc=(0, 1.2), ncol=group_size,
                  markerscale=mscale, fontsize=9, labelspacing=0.3,
                  title="Antenna", columnspacing=1.0)

        fig.suptitle(f"Table: {mdata.ms_name}", ha="center")
        new_filename = f"{name}_{self._fig.yaxis.axis_label}{ext}"
        fig.savefig(new_filename, bbox_inches='tight')
        
        snitch.info(f"Image at: {new_filename}")

    def potato(self, mdata, filename=None, dpi=None, group_size=None):
        """
        Save plots in png,ps, pdf and svg format split out per field,
        and antenna batch. Remember flags were inverted to make cds views!!!
        """        

        if filename is None:
            return 
        snitch.info("Setting up static image with subplots")
        name, ext = os.path.splitext(filename)
        ext = ext.lower() if ext else ".png"

        if dpi is None and ext==".png":
            dpi = 300

        # set up renderer sorting function
        skey = lambda x: int(x.id)
        
        marks = ["o", "x", "^", "+"]

        if mdata.corr_map is not None:
            marks = {mdata.reverse_corr_map[i]: m for i,m in enumerate(marks) if i in mdata.reverse_corr_map}
        else:
            marks = {i: m for i, m in enumerate(marks)}
                
        if group_size is None or group_size > len(mdata.active_antennas):
            group_size = len(mdata.active_antennas)

        ncols = int(np.sqrt(group_size))
        ncols = 5 if ncols > 5 else ncols
        nrows = int(np.ceil(group_size/ncols))
        if nrows > ncols:
            ncols, nrows = nrows, ncols
        
        for idx, fid in enumerate(mdata.active_fields):
            for bid in range(self.n_groups):
                # each group has 16 antennas
                plt.close("all")
                fig, ax = plt.subplots(
                    nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                    squeeze=False, gridspec_kw=dict(wspace=0, hspace=0.4),
                    figsize=(20, 8), dpi=dpi)
                row = -1
                for aidx, aid in enumerate(mdata.active_antennas):
                    if type(aid) is not int:
                        aid = mdata.ant_map[aid]
                    frends = [rend for rend in self._fig.renderers
                        if {f"b{bid}",f"a{aid}",f"f{fid}"}.issubset(
                            set(rend.tags))]
            
                    if len(frends)==0:
                        continue
                    frends = sorted(frends, key=skey)
                    col = aidx % ncols
                    
                    if col == 0:
                        row += 1
                    
                    for ridx, rend in enumerate(frends):
                        src = dict(rend.data_source.data)
                        if "flags" in src:
                            src["flags"] = ~src["flags"]
                        else:
                            src["flags"] = None
                        msize = 5 / (src["x"].size / 2000)
                        msize = 4 if msize > 4 else msize
                        mscale = 10 // msize

                        ax[row, col].plot(src["x"],
                            np.ma.masked_array(data=src["y"],
                                mask=src["flags"]),
                            marks[src['corr'][0]],
                            color=rend.glyph.fill_color,
                            label= f"corr {src['corr'][0]}",
                            markersize=msize)
                        ax_handles, labels = ax[row, col].get_legend_handles_labels()
                        labels = np.unique(labels).tolist()
                        ax[row, col].legend(ax_handles, labels, loc=(0, 1.02),
                            ncol=4, markerscale=mscale, fontsize=9,
                            labelspacing=0.3, title=f"{src['ant'][0]}",
                            columnspacing=1.0)
                        ax[row,col].tick_params("x", labelrotation=30)
                        for label in ax[row, col].get_xticklabels():
                            label.set_ha("right")
                fig.suptitle(f"Table: {mdata.ms_name} "
                              + mdata.reverse_field_map[fid], ha="center")
                fig.text(0.5, 0.04, self._fig.xaxis.axis_label, ha='center',
                        va='center')
                fig.text(0.1, 0.5, self._fig.yaxis.axis_label, ha='center',
                        va='center', rotation='vertical')

                new_filename = (f"{name}"
                    + f"_{self._fig.yaxis.axis_label}"
                    + f"_{mdata.reverse_field_map[fid]}"
                    + f"_grp{bid}{ext}")
                snitch.info(f"Sub image at: {new_filename}")
                fig.savefig(new_filename, bbox_inches='tight')


# Look here: https://www.geeksforgeeks.org/3d-scatter-plotting-in-python-using-matplotlib/
# https://people.duke.edu/~ccc14/pcfb/numpympl/MatplotlibBarPlots.html
# https://stackoverflow.com/questions/27267683/using-matplotlib-polycollection-to-plot-data-from-csv-files
