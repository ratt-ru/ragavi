import numpy as np

from itertools import product, zip_longest
from bokeh.layouts import column, grid
from bokeh.models import (Button, CDSView, CheckboxGroup, ColumnDataSource,
                          CustomJS, PreText, Scatter, Slider, Title, Whisker)
from bokeh.models.widgets import Div, DataTable, TableColumn

from processing import Processor
from widgets import F_CODES

from ipdb import set_trace


def activate_batch_if_no_other_sel(sel1):
    """Activate only selected batch if only the batch selection is ticked
    sel1: Name of other selector to check
    cb_obj: is the object to whose callback this function will be attached
    """
    return """
        errors.forEach(error => error.visible = false);
        terr.active = [];

        if (bsel.active.length > 0
            && %s.active.length == 0
            && cb_obj.active.length == 0){
            for (let b = 0; b < nbatches; b++) {
                if (bsel.active.includes(b)) {
                    ax.renderers.filter(({ tags }) => {
                        return tags.includes(`b${b}`);
                    }).forEach(rend => rend.visible = true);
                }
            }
        }
    """%(sel1)


def redux_fn():
    return """
        function reduceArray(inlist, widget, size, tag){
                /*
                inlist: Array to be reduced
                widget: widget this object depends on to show renderers
                size: size of items in expected in that widget 
                       that is passed into this function
                tag: tagnumber prefix. batch: "b", spw: "s", field: "f" corr: "c"
                */
                let intermediate = [];
                for (let i=0; i<size; i++) {
                    if (widget.active.includes(i)){
                        intermediate = intermediate.concat(
                            inlist.filter(({ tags }) => {
                                return tags.includes(`${tag}${i}`);
                                }));
                    }
                }
                return intermediate;
            }

    """


def ant_selector_callback():
    """JS callback for the selection and de-selection of antennas

    Returns
    -------
    code : :obj:`str`
    """

    code = """
        /*  Make all the Antennas available active
            ax: list containing glyph renderers
            bsel: batch selection group button
            nbatches: Number of batches available
            csel: corr_selector widget,
            ncorrs: Number of avaiblable corrs,
        */
        errors.forEach(error => error.visible = false);
        terr.active = [];
    
        if (cb_obj.active.includes(0)){
            ax.renderers.forEach(rend => rend.visible = true);
        
            //activate all the checkboxes whose antennas are active
            bsel.active = [...Array(nbatches).keys()];
            csel.active = [...Array(ncorrs).keys()];
            fsel.active = [...Array(nfields).keys()];
        }
        else{
            ax.renderers.forEach(rend => rend.visible = false);
            bsel.active = [];
            csel.active = [];
            fsel.active = [];
        }
        """

    return code


def batch_selector_callback():
    """JS callback for batch selection Checkboxes"""
    code = """
        errors.forEach(error => error.visible = false);
        terr.active = [];

        var final_array = ax.renderers;

        if (fsel.active.length > 0) {
            final_array = reduceArray(final_array, fsel, nfields, "f");
        }
        if (csel.active.length > 0){
            final_array = reduceArray(final_array, csel, ncorrs, "c");
        }

        for (let b = 0; b < nbatches; b++) {
            if (cb_obj.active.includes(b)) {
                final_array.filter(({ tags }) => {
                    return tags.includes(`b${b}`);
                }).forEach(rend => rend.visible = true);
            }
            else{
                final_array.filter(({ tags }) => {
                    return tags.includes(`b${b}`);
                }).forEach(rend => rend.visible = false);
            }
        }
    """
    return redux_fn() + code


def corr_selector_callback():
    """Correlation selection callback"""
    code = """        
        var final_array = ax.renderers;
        
        if (bsel.active.length > 0){
            final_array = reduceArray(final_array, bsel, nbatches, "b");
        }

        if (fsel.active.length > 0) {
            final_array = reduceArray(final_array, fsel, nfields, "f");
        }
        
        for (let corr in corrs){
            if (cb_obj.active.includes(Number(corr))) {
                final_array.filter(({tags}) =>{
                    return tags.includes(`c${corrs[corr]}`);
                }).forEach(rend => rend.visible = true);
            }
            else{
                final_array.filter(({tags}) =>{
                    return tags.includes(`c${corrs[corr]}`);
                }).forEach(rend => rend.visible=false);
            }
        }
       """
    return redux_fn() + code + activate_batch_if_no_other_sel("fsel")


def field_selector_callback():
    """Return JS callback for field selection checkboxes"""
    code = """
        var final_array = ax.renderers;

        if (bsel.active.length > 0){
            final_array = reduceArray(final_array, bsel, nbatches, "b");
        }
        if (csel.active.length > 0) {
            final_array = reduceArray(final_array, csel, ncorrs, "c");
        }

        for (let f=0; f<nfields; f++) {
            if (cb_obj.active.includes(f)) {
                final_array.filter(({ tags }) => {
                    return tags.includes(`f${f}`);
                }).forEach(rend => rend.visible = true);
            }
            else{
                final_array.filter(({ tags }) => {
                    return tags.includes(`f${f}`);
                }).forEach(rend => rend.visible = false);
            }
        }
       """
    return redux_fn() + code + activate_batch_if_no_other_sel("csel")


# Show additional data callbacks

def toggle_flagged_callback():
    """JS callback for the flagging button

    Returns
    -------
    code : :obj:`str`
    """

    code = """
            var src, view;
            for(let i in sources){
                src = sources.filter(({tags}) => tags.includes(Number(i)))[0];
                view = views.filter(({tags}) => tags.includes(Number(i)))[0];
                if (cb_obj.active.includes(0)){
                    view.filters[0].booleans = src.data.noflags;
                    src.change.emit()
                }
                else{
                    view.filters[0].booleans = src.data.flags;
                    src.change.emit()
                }
            }
           """
    return code


def toggle_error_callback():
    """Toggle errors only for visible renderers"""
    code = """
        let rendNumbers = [];
        let renderers = ax.renderers.filter(({visible}) => visible==true);

        //collect unique renderer number from its name into items array
        // Only visible renderers are considered names: "fig0_ren_0"
        renderers.forEach(rend => rendNumbers.push(rend.name.split("_")[2]))
        
        for (const num of rendNumbers){
            //get ebar with that number and make it visible.
            if (cb_obj.active.includes(0)){
                errors.filter(
                    ({tags}) => tags.includes(num)).forEach(
                        error=> error.visible=true);
            }
            else{
                errors.filter(
                    ({tags}) => tags.includes(num)).forEach(
                        error=> error.visible=false);
            }
        }
            """
    return code


def legend_toggle_callback():
    """JS callback for legend toggle Dropdown menu

    Returns
    -------
    code : :obj:`str`
    """
    code = """
        if (cb_obj.active.includes(0)){
            legends.forEach( legend => legend.visible = true);
        }
        else{
            legends.forEach( legend => legend.visible = false);
        }
        """
    return code


# Plot layout callbacks

def size_slider_callback():
    """JS callback to select size of glyphs"""

    code = """
            glyphs.forEach( glyph => glyph.size = cb_obj.value);
           """
    return code


def alpha_slider_callback():
    """JS callback to alter alpha of glyphs"""

    code = """
            glyphs.forEach( glyph => glyph.fill_alpha = cb_obj.value)
           """
    return code


def axis_fs_callback():
    """JS callback to alter axis label font sizes

    Returns
    -------
    code : :obj:`str`
    """
    code = """
            axes.forEach(axis => axis.axis_label_text_font_size = `${cb_obj.value}pt`);
           """
    return code


def title_fs_callback():
    """JS callback for title font size slider"""
    code = """
        titles.forEach(title => title.text_font_size = `${cb_obj.value}pt`);
        """
    return code


# Download data selection

def save_selected_callback():
    code = """
        var out = `x, y, ant, corr, field, scan, spw\n`;

        for (let src of sources){
            let sel_idx = src.selected.indices;
            let data = src.data;
            for (let idx of sel_idx){
                out += `${data['x'][idx]}, ` +
                        `${data['y'][idx]}, ` +
                        `${data['ant'][idx]}, ` +
                        `${data['corr'][idx]}, ` +
                        `${data['field'][idx]}, ` +
                        `${data['scan'][idx]}, ` +
                        `${data['spw'][idx]}\n`;
            }
        }
        let answer = confirm("Download selected data?");
        if (answer){
            let file = new Blob([out], {type: 'text/csv'});
            let element = window.document.createElement('a');
            element.href = window.URL.createObjectURL(file);
            element.download = "data_selection.csv";
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
        }

    """

    return code


def gen_checkbox_labels(msdata, group_size=8):
    """
    Auto-generating Check box labels

    Parameters
    ----------
    ant_names : :obj:`list`
        List of the antenna names
    group_size : :obj:`int`
        Number of items in a single batch
    Returns
    ------
    List containing batch labels for the batch selection check box group
    """
    ant_names = [msdata.ant_names[a] for a in msdata.active_antennas]
    lhs = ant_names[slice(0, None, group_size)]
    rhs = ant_names[slice(group_size-1, None, group_size)]
    last = ant_names[-1]

    return [f"{l} - {r}" for l, r in zip_longest(lhs, rhs, fillvalue=last)]


def make_stats_table(msdata, data_column, yaxes, subs):
    """
    Get the list of subs from theg get_table function in main
    """
    columns = ["spw", "field", "corr"] + yaxes.split(",")
    stats = {col: [] for col in columns}

    for sub in subs:
        for yaxis, corr in product(yaxes, msdata.active_corrs):
            pro = Processor(sub[data_column]).calculate(yaxis).sel(corr=corr)
            flags = sub.FLAG.sel(corr=corr)
            stats["spw"].append(sub.SPECTRAL_WINDOW_ID)
            stats["field"].append(sub.FIELD_ID)
            stats["corr"].append(corr)
            stats[yaxis].append(
                f"{np.nanmedian(pro.where(flags == False).values):.4}")
            
            # print(f"y-axis: {yaxis}, field: {f_names[field]}"+
            #       f"corr: {str(corr)} median: {str(med_y)}")

    stats = ColumnDataSource(data=stats)
    
    columns = [TableColumn(field=col, title=col.title()) for col in columns]

    return column([Div(text="Median Statistics"),
        DataTable(source=stats, columns=columns, fit_columns=True, height=150,
                  max_height=180, max_width=600, sizing_mode="stretch_width")],
        sizing_mode="stretch_both")


def make_table_name(version, tname):
    """Create div for stats data table"""
    div = PreText(text=f"ragavi: v{version} | Table: {tname}",
                  margin=(1, 1, 1, 1))
    return div


def make_widgets(msdata, fig, group_size=8):
    """
    Set up widgets and attach renderers to them
    
    Parameters
    ----------
    msdata: :obj:`ragdata.MsData`
        Object containing infoa about the current MS. Gotten from the main
        script
    fig: :obj:`bokeh.models.Figure`
        Bokeh figure object containing the attached renderers

    Returns
    -------
    Nothing    
    """

    batch_labels = gen_checkbox_labels(msdata, group_size)
    
    #number of batches avail. Depends on the group_size
    nbatch = len(batch_labels)
    corr_labels = [f"Corr {corr.upper()}" for corr in msdata.active_corrs]

    field_labels = [f"Dir {msdata.reverse_field_map[f]} {F_CODES[fi]}"
                    for fi, f in enumerate(msdata.active_fields)]

    # Selection group
    #select and deselect all antennas
    ant_selector = CheckboxGroup(labels=["Select all antennas"], active=[],
                                width=150, height=30)

    #select antennas in batches
    batch_selector = CheckboxGroup(labels=batch_labels, active=[0], width=150,
                                    height=70)

    corr_selector = CheckboxGroup(labels=corr_labels, active=[], width=150)

    field_selector = CheckboxGroup(labels=field_labels, active=[], width=150,
                                   height=30)

    # configuring toggle button for showing all the errors
    toggle_error = CheckboxGroup(labels=["Show error bars"], active=[],
                                 width=150, height=30)
    toggle_error.js_on_change("active", CustomJS(
        args=dict(ax=fig, errors=fig.select(tags=["ebar"], type=Whisker)),
        code=toggle_error_callback()))
    
    ant_selector.js_on_change("active", CustomJS(
        args=dict(ax=fig, bsel=batch_selector, csel=corr_selector,
                  fsel=field_selector, nbatches=nbatch, 
                  nfields=len(msdata.active_fields),
                  ncorrs=len(msdata.active_corrs), terr=toggle_error,
                  errors=fig.select(tags=["ebar"], type=Whisker)),
        code=ant_selector_callback()))
    batch_selector.js_on_change("active", CustomJS(
        args=dict(ax=fig, bsize=group_size,
                  nants=msdata.num_ants, nbatches=nbatch, 
                  nfields=len(msdata.active_fields),
                  ncorrs=len(msdata.active_corrs), corrs=msdata.active_corrs,
                  csel=corr_selector, antsel=ant_selector, fsel=field_selector,
                  errors=fig.select(tags=["ebar"], type=Whisker),
                  terr=toggle_error),
        code=batch_selector_callback()))
    corr_selector.js_on_change("active", CustomJS(
        args=dict(bsel=batch_selector, fsel=field_selector,
                  nants=msdata.num_ants, corrs=msdata.active_corrs, 
                  nfields=len(msdata.active_fields),
                  nbatches=nbatch, ax=fig, errors=fig.select(tags=["ebar"],
                  type=Whisker), terr=toggle_error),
        code=corr_selector_callback()))
    field_selector.js_on_change("active", CustomJS(
        args=dict(bsel=batch_selector, csel=corr_selector,
                  nants=msdata.num_ants, nfields=len(msdata.active_fields),
                  ncorrs=len(msdata.active_corrs), nbatches=nbatch,
                  ax=fig, errors=fig.select(tags=["ebar"], type=Whisker),
                  terr=toggle_error),
        code=field_selector_callback()))

    # Additions group
    # Checkbox to hide and show legends
    legend_toggle = CheckboxGroup(labels=["Show legends"], active=[0], width=150,
                                height=30)
    legend_toggle.js_on_change("active", CustomJS(
        args=dict(legends=fig.select(tags=["legend"])),
        code=legend_toggle_callback()))


    save_selected = Button(label="Download data selection",
                            button_type="success", margin=(7, 5, 3, 5),
                            sizing_mode="fixed", width=150, height=30)
    save_selected.js_on_click(CustomJS(args=dict(
        sources=fig.select(type=ColumnDataSource)),
        code=save_selected_callback()))
        

    toggle_flagged = CheckboxGroup(labels=["Show flagged data"], active=[],
                                width=150, height=30)
    toggle_flagged.js_on_change("active", CustomJS(
        args=dict(sources=fig.select(type=ColumnDataSource),
                  views=fig.select(type=CDSView)),
        code=toggle_flagged_callback()))


    # margin = [top, right, bottom, left]
    size_slider = Slider(end=15, start=0.4, step=0.1,
                        value=10, title="Glyph size", margin=(3, 5, 7, 5),
                        bar_color="#6F95C3", width=150, height=30)
    size_slider.js_on_change("value",CustomJS(
        args=dict(slide=size_slider,
                  glyphs=fig.select(tags="glyph")),
        code=size_slider_callback()))


    # Alpha slider for the glyphs
    alpha_slider = Slider(end=1, start=0.1, step=0.1, value=1,
                            margin=(3, 5, 7, 5), title="Glyph alpha",
                            bar_color="#6F95C3", width=150, height=30)
    alpha_slider.js_on_change("value",CustomJS(
        args=dict(glyphs=fig.select(tags="glyph")),
        code=alpha_slider_callback()))


    axis_fontslider = Slider(end=20, start=3, step=0.5, value=10,
                            margin=(7, 5, 3, 5), title="Axis label size",
                            bar_color="#6F95C3", width=150, height=30)
    axis_fontslider.js_on_change("value", CustomJS(
        args=dict(axes=fig.axis), code=axis_fs_callback()))


    title_fontslider = Slider(end=25, start=10, step=1, value=15,
                                margin=(3, 5, 7, 5), title="Title size",
                                bar_color="#6F95C3", width=150, height=30)
    title_fontslider.js_on_change("value", CustomJS(
        args=dict(titles=fig.select(tags="title")),
        code=title_fs_callback()))

    return [
        grid(children=[
            [ant_selector, toggle_error, size_slider, title_fontslider],
            [legend_toggle, toggle_flagged, alpha_slider, axis_fontslider],
            [Div(text="Select antenna group"), Div(text="Directions"),
                Div(text="Select correlation")],
            [batch_selector, field_selector, corr_selector]]),
        save_selected]
