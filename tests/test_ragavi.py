"""Tests for the current gain plotter data and rendering interfaces."""

import numpy as np
import pytest
import xarray as xr
import dask.array as da
from bokeh.embed import file_html
from bokeh.models import (Button, CheckboxGroup, ColumnDataSource,
                          GlyphRenderer, Legend)
from bokeh.resources import CDN

from ragavi.plotting import create_bk_fig
from ragavi.arguments import gains_argparser
import ragavi.ragavi as gain_module
from ragavi.ragavi import DataCoreProcessor, field_marker, field_symbol, make_plots


@pytest.mark.parametrize(
    "axis, expected",
    [
        ("amplitude", [np.sqrt(2), np.sqrt(2)]),
        ("real", [1, 1]),
        ("imaginary", [1, -1]),
        ("phase", [45, -45]),
    ],
)
def test_gain_data_processing(axis, expected):
    values = xr.DataArray(np.array([1 + 1j, 1 - 1j]), dims="row")
    processor = DataCoreProcessor(None, "unused", "G", axis)

    np.testing.assert_allclose(processor.process_data(values), expected)


@pytest.mark.parametrize("gain_type, expected", [("B", "channel"), ("G", "time"),
                                                  ("K", "time"), ("F", "time")])
def test_gain_default_xaxis(gain_type, expected):
    processor = DataCoreProcessor(None, "unused", gain_type, "amplitude")
    assert processor.xaxis == expected


@pytest.mark.parametrize("field_index", [0, 4, 5, 6, 11, 12])
@pytest.mark.parametrize("flagged", [False, True])
def test_field_markers_render_for_many_fields(field_index, flagged):
    y = np.array([1.0, np.nan if flagged else 2.0])
    source = ColumnDataSource(data={"x": np.array([1, 2]), "y1": y})
    glyph, whisker = make_plots(source, fid=field_index, yidx=1,
                                yerr=(y + 0.1, y - 0.1))

    assert field_symbol(field_index) == field_symbol(field_index % 6)
    if flagged:
        assert glyph.marker == "fmarkers"
        assert source.data["fmarkers"].tolist() == [field_marker(field_index),
                                                      "circle_x"]
    else:
        assert glyph.marker == field_marker(field_index)

    fig = create_bk_fig(xlab="Time", ylab="Amplitude", title="Gain plot",
                        pw=900, ph=710, x_name="time", y_name="amplitude",
                        add_grid=True, add_xaxis=True, add_yaxis=True,
                        fix_plotsize=False)
    renderer = fig.add_glyph(source, glyph)
    fig.add_layout(whisker)
    fig.add_layout(Legend(items=[("field", [renderer])]), "above")

    assert fig.width == 900
    assert fig.height == 710
    assert "Gain plot" in file_html(fig, CDN, "Gain plot")


def test_make_plots_without_error_bars():
    source = ColumnDataSource(data={"x": np.array([1, 2]),
                                    "y1": np.array([1.0, 2.0])})
    glyph, whisker = make_plots(source, fid=6, yidx=1)

    assert glyph.marker == field_marker(6)
    assert whisker is None


def test_sparse_gain_plot_selection_metadata(monkeypatch):
    """Sparse fields, correlations, and antenna batches retain their keys."""
    subtables = []
    for field_id in range(7):
        for antenna in (range(17) if field_id == 0 else [16]):
            flags = np.zeros((2, 1, 2), dtype=bool)
            if field_id == 6:
                flags[0, 0, 0] = True
            subtable = xr.Dataset(
                {
                    "CPARAM": (("row", "chan", "corr"), da.from_array(
                        np.array([[[1 + 1j, 2 + 1j]],
                                  [[1 - 1j, 2 - 1j]]]))),
                    "FLAG": (("row", "chan", "corr"), da.from_array(flags)),
                    "PARAMERR": (("row", "chan", "corr"), da.from_array(
                        np.full((2, 1, 2), 0.1))),
                    "TIME": ("row", da.from_array(
                        np.array([3506716801.0, 3506716802.0]))),
                    "SCAN_NUMBER": ("row", [1, 2]),
                },
                attrs={"SPECTRAL_WINDOW_ID": 0, "FIELD_ID": field_id,
                       "ANTENNA1": antenna},
            )
            subtables.append(subtable)

    monkeypatch.setattr(gain_module, "get_table", lambda *a, **kw: (subtables, "G"))
    monkeypatch.setattr(gain_module, "get_time_range",
                        lambda *a, **kw: (3506716801.0, 3506716802.0))
    monkeypatch.setattr(gain_module, "stats_display",
                        lambda **kw: [kw["spwid"], f"field{kw['field']}", kw["corr"], 1.0])
    monkeypatch.setattr(gain_module.vu, "get_antennas",
                        lambda *a, **kw: xr.DataArray(
                            [f"ant{i}" for i in range(17)]))
    monkeypatch.setattr(gain_module.vu, "get_fields",
                        lambda *a, **kw: xr.DataArray([f"field{i}" for i in range(7)]))
    monkeypatch.setattr(gain_module.vu, "get_frequencies",
                        lambda *a, **kw: xr.DataArray([[1e9]], dims=("spw", "chan")))
    monkeypatch.setattr(gain_module.vu, "update_log_levels", lambda *a, **kw: None)
    monkeypatch.setattr(gain_module.vu, "update_logfile_name", lambda *a, **kw: None)
    monkeypatch.setattr(gain_module, "set_tempdir", lambda *a, **kw: None)
    output = {}
    monkeypatch.setattr(gain_module, "save_html",
                        lambda name, layouts: output.update(
                            layout=layouts[0],
                            html=file_html(layouts[0], CDN, "Gain plot")))

    options = gains_argparser().parse_args(
        ["-t", "synthetic.G", "-c", "0,1", "-y", "ap"])
    assert gain_module.main(options=options) == 0
    assert "field6" in output["html"]

    renderers = [renderer for renderer in output["layout"].select(
        {"type": GlyphRenderer}) if renderer.tags and
        renderer.tags[0].get("ragavi_selection")]
    assert len(renderers) == 92
    selection_keys = {tuple(renderer.tags[0][key] for key in
                            ("field", "corr", "batch")) for renderer in renderers}
    expected_keys = {(0, corr, batch) for corr in (0, 1)
                     for batch in (0, 1)}
    expected_keys.update((field, corr, 1) for field in range(1, 7)
                         for corr in (0, 1))
    assert selection_keys == expected_keys
    assert sum(renderer.visible for renderer in renderers) == 32
    assert all(renderer.muted_glyph is not None for renderer in renderers)
    assert sum("change:muted" in renderer.js_property_callbacks
               for renderer in renderers) == 46

    legends = list(output["layout"].select({"type": Legend}))
    assert len(legends) == 4
    assert all(legend.click_policy == "mute" for legend in legends)

    checkboxes = list(output["layout"].select({"type": CheckboxGroup}))
    selection_widgets = [widget for widget in checkboxes if any(
        callback.code == gain_module.selection_callback()
        for callback in widget.js_property_callbacks.get("change:active", []))]
    assert len(selection_widgets) == 4
    buttons = list(output["layout"].select({"type": Button}))
    assert any(button.label == "Select all antennas" for button in buttons)
