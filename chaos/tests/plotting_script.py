import numpy as np

from ipdb import set_trace
from bokeh.models import Circle
from bokeh.palettes import linear_palette, Magma256
from bokeh.layouts import gridplot

from holy_chaos.chaos import plotting
print("Running ", end=" ")
x = range(10)
y = [_**2 for _ in x]
errors = np.random.random_sample(len(x))

ys = ["Amplitude", "Phase"]

print(".", end=" ")

figures = []
for idx, _ in enumerate(ys):
    new_figure = plotting.FigRag(width=820, height=720, add_toolbar=True)

    new_figure.update_xlabel("Single")
    new_figure.update_ylabel(_)
    new_figure.update_title(f"{_} vs Single Plot Test")

    print(".", end=" ")
    colours = linear_palette(Magma256, 25)
    for i in range(13):
        new_figure.add_glyphs(Circle, {"x": x, "y": np.array(y)*i}, errors=errors, legend=f"Legend {i}",
                            size=12, fill_alpha=1, fill_color=colours[i], line_color=None)

    new_figure.add_legends(group_size=5, visible=True)

    figures.append(new_figure)

figures[0].link_figures(*figures[1:])

figures = [f.get_figure() for f in figures]

out = gridplot(children=figures, toolbar_location="left", ncols=2)
plotting.save(out, "grid.html")
print(".", end="\n")

print("Done")
