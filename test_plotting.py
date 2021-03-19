import numpy as np

from ipdb import set_trace
from bokeh.models import Circle
from bokeh.palettes import linear_palette, Magma256
import plotting

print("Running ", end=" ")
x = range(10)
y = [_**2 for _ in x]
errors = np.random.random_sample(len(x))

print(".", end=" ")

for name, edx in [("x", x), ("y", y), ("errs", errors)]:
    print(name)
    print("="*20)
    print(edx)

print(".", end=" ")

new_figure = plotting.FigRag(add_toolbar=True, plot_args=dict(plot_width=300, plot_height=300))

new_figure.update_xlabel("Single")
new_figure.update_ylabel("Squared")
new_figure.update_title("Squared vs Single Plot Test")

print(".", end=" ")
colours = linear_palette(Magma256, 13)
for i in range(13):
    new_figure.add_glyphs(Circle, {"x": x, "y": np.array(y)*i}, errors=errors, legend=f"Legend {i}",
                          size=12, fill_alpha=1, fill_color=colours[i], line_color=None)

new_figure.add_legends(group_size=5)
set_trace()
new_figure.write_out("test_plot.html")
print(".", end="\n")

print("Done")
