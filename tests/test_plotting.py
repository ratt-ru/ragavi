import numpy as np

from ipdb import set_trace
from bokeh.models import Circle

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

new_figure = plotting.FigRag(plot_args=dict(plot_width=300, plot_height=300, add_toolbar=True))

new_figure.update_xlabel("Single")
new_figure.update_ylabel("Squared")
new_figure.update_title("Squared vs Single Plot Test")

print(".", end=" ")
for i in range(13):
    new_figure.add_glyphs(Circle, {"x": x, "y": np.array(y)*i}, errors=errors, legend=f"Legend {i}")

new_figure.add_legends(group_size=5)
new_figure.write_out("test_plot.html")
print(".", end="\n")

print("Done")
