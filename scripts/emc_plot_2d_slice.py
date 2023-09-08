import numpy
import h5py
import matplotlib.pyplot
import argparse

from mpl_toolkits.axes_grid1 import make_axes_locatable

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
args = parser.parse_args()

with h5py.File(args.model, "r") as file_handle:
    data = file_handle["real"][file_handle["real"].shape[0]//2, :, :]

fig = matplotlib.pyplot.figure("Plot 2D slice", figsize=(10, 10))
ax = fig.add_subplot(111)
im = ax.imshow(data, norm=matplotlib.colors.LogNorm(), vmin=0.1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
matplotlib.pyplot.show()
