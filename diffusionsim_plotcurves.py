import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt

params = {
  'axes.labelsize': 12,
  'font.size': 12,
  'legend.fontsize': 12,
  'xtick.labelsize': 12,
  'ytick.labelsize': 12,
  'text.usetex': True,
  'figure.figsize': [6, 4],
  'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
   }
plt.rcParams.update(params)

fig = plt.figure(figsize=(10,40))
plt.axes(frameon=0) # turn off frames
plt.grid(axis='y', color='0.9', linestyle='-', linewidth=1)

ax = plt.subplot(511)
plt.title('Training Loss vs Iterations for Reaction-Diffusion Equation')
ax.set_yscale('log')

marker_size = 10

prefixes = [("0.001"),("0.002"), ("0.005"), ("0.01"), ("0.1"), ("1.0")]


for pref in prefixes:
    my_data = genfromtxt('{}_loss.csv'.format(pref), delimiter=',')
    ax.plot(my_data, label=pref, ms=marker_size)

ax.legend(title='Memory fraction')
fig.savefig('diffusion_sim_curves.pdf')
