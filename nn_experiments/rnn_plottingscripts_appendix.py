import os
import re
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt

PKL_FILE_REGEX = re.compile(".+mnistirnn\.pkl")

def extract_plot_data(fname):
    with open(fname, 'rb') as f:
        all_ckpts = pickle.load(f)

    iterations, train_losses, test_losses, test_accuracies = [], [], [], []
    train_test_iterations, train_test_losses, train_test_accuracies = [], [], []
    epoch_times = []
    for ckpt in all_ckpts:
        iteration = ckpt[0]
        if iteration == 'final':  # Skip for now
            continue
        train_ckpt, test_ckpt = itemgetter('train', 'test')(ckpt[1])

        iterations.append(iteration)
        train_losses.append(train_ckpt['loss'])
        test_losses.append(test_ckpt['loss'])
        test_accuracies.append(test_ckpt['accuracy'])
        epoch_times.append(float(train_ckpt['time']))

        if 'train_test' in ckpt[1]:
            train_test_ckpt = ckpt[1]['train_test']
            train_test_iterations.append(iteration)
            train_test_losses.append(train_test_ckpt['loss'])
            train_test_accuracies.append(train_test_ckpt['accuracy'])

    return iterations, train_losses, test_losses, test_accuracies, train_test_iterations, train_test_losses, \
           train_test_accuracies, epoch_times

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
plt.title('Training Loss vs Iterations for IRNN on Sequential-MNIST')
ax.set_yscale('log')

ax2 = plt.subplot(512)
plt.title('Training Accuracy vs Iterations for IRNN on Sequential-MNIST')
ax2.set_ylim((15, 80))

ax3 = plt.subplot(513)
plt.title('Test Loss vs Iterations for IRNN on Sequential-MNIST')
ax3.set_yscale('log')

ax4 = plt.subplot(514)
plt.title('Test Accuracy vs Iterations for IRNN on Sequential-MNIST')
ax4.set_ylim((15, 80))
#ax4.hlines(y=92.0, xmin=0, xmax=1000)
ax4.grid(True)

ax5 = plt.subplot(515)
plt.title('Training time vs Iterations for IRNN on Sequential-MNIST')

marker_size = 10

EXP_BASE_DIR = '../rnn_stuff/mnist_rnn_exps'  # Replace with path to folder containing experiments
exp_folders = ['irnn_baseline', 'irnn_small_batch', 'rand_irnn_sparse', 'rand_irnn_sparse_full', 'rand_irnn_rp', 'rand_irnn_rp_full']
colors = ['pink', 'r', 'g', 'k', 'b', 'y']
markers = ['o', '^', 'x', '*', 'o', 'd']
labels = ['Baseline', 'Reduced batch', 'Same Sample', 'Different Sample', 'Project', 'Different Project']

for i, exp_folder in enumerate(exp_folders):
    exp_folder = os.path.join(EXP_BASE_DIR, exp_folder)
    for j, seed in enumerate(os.listdir(exp_folder)):
        data_dir = os.path.join(exp_folder, seed, "pickles")
        data_filename = os.path.join(data_dir, list(filter(PKL_FILE_REGEX.match, os.listdir(data_dir)))[0])

        iterations, train_losses, test_losses, test_accuracies, train_test_iterations, train_test_losses, \
        train_test_accuracies, epoch_times = extract_plot_data(data_filename)

        label = labels[i] if j == 0 else None

        marker = markers[i]
        exp_name = label
        color = colors[i]

        ax.plot(train_test_iterations, train_test_losses, marker=marker, label=exp_name, c=color, ms=marker_size, markevery=5)
        ax2.plot(train_test_iterations, train_test_accuracies, marker=marker, label=exp_name, c=color, ms=marker_size, markevery=5)
        ax3.plot(iterations, test_losses, marker=marker, label=exp_name, c=color, ms=marker_size, markevery=50)
        ax4.plot(iterations, test_accuracies, marker=marker, label=exp_name, c=color, ms=marker_size, markevery=50)
        ax5.plot(iterations, epoch_times, marker=marker, label=exp_name, c=color, ms=marker_size, markevery=50)


ax.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend(loc='center right')

#plt.show()
fig.savefig('../Plots/IRNN_all_curves.pdf')

