# Running Feedforward Experiments

## Requirements

See requirements.txt file for full pip requirements.\
Major requirements: torch, torchvision, tensorboard, matplotlib

NOTE: Only verified compatibility with torch version 1.3.1 + torchvision version 0.4.2\
torch version 1.5.0 is confirmed NOT to work.

## Reproducing results from paper

``cifarffcommands.txt`` and ``mnistffcommands.txt`` contains the commands to run the experiments for the feedforward experiments in the main text. The ``lr`` and ``weight_decay`` hyper-parameters have been tuned separately for each experiment, as described in the main text.

## Plotting results from paper

``plot_mnist.py`` and ``plot_cifar.py`` create the plots that result from the experiments.

All the subdirectories will be created in the current directory. These can be changed using the ``exp_root`` argument for ``{cifar, mnist}_launch.py`` and ``EXP_ROOT`` global parameter in ``plot_{cifar, mnist}.py``.

## Datasets

Data will be downloaded in the ``data`` subdirectory in the current directory, unless the ``data_root`` argument is provided to ``{cifar, mnist}_launch.py``.
