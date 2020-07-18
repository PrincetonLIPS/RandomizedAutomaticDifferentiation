import os
import sys
import pickle
import random
import signal
import shutil
import argparse
import numpy as np
import collections

import data as rpdata
import models as rpmodels
import utils as rputils

from train_and_eval import run_model

import torch
import torch.utils.tensorboard as tb


'''
To override these parameters, specify in command line, such as:
python mnist_launch.py --batch_size=100

It is important to precede the argument with '--' and include an '=' sign.
Do not use quotes around the value.
'''
args_template = rputils.ParameterMap(
    # experiment name. prepended to pickle name
    exp_name='',

    # input batch size for training (default: 64)
    batch_size=128,

    # accepted dataset
    dataset='cifar10',

    # input batch size for testing (default: 1000)
    test_batch_size=1000,

    # number of epochs to train (default: 14)
    epochs=200,

    # learning rate (default: 1.0)
    lr=0.1,

    # Learning rate step gamma (default: 0.7)
    gamma=0.2,

    # disables CUDA training
    no_cuda=False,

    # random seed. do not set seed if 0.
    seed=0,

    # how many batches to wait before logging training status
    log_interval=10,

    # the fraction of activations to reduce to with RAD
    keep_frac=1.0,

    # hidden layer size for MNISTFCNet
    hidden_size=300,

    # number of epochs after which to drop the lr
    lr_drop_step=1,

    # l2 weight decay on the parameters
    weight_decay=5e-4,

    # For Saving the current Model
    save_model=True,

    # CIFAR lr schedule
    training_schedule='cifar',

    # Use adam optimizer in cifar
    cifar_adam=False,

    # If true, runs baseline without RP
    rp_layer='rpconv',

    # Number of epochs to test on the train set.
    train_test_interval=5,

    # Data Root
    data_root='./data',

    # Experiment Root
    exp_root='',

    # If true, randomly splits the training set into train/val (validation 5000). Ignores test set.
    validation=False,

    # Whether to sample with replacement or not while training. True gives real SGD.
    with_replace=False,

    # Whether to use augmentation in training dataset.
    augment=True,

    # Additive noise to add.
    rand_noise=0.0,

    # Wide-ResNet width multiplier.
    width_multiplier=1,

    # If experiment exists, overwrite
    override=False,

    # If true, generate random matrix independent across batch
    full_random=False,

    # If > 0, save this many intermediate checkpoints
    save_inter=0,

    # Whether to do simple iteration based training instead of epoch based.
    simple=False,

    # Following are only used when simple is True.
    max_iterations=-1,
    simple_log_frequency=-1,
    simple_test_eval_frequency=-1,
    simple_test_eval_per_train_test=-1,
    simple_scheduler_step_frequency=-1,
    simple_model_checkpoint_frequency=-1,

    # If true, samples training set with replacement.
    bootstrap_train=False,

    # If false, uses random projections. If true, uses sampling.
    sparse=False,

    # If true, also uses RAD on ReLU layers.
    rand_relu=False,
)


def main(additional_args):
    args = args_template.clone()
    rputils.override_arguments(args, additional_args)

    # If simple is set, default to these arguments.
    # Note that we override again at the end, so specified
    # arguments take precedence over defaults.
    if args.simple:
        args.max_iterations = 100000
        args.simple_log_frequency = 10
        args.simple_test_eval_frequency = 400
        args.simple_test_eval_per_train_test = 10
        args.simple_scheduler_step_frequency = 10000
        args.simple_model_checkpoint_frequency = 10000
        args.save_inter = 1

        args.batch_size = 150
        args.gamma = 0.6
        args.training_schedule = 'epoch_step'
        args.cifar_adam = True
        args.lr = 0.002
        args.with_replace = True
        args.augment = False
        args.validation = False
        args.lr_drop_step = 1
        rputils.override_arguments(args, additional_args)

    if not os.path.exists(args.exp_root):
        print('Creating experiment root directory {}'.format(args.exp_root))
        os.mkdir(args.exp_root)
    if not args.exp_name:
        args.exp_name = 'exp{}'.format(random.randint(100000, 999999))

    if args.seed == 0:
        args.seed = random.randint(10000000, 99999999)

    args.exp_dir = os.path.join(args.exp_root, args.exp_name)
    os.environ['LAST_EXPERIMENT_DIR'] = args.exp_dir
    if args.override and os.path.exists(args.exp_dir):
        print("Overriding existing directory.")
        shutil.rmtree(args.exp_dir)
    assert not os.path.exists(args.exp_dir)
    print("Creating experiment with name {} in {}".format(args.exp_name, args.exp_dir))
    os.mkdir(args.exp_dir)
    with open(os.path.join(args.exp_dir, 'experiment_args.txt'), 'w') as f:
        f.write(str(args))

    if args.save_inter > 0:
        args.inter_dir = os.path.join(args.exp_dir, 'intermediate_checkpoints')
        if not os.path.exists(args.inter_dir):
            print('Creating directory for intermediate checkpoints.')
            os.mkdir(args.inter_dir)

    args.pickle_dir = os.path.join(args.exp_dir, 'pickles')
    if not os.path.exists(args.pickle_dir):
        print('Creating pickle directory in experiment directory.')
        os.mkdir(args.pickle_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('Seed is {}'.format(args.seed))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    rp_args = {}
    rp_args['rp_layer'] = args.rp_layer
    rp_args['keep_frac'] = args.keep_frac
    rp_args['rand_noise'] = args.rand_noise
    rp_args['width_multiplier'] = args.width_multiplier
    rp_args['full_random'] = args.full_random
    rp_args['sparse'] = args.sparse

    models = [
        (rpmodels.CIFARConvNet(rp_args=rp_args, rand_relu=args.rand_relu), args.exp_name + "cifarconvnet8", args.exp_name + "cifarconvnet8"),
    ]

    # Check if correct dataset is used for each model.
    for model, _, _ in models:
        if model.kCompatibleDataset != args.dataset:
            raise NotImplementedError(
                'Unsupported dataset {} with model {}'.format(args.dataset, model.__class__.__name__)
            )

    for model, pickle_string, model_string in models:
        run_model(model, args, device, None, pickle_string, model_string)


if __name__ == '__main__':
    main(sys.argv[1:])

