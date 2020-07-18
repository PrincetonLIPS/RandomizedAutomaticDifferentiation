import os
import sys
import random
import shutil
from operator import itemgetter

import numpy as np

import rnn_models as rnn_rpmodels
import utils as rputils

from rnn_train_and_eval import run_rnn_model

import torch
import torch.utils.tensorboard as tb

'''
To override these parameters, specify in command line, such as:
python rnn_mnist_launch.py --batch_size=100

It is important to precede the argument with '--' and include an '=' sign.
Do not use quotes around the value.
'''
args_template = rputils.ParameterMap(
    # experiment name. prepended to pickle name
    exp_name='',

    # input batch size for training (default: 64)
    batch_size=128,

    # accepted dataset
    dataset='mnist',

    # input batch size for testing (default: 1000)
    test_batch_size=1000,

    # number of epochs to train (default: 14)
    epochs=200,

    # learning rate (default: 1.0)
    lr=0.1,

    # disables CUDA training
    no_cuda=False,

    # random seed. do not set seed if 0.
    seed=0,

    # how many batches to wait before logging training status
    log_interval=10,

    # keep_prob for the RandLinear layer
    keep_frac=1.0,

    # hidden layer size
    hidden_size=300,

    # l2 weight decay on the parameters
    weight_decay=5e-4,

    # For Saving the current Model
    save_model=True,

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

    # If experiment exists, overwrite
    override=False,

    # If true, generate random matrix independent across batch
    full_random=False,

    # If > 0, save this many intermediate checkpoints (Not used for simple training, only needed to create intermediates
    # directory)
    save_inter=0,

    # Whether to do simple iteration based training instead of epoch based.
    simple=False,

    # Following are only used when simple is True.
    max_iterations=-1,
    simple_log_frequency=-1,
    simple_test_eval_frequency=-1,
    simple_test_eval_per_train_test=-1,
    simple_model_checkpoint_frequency=-1,

    # If true, samples training set with replacement.
    bootstrap_train=False,

    sparse=False,

    # If true, use a TensorBoard writer
    use_writer=False,

    # Gradient norm clipping parameter
    clip=0.25,

    # No. of workers for data loader
    num_workers=2,

    # Whether to resume training from the most recent checkpoint (if it exists)
    resume=False,
)



def main(additional_args):
    #signal.signal(signal.SIGINT, receiveSignal)
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
        args.simple_model_checkpoint_frequency = 10000
        args.save_inter = 1

        args.batch_size = 150
        args.lr = 0.002
        args.with_replace = True
        args.augment = False
        args.validation = False
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
    if not args.resume or not os.path.exists(args.exp_dir):
        assert not os.path.exists(args.exp_dir)
        print("Creating experiment with name {} in {}".format(args.exp_name, args.exp_dir))
        os.mkdir(args.exp_dir)
        with open(os.path.join(args.exp_dir, 'experiment_args.txt'), 'w') as f:
            f.write(str(args))
    assert os.path.exists(args.exp_dir)


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

    # Tensorboard SummaryWriter
    writer = tb.SummaryWriter(log_dir=args.exp_dir) if args.use_writer else None

    rp_args = {}
    rp_args['rp_layer'] = args.rp_layer
    rp_args['keep_frac'] = args.keep_frac
    rp_args['rand_noise'] = args.rand_noise
    rp_args['full_random'] = args.full_random
    rp_args['sparse'] = args.sparse

    model = rnn_rpmodels.MNISTIRNN(hidden_size=args.hidden_size, rp_args=rp_args)
    pickle_string, model_string = args.exp_name + "mnistirnn", args.exp_name + "mnistirnn"

    optimizer_state_dict, iteration = None, 0
    if args.resume and args.save_inter > 0:
        # Get most recent checkpoint if it exists
        ckpt_filenames = os.listdir(args.inter_dir)
        if len(ckpt_filenames) > 0:
            most_recent_ckpt = sorted(ckpt_filenames, key=lambda f: int(f.split('_')[1].split('.')[0]))[-1]
            print("Using checkpoint {}".format(most_recent_ckpt))
            iteration, model_state_dict, optimizer_state_dict = \
                itemgetter('iteration', 'model_state_dict', 'optimizer_state_dict')(
                    torch.load(os.path.join(args.inter_dir, most_recent_ckpt), map_location=device)
                )
            print("Reloading model state.")
            model.load_state_dict(model_state_dict)


    # Check if correct dataset is used for model.
    if model.kCompatibleDataset != args.dataset:
        raise NotImplementedError(
            'Unsupported dataset {} with model {}'.format(args.dataset, model.__class__.__name__)
        )

    # Run training
    run_rnn_model(
        model, args, device, writer, pickle_string, model_string, num_workers=args.num_workers, iteration=iteration,
        optimizer_state_dict=optimizer_state_dict
    )


if __name__ == '__main__':
    main(sys.argv[1:])

