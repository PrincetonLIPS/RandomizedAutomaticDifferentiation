# Running RNN code

## Requirements

See requirements.txt file for full pip requirements.\
Major requirements: torch, torchvision, tensorboard, matplotlib

NOTE: Only verified compatibility with torch version 1.3.1 + torchvision version 0.4.2\
torch version 1.5.0 is confirmed NOT to work.

## Reproducing results in paper

To reproduce the RNN results in the paper, run the following commands for each type of experiment. Replace `{exp_dir}` 
with the path to the directory where results will be stored and ``{data_dir}`` with the path to which the MNIST data will 
be downloaded and saved or already exists. Replace ``{exp_name}`` with the name of the experiment and ``{seed}`` with the
given random seeds below.
   
##### Baseline #####

Seeds are ``1, 2, 28222136``

``python rnn_mnist_launch.py --exp_name=irnn_baseline --seed={seed} --batch_size=150 --dataset=mnist --lr=1e-4 --keep_frac=1.0 --hidden_size=100 --weight_decay=0. --clip=1. --save_inter=10 --data_root={data_dir} --exp_root={exp_dir} --augment=False --use_writer=False --with_replace=True --simple=True --max_iterations=200000 --simple_test_eval_frequency=400``

##### Small Batch #####

Seeds are ``1, 2, 3``

``python rnn_mnist_launch.py --exp_name=irnn_small_batch --seed={seed} --batch_size=21 --dataset=mnist --lr=1e-4 --keep_frac=1.0 --hidden_size=100 --weight_decay=0. --clip=1. --save_inter=10 --data_root={data_dir} --exp_root={exp_dir} --augment=False --use_writer=False --with_replace=True --simple=True --max_iterations=200000 --simple_test_eval_frequency=400``

##### Sparse #####

Seeds are ``1, 2, 3``

``python rnn_mnist_launch.py --exp_name=rand_irnn_sparse --seed={seed} --batch_size=150 --dataset=mnist --lr=1e-4 --keep_frac=0.1 --hidden_size=100 --weight_decay=0. --clip=1. --save_inter=10 --data_root={data_dir} --exp_root={exp_dir} --augment=False --use_writer=False --with_replace=True --simple=True --max_iterations=200000 --simple_test_eval_frequency=400 --sparse=True --full_random=False``

##### Full Sparse #####

Seeds are ``1, 2, 3``

``python rnn_mnist_launch.py --exp_name=rand_irnn_sparse_full --seed={seed} --batch_size=150 --dataset=mnist --lr=1e-4 --keep_frac=0.1 --hidden_size=100 --weight_decay=0. --clip=1. --save_inter=10 --data_root={data_dir} --exp_root={exp_dir} --augment=False --use_writer=False --with_replace=True --simple=True --max_iterations=200000 --simple_test_eval_frequency=400 --sparse=True --full_random=True``

##### RP #####

Seeds are ``1, 2, 20268186``

``python rnn_mnist_launch.py --exp_name=rand_irnn_rp --seed={seed} --batch_size=150 --dataset=mnist --lr=1e-4 --keep_frac=0.1 --hidden_size=100 --weight_decay=0. --clip=1. --save_inter=10 --data_root={data_dir} --exp_root={exp_dir} --augment=False --use_writer=False --with_replace=True --simple=True --max_iterations=200000 --simple_test_eval_frequency=400 --sparse=False --full_random=False``

##### Full RP #####

Seeds are ``1, 2, 3``

``python rnn_mnist_launch.py --exp_name=rand_irnn_rp_full --seed={seed} --batch_size=150 --dataset=mnist --lr=1e-4 --keep_frac=0.1 --hidden_size=100 --weight_decay=0. --clip=1. --save_inter=10 --data_root={data_dir} --exp_root={exp_dir} --augment=False --use_writer=False --with_replace=True --simple=True --max_iterations=200000 --simple_test_eval_frequency=400 --sparse=False --full_random=True``


## Plotting results in paper

``rnn_plottingscripts.py`` creates the plot from the main text, and ``rnn_plottingscripts_appendix.py`` creates the plot for the appendix. The ``EXP_BASE_DIR`` variable in the plotting scripts should be changed to the chosen ``{exp_dir}`` above, and ``exp_folders`` should contain the chosen ``--exp_name``s.
