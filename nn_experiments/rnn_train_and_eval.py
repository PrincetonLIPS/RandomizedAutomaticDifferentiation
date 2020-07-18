import os
import pickle
import time

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import data as rpdata
from train_and_eval import test, test_list

def rnn_simple_train(args, model, device, train_loader, optimizer, test_loader, train_test_loader, iteration=0):
    all_checkpoints = []
    before_epoch = time.time()

    params = list(model.parameters())
    while iteration < args.max_iterations:
        for data, target in train_loader:
            iteration += 1
            if iteration > args.max_iterations:
                break

            random_seed = torch.randint(low=10000000000, high=99999999999, size=(1,))
            model.train()
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)

            with torch.random.fork_rng():
                torch.random.manual_seed(random_seed)
                output = model(data)

            loss = F.nll_loss(output, target)
            loss.backward()

            if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)

            optimizer.step()

            if iteration % args.simple_log_frequency == 0:
                # Logging every 10 iterations.
                print('Train Iteration: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iteration, args.max_iterations,
                    100. * iteration / args.max_iterations, loss.item()))

            if iteration % args.simple_test_eval_frequency == 0:
                # Evaluate every 300 iterations.
                after_epoch = time.time()
                print('{} Iteration time: {}'.format(args.simple_test_eval_frequency, after_epoch - before_epoch))
                train_ckpt = {}
                train_ckpt['loss'] = loss.item()
                train_ckpt['time'] = after_epoch - before_epoch
                train_ckpt['iteration'] = iteration

                start_testing_time = time.time()
                test_ckpt = test(args, model, device, test_loader)
                if (iteration // args.simple_test_eval_frequency) % args.simple_test_eval_per_train_test == 0:
                    # Every few epochs calculate total train loss and accuracy.
                    train_test_ckpt = test(args, model, device, train_test_loader, split='Train')
                    all_checkpoints.append((iteration, {'train': train_ckpt, 'test': test_ckpt, 'train_test': train_test_ckpt}))
                else:
                    all_checkpoints.append((iteration, {'train': train_ckpt, 'test': test_ckpt}))
                print("Test time at iteration {}: {}".format(iteration, time.time() - start_testing_time))

                before_epoch = time.time()

            if iteration % args.simple_model_checkpoint_frequency == 0:
                if 'inter_dir' not in args:
                    print("ERROR: Intermediate directory not created. This is a mistake.")
                else:
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, os.path.join(args.inter_dir, 'iteration_{0:09}.pt'.format(iteration)))

                    with open(os.path.join(args.pickle_dir, 'ckpts_iteration_{0:09}.pkl'.format(iteration)), 'wb') as f:
                        pickle.dump(all_checkpoints, f)

    test_ckpt = test(args, model, device, test_loader)
    train_test_ckpt = test(args, model, device, train_test_loader, split='Train')
    all_checkpoints.append(('final', {'test': test_ckpt, 'train_test': train_test_ckpt}))

    return all_checkpoints

def run_rnn_model(model, args, device, writer, pickle_string, model_string, num_workers=2, iteration=0,
                  optimizer_state_dict=None):
    train_loader, test_loader, train_test_loader, num_classes = \
        rpdata.get_dataset(args.dataset, batch_size=args.batch_size, test_batch_size=args.test_batch_size,
                           validation=args.validation, with_replace=args.with_replace,
                           data_root=args.data_root, augment=args.augment,
                           bootstrap_train=args.bootstrap_train, num_workers=num_workers)
    # Calculate hash of test dataset to make sure test set is fixed.
    with torch.autograd.grad_mode.no_grad():
        data_sum = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):
            data_sum += torch.sum(data)
        print('Testing dataset has a sum of: {}'.format(data_sum))
        if args.dataset == 'cifar10':
            if args.validation:
                assert data_sum == 116150.875
            else:
                assert data_sum == 492130.3125
        print('Make sure this is different: {}'.format(torch.randn(3)))
        print('Validation is {}'.format(args.validation))

    data, _ = next(iter(train_loader))
    if writer:
        grid = torchvision.utils.make_grid(data)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, data)

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optimizer_state_dict:
        print("Reloading optimizer state.")
        optimizer.load_state_dict(optimizer_state_dict)

    all_checkpoints = rnn_simple_train(args, model, device, train_loader, optimizer, test_loader, train_test_loader, iteration=iteration)

    with open(os.path.join(args.pickle_dir, '{}.pkl'.format(pickle_string)), 'wb') as f:
        pickle.dump(all_checkpoints, f)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "{}.pt".format(model_string)))

    if writer:
        writer.close()
