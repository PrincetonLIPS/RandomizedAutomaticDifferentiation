import os
import gc
import sys
import time
import pickle
import argparse
import numpy as np

import data as rpdata
import models as rpmodels

import torch
import torchvision
import torch.optim as optim
import torch.utils.tensorboard as tb
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F


def simple_train(args, model, device, train_loader, optimizer, scheduler, test_loader, train_test_loader):
    all_checkpoints = []
    before_epoch = time.time()
    iteration = 0

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

                test_ckpt = test(args, model, device, test_loader)
                if (iteration // args.simple_test_eval_frequency) % args.simple_test_eval_per_train_test == 0:
                    # Every few epochs calculate total train loss and accuracy.
                    train_test_ckpt = test(args, model, device, train_test_loader, split='Train')
                    all_checkpoints.append((iteration, {'train': train_ckpt, 'test': test_ckpt, 'train_test': train_test_ckpt}))
                else:
                    all_checkpoints.append((iteration, {'train': train_ckpt, 'test': test_ckpt}))
                before_epoch = time.time()

            if iteration % args.simple_scheduler_step_frequency == 0:
                scheduler.step()
                print('Learning rate is now decreased by {}'.format(args.gamma))

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

    test_ckpt = test(args, model, device, test_loader)
    train_test_ckpt = test(args, model, device, train_test_loader, split='Train')
    all_checkpoints.append(('final', {'test': test_ckpt, 'train_test': train_test_ckpt}))

    return all_checkpoints


def test(args, model, device, test_loader, writer=None, split='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    before_test = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    after_test = time.time()

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        split, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    checkpoint = {}
    checkpoint['loss'] = test_loss
    checkpoint['accuracy'] = 100. * correct / len(test_loader.dataset)
    checkpoint['time'] = after_test - before_test

    return checkpoint


def run_model(model, args, device, writer, pickle_string, model_string):
    train_loader, test_loader, train_test_loader, num_classes = \
        rpdata.get_dataset(args.dataset, batch_size=args.batch_size, test_batch_size=args.test_batch_size,
                           validation=args.validation, with_replace=args.with_replace,
                           data_root=args.data_root, augment=args.augment,
                           bootstrap_train=args.bootstrap_train)

    data, _ = next(iter(train_loader))
    if writer:
        grid = torchvision.utils.make_grid(data)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, data)

    model = model.to(device)

    if 'cifar_adam' in args and args.cifar_adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(
                model.parameters(), 
                momentum=0.9, lr=args.lr,
                weight_decay=args.weight_decay)

    if 'training_schedule' not in args or args.training_schedule == 'cifar':
        print('Using CIFAR schedule.')
        scheduler = lrs.MultiStepLR(optimizer, milestones=[int(args.epochs * 0.3),
                                                           int(args.epochs * 0.6),
                                                           int(args.epochs * 0.8)], gamma=0.2)
    elif args.training_schedule == 'extended':
        print('Using extended CIFAR schedule.')
        scheduler = lrs.MultiStepLR(optimizer, milestones=[int(args.epochs * 0.1),
                                                           int(args.epochs * 0.2),
                                                           int(args.epochs * 0.3),
                                                           int(args.epochs * 0.4),
                                                           int(args.epochs * 0.5),
                                                           int(args.epochs * 0.6),
                                                           int(args.epochs * 0.7),
                                                           int(args.epochs * 0.8),
                                                           int(args.epochs * 0.9)], gamma=0.4)
    elif args.training_schedule == 'epoch_step':
        scheduler = lrs.StepLR(optimizer, step_size=args.lr_drop_step, gamma=args.gamma)
    else:
        raise NotImplementedError('Invalid training schedule.')

    all_checkpoints = simple_train(args, model, device, train_loader, optimizer, scheduler, test_loader, train_test_loader)

    with open(os.path.join(args.pickle_dir,'{}.pkl'.format(pickle_string)), 'wb') as f:
        pickle.dump(all_checkpoints, f)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.exp_dir, "{}.pt".format(model_string)))
    if writer:
        writer.close()
