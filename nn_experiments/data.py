import torch
from torchvision import datasets, transforms


class ApplyTransform(torch.utils.data.Dataset):
    """
    Source: https://stackoverflow.com/a/56587747 -- Accessed: 04/05/2020

    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        # yes, you don't need these 2 lines below :(
        if transform is None and target_transform is None:
            print("Am I a joke to you? :)")

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)


def get_dataset(dataset, batch_size, test_batch_size=100, with_replace=False,
                num_workers=2, data_root='./data', validation=False, augment=True, bootstrap_loader=False, bootstrap_train=False, **kwargs):
    if dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset_cls = datasets.MNIST
        num_classes = 10
        input_size = (1, 28, 28)
    elif dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset_cls = datasets.CIFAR10
        num_classes = 10
        input_size = (3, 32, 32)
    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset_cls = datasets.CIFAR100
        num_classes = 100
        input_size = (3, 32, 32)
    else:
        raise NotImplementedError('Unsupported Dataset {}'.format(dataset))

    if validation:
        trainset_pre = dataset_cls(root=data_root, train=True, download=True)
        total_len = len(trainset_pre)
        val_set_size = 5000
        with torch.random.fork_rng():
            torch.random.manual_seed(17) # So that validation dataset is deterministic
            trainset, valset = \
                    torch.utils.data.random_split(trainset_pre, [total_len - val_set_size, val_set_size])
        if bootstrap_train:
            bootstrap_samples = torch.randint(low=0, high=len(trainset), size=(len(trainset),)).tolist()
            print('Bootstrapping training set with samples: {}'.format(bootstrap_samples[:100]))
            trainset = torch.utils.data.Subset(trainset, bootstrap_samples)

        if augment:
            trainset = ApplyTransform(trainset, train_transform)
        else:
            trainset = ApplyTransform(trainset, test_transform)

        train_sampler = torch.utils.data.RandomSampler(trainset, replacement=with_replace)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=with_replace,
                                                   sampler=train_sampler, num_workers=num_workers)

        valset = ApplyTransform(valset, test_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=test_batch_size,
                                                 shuffle=False, num_workers=num_workers)
        train_test_loader = torch.utils.data.DataLoader(trainset, batch_size=test_batch_size,
                                                        shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, train_test_loader, num_classes
    else:
        trainset = dataset_cls(root=data_root, train=True, download=True)
        if bootstrap_train:
            bootstrap_samples = torch.randint(low=0, high=len(trainset), size=(len(trainset),)).tolist()
            print('Bootstrapping training set with samples: {}'.format(bootstrap_samples[:100]))
            trainset = torch.utils.data.Subset(trainset, bootstrap_samples)

        if augment:
            trainset = ApplyTransform(trainset, train_transform)
        else:
            trainset = ApplyTransform(trainset, test_transform)

        train_sampler = torch.utils.data.RandomSampler(trainset, replacement=with_replace)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=with_replace,
                                                   sampler=train_sampler, num_workers=num_workers)

        testset = dataset_cls(root=data_root, train=False, download=True, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                  shuffle=False, num_workers=num_workers)
        train_test_loader = torch.utils.data.DataLoader(trainset, batch_size=test_batch_size,
                                                        shuffle=False, num_workers=num_workers)

        return train_loader, test_loader, train_test_loader, num_classes