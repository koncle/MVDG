import copy
import os
import threading
from pathlib import Path

import torch
import functools
import numpy as np
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.datasets.folder import default_loader as img_loader

from dataloader.augmentations import Aug
from framework.registry import Datasets
from utils.tensor_utils import Timer

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class MetaDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        # kwargs.update({'timeout': 60})
        super(MetaDataLoader, self).__init__(*args, **kwargs)
        self.iter = None
        # self.lock = threading.Lock()

    def __next__(self):
        return self.next()

    def next(self):
        # if self.lock.acquire(timeout=10):
        # with self.lock:
        if self.iter is None:
            self.iter = iter(self)
        try:
            ret = next(self.iter)
        except:
            self.iter = iter(self)
            ret = next(self.iter)
            # self.lock.release()
        return ret
        # else:
        #     raise TimeoutError("DeadLock happens ")


DataSource = None
ToMemory = False


class DGDataset(Dataset):
    def __init__(self, samples, split, args, extra_aug_func_dict=None):
        self.args = args
        self.samples = samples
        self.img_size = self.args.img_size
        self.min_scale = self.args.min_scale
        self.pre_transform, self.transform = self.set_transform(split)
        self.extra_aug_func_dict = extra_aug_func_dict
        self.augmentations =  Aug(args)

    def __len__(self):
        return len(self.samples)

    def set_transform(self, split):
        transform = [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0, 0, 0], [1,1,1])
        ]

        if split == 'train' and not self.args.do_not_transform:
            pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(self.min_scale, 1.0)),
                transforms.RandomHorizontalFlip()]
            )
            if self.args.color_jitter:
                transform.insert(0, transforms.ColorJitter(.4, .4, .4, .4))
        else:
            pre_transform = transforms.Compose([
                # transforms.RandomResizedCrop(self.img_size, scale=(0.7, 1)),
                transforms.Resize((self.img_size, self.img_size))
            ])

        return pre_transform, transforms.Compose(transform)

    def __getitem__(self, index):
        path, target, domain = self.samples[index]
        target, domain = int(target), int(domain)

        ret = {}
        if DataSource is not None:
            origin_image = DataSource[path]
        else:
            origin_image = img_loader(path)
        # origin_image = self.pre_transform(origin_image)

        # image_o = self.transform(origin_image)
        image_o = self.transform(self.pre_transform(origin_image))
        ret.update({'x': image_o, 'label': target})

        if self.args.domain_label:
            ret.update({'domain_label': torch.tensor(domain).long()})

        # if self.args.aug:
        #     ret.update({'aug_x': self.augmentations['randaug'](origin_image, self.args.aug_N, self.args.aug_M)})
        if self.args.MVP:
            ret.update({'aug_x': self.augmentations.test_aug(origin_image, self.args.MVP_bs)})

        if self.extra_aug_func_dict is not None:
            for key, func in self.extra_aug_func_dict.items():
                ret.update({key: func(origin_image)})

        if self.args.data_path:
            ret.update({'data_path': path})
        return ret


class BaseDatasetConfig(object):
    Name = 'Base'
    NumClasses = 0
    Domains = []
    SplitRatio = -1
    RelativePath = ''
    Classes = None
    ClassOffset = 0

    def __init__(self, args):
        self.args = args
        self.root_dir = os.path.join(args.data_root, self.RelativePath)
        self.source_domains, self.target_domains = self.split_train_test_domains(args)
        self.aug_funcs = None
        self.to_memory = ToMemory
        # Note : multiple dataloaders with pin_memory=True may cause serveral errors,
        # including read-write error, dataloader deadlock, unable to open shared memory error
        self.loader_args = {
            'pin_memory': False,
            'num_workers': self.args.workers,
        }

    def split_train_test_domains(self, args):
        if args.src[0] != -1:
            source_domain, target_domain = [], []
            for i in args.src:
                source_domain.append(self.Domains[i])
            for j in args.tgt:
                target_domain.append(self.Domains[j])
        else:
            source_domain = deepcopy(self.Domains)
            target_domain = [source_domain.pop(args.exp_num[0])]

        # print('Source domain: ', end=''), [print(domain, end=', ') for domain in source_domain]
        # print('Target domain: ', end=''), [print(domain) for domain in target_domain]
        return source_domain, target_domain

    def load_classes(self):
        class_text_path = Path(__file__).parent / 'text_lists' / self.Name / 'classes.txt'
        with open(str(class_text_path), 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        self.Classes = {i: c for i, c in enumerate(classes)}

    def load_text(self, domain, split, domain_idx):
        # text is in the same folder of this dataset
        filename = f'{domain}_{split}.txt'
        text_path = Path(__file__).parent / 'text_lists' / self.Name / filename
        # print(text_path, 'Loaded')
        samples = []
        class_nums = [0] * self.NumClasses
        with open(str(text_path), 'r') as f:
            for line in f.readlines():
                path, claz = line.split(' ')
                path = os.path.join(self.root_dir, path)
                class_nums[int(claz) + self.ClassOffset] += 1
                samples.append((path, int(claz) + self.ClassOffset, domain_idx))
        return samples

    def load_dataset(self, mode):
        assert mode in ['train', 'val', 'test']
        datasets = []
        domains = self.source_domains if mode != 'test' else self.target_domains
        all_samples = []
        for i, d in enumerate(domains):
            samples = self.load_text(d, mode, i)
            dataset = DGDataset(samples, mode, self.args, self.aug_funcs)
            datasets.append(dataset)
            all_samples.extend(samples)
            # print(f"{mode}: len({d})={len(samples)}", end=', ')
        # print()
        return datasets, all_samples

    def preload_images(self, samples):
        global DataSource
        if DataSource is None:
            DataSource = {}
            with Timer():
                for i, (path, _, _) in enumerate(samples):
                    # print('{}/{}'.format(i, len(samples)))
                    DataSource[path] = img_loader(path)
                print('Preloaded all images')

    def random_split(self, dataset: DGDataset):
        lengths, samples = len(dataset), np.array(dataset.samples)
        indices = torch.randperm(lengths).tolist()
        train_indices, val_indices = indices[:int(lengths * self.SplitRatio)], indices[int(lengths * self.SplitRatio):]
        return DGDataset(samples[train_indices], 'train', self.args, self.aug_funcs), DGDataset(samples[val_indices], 'val', self.args, self.aug_funcs)

    def get_datasets(self, aug_funcs=None):
        all_samples = []
        self.aug_funcs = aug_funcs
        # use official split or random split data
        if self.SplitRatio == -1:
            # use official split for train and val set
            train_datasets, train_samples = self.load_dataset('train')
            val_datasets, val_samples = self.load_dataset('val')
            all_samples.extend(train_samples), all_samples.extend(val_samples)
        else:
            # random split from the train set
            train_val_datasets, train_val_samples = self.load_dataset('train')
            all_samples.extend(train_val_samples)
            train_datasets, val_datasets = [], []
            for d in train_val_datasets:
                t_d, v_d = self.random_split(d)
                train_datasets.append(t_d), val_datasets.append(v_d)

        test_datasets, test_samples = self.load_dataset('test')
        all_samples.extend(test_samples)
        if self.to_memory:
            self.preload_images(all_samples)
        return train_datasets, val_datasets, test_datasets

    def analyze_datasets(self, datasets):
        classes = [0] * self.NumClasses
        for d in datasets:
            for _, claz, _ in d.samples:
                classes[int(claz)] += 1
        print(classes)

    def get_loaders(self, aug_funcs):
        datasets = self.get_datasets(aug_funcs)
        # [self.analyze_datasets(d) for d in datasets]
        train_datasets, val_datasets, test_datasets = [ConcatDataset(d) for d in datasets]

        bs = self.args.batch_size
        if self.args.loader == 'meta':
            train_loader = MetaDataLoader(train_datasets, batch_sampler=DomainSampler(train_datasets, bs, replace=self.args.replace, mvrml='mvrml' in self.args.train),
                                          **self.loader_args)
            val_loader = MetaDataLoader(val_datasets, batch_sampler=DomainSampler(val_datasets, bs, replace=self.args.replace, mvrml='mvrml' in self.args.train),
                                        **self.loader_args)
            test_loader = MetaDataLoader(test_datasets, batch_size=bs, shuffle=False, **self.loader_args)
        else:
            train_loader = MetaDataLoader(train_datasets, drop_last=True, batch_size=bs, shuffle=True, **self.loader_args)
            val_loader = MetaDataLoader(val_datasets, drop_last=False, batch_size=bs, shuffle=False, **self.loader_args)
            test_loader = MetaDataLoader(test_datasets, batch_size=bs, shuffle=False, **self.loader_args)
        loaders = [train_loader, val_loader, test_loader]
        return loaders


class DomainSampler(object):
    def __init__(self, concatedDataset, batch_size, replace=False, mvrml=False):
        assert isinstance(concatedDataset, ConcatDataset)
        self.domain_sizes = concatedDataset.cumulative_sizes
        self.domains = len(self.domain_sizes)
        self.batch_size = batch_size
        self.num_batches = self.domain_sizes[-1] // (batch_size * self.domains)
        self.domain_sizes = [0] + self.domain_sizes

        self.replace = replace  # if replace is set to True, samples are put back
        self.sample_temperature = 0  # equally sampled from all domains
        self.mvrml = mvrml
        # if self.mvrml:
        #     print("Training with mvrml!!!")

    def __iter__(self):
        domains = range(len(self.domain_sizes) - 1)

        real_batch_size = self.batch_size * self.domains
        domain_prob = (torch.rand(self.domains) * self.sample_temperature).softmax(0)
        batch_sizes = [int(p.item() * real_batch_size) for p in domain_prob]
        left_samples = real_batch_size - np.sum(batch_sizes)
        if left_samples > 0:
            batch_sizes[-1] += left_samples

        for iter_idx in range(self.num_batches):
            rand_domains = list(np.random.choice(domains, size=len(domains), replace=self.replace))
            # if self.mvrml:
            #     rand_domains += [rand_domains[-1]]
            sampled_idx = [np.random.choice(range(self.domain_sizes[idx], self.domain_sizes[idx + 1]), size=batch_sizes[idx], replace=False)
                           for idx in rand_domains]
            sampled_idx = np.concatenate(sampled_idx)
            yield sampled_idx

    def __len__(self):
        return self.num_batches


@Datasets.register("PACS")
class PACS(BaseDatasetConfig):
    # PACS follow <MLDG>, official split
    Name = 'PACS'
    NumClasses = 7
    SplitRatio = -1
    RelativePath = 'PACS/kfold'   # Relative path to the domain folders
    Domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    ClassOffset = -1  # text_lists start from 1 not 0
    Classes = {i: k for i, k in enumerate(['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'])}


@Datasets.register("OH")
@Datasets.register("OfficeHome")
class OfficeHome(BaseDatasetConfig):
    Name = 'OfficeHome'
    NumClasses = 65
    SplitRatio = 0.9
    RelativePath = 'OfficeHome'
    Domains = ['Art', 'Clipart', 'Product', 'RealWorld']


@Datasets.register("VLCS")
class VLCS(BaseDatasetConfig):
    # VLCS follow <Domain Generalization for Object Recognition with Multi-task Autoencoders>, split with 0.7 vs 0.3
    Name = 'VLCS'
    NumClasses = 5
    SplitRatio = -1
    RelativePath = 'VLCS'
    Domains = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
