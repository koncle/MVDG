import copy
import errno
import functools
import os
import signal
import threading

import numpy as np
import torch
import torch.nn as nn
from torch._utils import ExceptionWrapper

from framework.loss_and_acc import get_loss_and_acc
from utils.tensor_utils import to, zero_and_update, Timer


def put_theta(model, theta):
    if theta is None:
        return model

    def k_param_fn(tmp_model, name=None):
        if len(theta) == 0:
            return

        if len(tmp_model._modules) != 0:
            for (k, v) in tmp_model._modules.items():
                if name == '':
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name + '.' + k))

        # WARN : running_mean, 和 running_var 不是 parameter，所以在 new 中不会被更新
        for (k, v) in tmp_model._parameters.items():
            if isinstance(v, torch.Tensor) and str(name + '.' + k) in theta.keys():
                tmp_model._parameters[k] = theta[str(name + '.' + k)]
            # else:
            #     print(name+'.'+k)
            # theta.pop(str(name + '.' + k))

        for (k, v) in tmp_model._buffers.items():
            if isinstance(v, torch.Tensor) and str(name + '.' + k) in theta.keys():
                tmp_model._buffers[k] = theta[str(name + '.' + k)]
            # else:
            #     print(k)
            # theta.pop(str(name + '.' + k))

    k_param_fn(model, name='')
    return model


def get_parameters(model):
    # note : you can direct manipulate these data reference which is related to the original models
    parameters = dict(model.named_parameters())
    states = dict(model.named_buffers())
    return parameters, states


def put_parameters(model, param, state):
    model = put_theta(model, param)
    model = put_theta(model, state)
    return model


def update_parameters(loss, names_weights_dict, lr, use_second_order, retain_graph=True, grads=None, ignore_keys=None):
    def contains(key, target_keys):
        if isinstance(target_keys, (tuple, list)):
            for k in target_keys:
                if k in key:
                    return True
        else:
            return key in target_keys

    new_dict = {}
    for name, p in names_weights_dict.items():
        if p.requires_grad:
            new_dict[name] = p
        # else:
        #     print(name)
    names_weights_dict = new_dict

    if grads is None:
        grads = torch.autograd.grad(loss, names_weights_dict.values(), create_graph=use_second_order, retain_graph=retain_graph, allow_unused=True)
    names_grads_wrt_params_dict = dict(zip(names_weights_dict.keys(), grads))
    updated_names_weights_dict = dict()

    for key in names_grads_wrt_params_dict.keys():
        if names_grads_wrt_params_dict[key] is None:
            continue  # keep the original state unchanged

        if ignore_keys is not None and contains(key, ignore_keys):
            # print(f'ignore {key}' )
            continue

        updated_names_weights_dict[key] = names_weights_dict[key] - lr * names_grads_wrt_params_dict[key]
    return updated_names_weights_dict


def cat_meta_data(data_list):
    new_data = {}
    for k in data_list[0].keys():
        l = []
        for data in data_list:
            l.append(data[k])
        new_data[k] = torch.cat(l, 0)
    return new_data


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


# @timeout(3)
def get_image_and_label(loaders, idx_list, device):
    if not isinstance(idx_list, (list, tuple)):
        idx_list = [idx_list]

    data_lists = []
    for i in idx_list:
        # with Timer('load data from domain {}'.format(i), thresh=0.5):
        data = loaders[i].next()
        data = to(data, device)  # , non_blocking=True)
        # data = loaders[i].next()
        data_lists.append(data)
    return cat_meta_data(data_lists)


def split_image_and_label(data, size, loo=False):
    n_domains = list(data.values())[0].shape[0] // size
    idx_sequence = list(np.random.permutation(n_domains))
    if loo:
        n_domains = 2
    res = [{} for _ in range(n_domains)]

    for k, v in data.items():
        split_data = torch.split(v, size)
        if loo:  # meta_train, meta_test
            res[0][k] = torch.cat([split_data[_] for _ in idx_sequence[:2]])
            # res[1][k] = split_data[idx_sequence[-1]]
            res[1][k] = torch.cat([split_data[_] for _ in idx_sequence[2:]])
        else:
            for i, d in enumerate(split_data):
                res[i][k] = d
    return res


def new_split_image_and_label(data, size, loo=False):
    n_domains = list(data.values())[0].shape[0] // size
    if loo:
        n_domains = 2
    res = [{} for _ in range(n_domains)]

    for k, v in data.items():
        split_data = torch.split(v, size)
        if loo:  # meta_train, meta_test
            res[0][k] = torch.cat(split_data[:2])
            res[1][k] = torch.cat(split_data[2:])
        else:
            for i, d in enumerate(split_data):
                res[i][k] = d
    return res


def split_data(train_data, device, leave_one_out=False):
    domains = len(train_data)
    indices = list(np.random.permutation(domains))

    if leave_one_out:
        indices = indices[:-1], indices[-1:]  # random_split to meta-train and meta-test
    else:
        indices = [[indices[i]] for i in range(domains)]

    train_inputs_lists = []
    for idx in indices:  #
        inputs = get_image_and_label(train_data, idx, device)
        train_inputs_lists.append(inputs)
    return train_inputs_lists


def init_network(meta_model, meta_lr, previous_opt=None, momentum=0.9, Adam=False, beta1=0.9, beta2=0.999, device=None):
    fast_model = copy.deepcopy(meta_model).train()
    if device is not None:
        fast_model.to(device)
    if Adam:
        fast_opts = torch.optim.Adam(fast_model.parameters(), lr=meta_lr, betas=(beta1, beta2), weight_decay=5e-4)
    else:
        fast_opts = torch.optim.SGD(fast_model.parameters(), lr=meta_lr, weight_decay=5e-4, momentum=momentum)

    if previous_opt is not None:
        fast_opts.load_state_dict(previous_opt.state_dict())
    return fast_model, fast_opts


def update_meta_model(meta_model, fast_param_list, optimizers, meta_lr=1):
    meta_params, meta_states = get_parameters(meta_model)

    optimizers.zero_grad()

    # update grad
    for k in meta_params.keys():
        new_v, old_v = 0, meta_params[k]
        for m in fast_param_list:
            new_v += m[0][k]
        new_v = new_v / len(fast_param_list)
        meta_params[k].grad = ((old_v - new_v) / meta_lr).data

    # update with original optimizers
    optimizers.step()

    # for k in meta_states.keys():
    #     new_v, old_v = 0, meta_states[k]
    #     for m in fast_param_list:
    #         new_v += m[1][k]
    #         break
    #     #new_v = new_v / len(fast_param_list)
    #     meta_states[k].data = new_v.data


class AveragedModel(nn.Module):
    def __init__(self, start_epoch=0, device=None, lam=None, avg_fn=None):
        super(AveragedModel, self).__init__()
        self.device, self.start_epoch = device, start_epoch
        self.module = None
        self.lam = lam
        self.register_buffer('n_averaged', torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, lamd):
                return lamd * averaged_model_parameter + (1 - lamd) * model_parameter
        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.module.step(*args, **kwargs)

    def init_model(self, model, epoch):
        if self.module is None:
            self.module = copy.deepcopy(model)
            if self.device is not None:
                self.module = self.module.to(self.device)

    def update_parameters(self, model, epoch):
        if epoch < self.start_epoch:
            return

        if self.module is None:
            self.module = copy.deepcopy(model)
            if self.device is not None:
                self.module = self.module.to(self.device)
            return

        if self.lam is None:
            lam = self.n_averaged.to(self.device) / (self.n_averaged.to(self.device) + 1)
        else:
            lam = self.lam
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, lam))
        self.n_averaged += 1

    def update_bn(self, loader, epoch, iters=None, model=None, meta=False):
        model = self.module if model is None else model
        if epoch < self.start_epoch:
            return
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            return

        was_training = model.training
        model.train()
        for module in momenta.keys():
            module.momentum = None
            module.num_batches_tracked *= 0

        if meta:
            with torch.no_grad():
                inner_loops = len(loader) if iters is None else iters
                for i in range(inner_loops):
                    data_list = get_image_and_label(loader, [0, 1, 2], device=self.device)
                    model.step(**data_list)
        else:
            with torch.no_grad():
                inner_loops = len(loader) if iters is None else iters
                loader = iter(loader)
                for i in range(inner_loops):
                    data_list = to(next(loader), self.device)
                    model.step(**data_list)

        for bn_module in momenta.keys():
            bn_module.momentum = momenta[bn_module]
        model.train(was_training)


class ParallelUpdate(nn.Module):
    def  __init__(self, module, meta_opt, meta_lr, devices, paths, tasks):
        super(ParallelUpdate, self).__init__()
        self.paths, self.tasks = paths, tasks
        self.module = module
        self.meta_opt = meta_opt
        self.meta_lr = meta_lr
        self.devices = devices
        self.fast_opts = [None] * len(devices)
        self.opts = [None] * len(devices)
        self.models = [None] * len(devices)

    def forward(self, data):
        self.models, self.opts = self.replicate(self.fast_opts)
        self.parallel_apply(self.models, self.opts, data, self.devices)
        fast_params = self.gather(self.models)
        update_meta_model(self.module, fast_params, self.meta_opt, meta_lr=1)

    def replicate(self, init_fast_opts):
        models, opts = [], []
        for i, device in enumerate(self.devices):
            fast_model, fast_opt = init_network(self.module, self.meta_lr, init_fast_opts[i], Adam=True, beta1=0.9, beta2=0.999, device=device)
            models.append(fast_model), opts.append(fast_opt)
        return models, opts

    def gather(self, models):
        return [get_parameters(m.to(self.devices[0])) for m in models]

    def parallel_apply(self, models, opts, data_list, devices):
        results = {}
        lock = threading.Lock()

        def _worker(i, model, opt, data, device):
            try:
                with torch.cuda.device(device):
                    # with lock
                    for j in range(self.tasks):
                        with Timer(name=f'fetch data-{i}: ', thresh=1):
                            for _ in range(1):
                                batch_data = to(next(data), device)
                                # try:
                                #
                                #     break
                                # except Exception as e:
                                #     print(f'Get data Error-{i} : Empty data')
                                #     print(traceback.format_exc())

                        with Timer(name=f'path-learning-{i}', thresh=5):
                            meta_train_data, meta_test_data = split_image_and_label(batch_data, size=64, loo=True)
                            zero_and_update(opt, get_loss_and_acc(model.step(**meta_train_data)))
                            zero_and_update(opt, get_loss_and_acc(model.step(**meta_test_data)))
            except Exception:
                with lock:
                    results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

        threads = [threading.Thread(target=_worker, args=(i, module, opt, data, device))
                   for i, (module, opt, data, device) in enumerate(zip(models, opts, data_list, devices))]

        [t.start() for t in threads]
        [t.join() for t in threads]

        outputs = []
        for i in range(len(models)):
            if i in results.keys():
                output = results[i]
                if isinstance(output, ExceptionWrapper):
                    output.reraise()
                outputs.append(output)
        return outputs
