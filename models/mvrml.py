import traceback

from models.meta_util import *
from framework.registry import TrainFuncs
from utils.tensor_utils import AverageMeterDict, zero_and_update
from queue import Queue
from threading import Thread


class CudaDataLoader:
    """ Load data asynchronously from cpu to gpu """
    # https://www.zhihu.com/question/307282137/answer/1560137140
    def __init__(self, loader, id, device, queue_size=10):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader
        self.id = id

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0
        self.stop_flag = False
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()
        self.lock = threading.Lock()

        self.balanced = 0

    def stop(self):
        self.stop_flag = True

    def load_loop(self):
        # The loop that will load data into the queue in the background
        torch.cuda.set_device(self.device)
        while not self.stop_flag:
            for i, sample in enumerate(self.loader):
                if self.stop_flag:
                    break
                s = self.load_instance(sample)
                try:
                    self.queue.put(s, timeout=10)
                    if self.balanced < 0:
                        self.balanced += 1
                        # print("I'm putting data, why timeout?")
                except Exception as e:
                    self.balanced += 1
                    if self.balanced > 3:
                        # print(traceback.format_exc())
                        print('PPPPPut Timeout-{}, Queue is Full? : {}, Thread Terminated? : {}'.format(self.device, not self.queue.empty(), not self.worker.is_alive()))
                        print('If it is in Evaluation, the process may be stuck and the code need to be re-runned.')
                        raise Exception("Put data error")
        while not self.queue.empty():
            self.queue.get()
            self.queue.task_done()
        self.queue.join()
        print('stops')

    def load_instance(self, sample):
        """ 将batch数据从CPU加载到GPU中 """
        # with self.lock
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) == str:
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            print('Cuda : {}. Thread terminated'.format(self.id))
            raise StopIteration
        else:
            out = None
            for _ in range(4):
                try:
                    out = self.queue.get(timeout=10)
                    self.queue.task_done()
                    if self.balanced > 0:
                        self.balanced -= 1
                        # print("I'm {} getting data... Why Timeout?".format(self.device))
                    self.idx += 1
                    return out
                except Exception as e:
                    self.balanced -= 1
                    if self.balanced < -3:
                        # print(traceback.format_exc())
                        print('GGGGGet Timeout-{}, Queue is Empty? : {}, Thread Terminated? : {}'.format(self.device, self.queue.empty(), not self.worker.is_alive()))
                        print('If it is in Evaluation, the process may be stuck and the code need to be re-runned.')
                        raise Exception("Get data error")
            raise Exception("Get data error : {}. Don't know why...".format(out))

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def get_data(data, i, device, random=True):
    a = list(range(len(data)))
    if random:
        i = np.random.randint(len(data))
    a.pop(i)
    return get_image_and_label(data, a, device), get_image_and_label(data, [i, i], device)


@TrainFuncs.register('mvrml')
def MVRML(meta_model, train_data, meta_lr, epoch, args, engine, mode):
    assert args.loader != 'standard'
    device, meta_optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_lr = meta_optimizers.param_groups[0]['lr'] / args.meta_lr_weight

    trajectory, length = args.trajectory, args.length
    print('Meta lr : {}, loops : {}, path : {}, length : {}'.format(meta_lr, len(train_data), trajectory, length))

    iters = CudaDataLoader(train_data, id=1, device=device, queue_size=10)
    # iters = train_data

    fast_opts = None if not hasattr(meta_model, 'fast_opts') else meta_model.fast_opts
    for it in range(len(train_data)):
        fast_models = []
        for i in range(trajectory):
            fast_model, fast_opts = init_network(meta_model, meta_lr, fast_opts, Adam=True, beta1=0.9, beta2=0.999)
            for j in range(length):
                # if args.loader == 'original':
                #     meta_train_data, meta_test_data = get_data(train_data, i, device, random=True)
                # else:
                meta_train_data, meta_test_data = split_image_and_label(to(next(iters), device), size=args.batch_size, loo=True)

                zero_and_update(fast_opts, get_loss_and_acc(fast_model.step(**meta_train_data), running_loss, running_corrects))
                zero_and_update(fast_opts, get_loss_and_acc(fast_model.step(**meta_test_data), running_loss, running_corrects))
            fast_models.append(get_parameters(fast_model))
        # print(it)
        update_meta_model(meta_model, fast_models, meta_optimizers, meta_lr=1)
    meta_model.fast_opts = fast_opts
    # re-estimate BN
    # iters.stop()
    AveragedModel(start_epoch=0, device=device).update_bn(train_data, epoch, iters=len(train_data) // 2, model=meta_model)
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()


@TrainFuncs.register('mvrml_p')
def MVRML_paralleled(meta_model, train_data, meta_lr, epoch, args, engine, mode):
    assert args.loader == 'meta'
    device, meta_optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_lr = meta_optimizers.param_groups[0]['lr'] / args.meta_lr_weight
    domains, inner_loops = len(train_data), len(train_data)

    trajectory, length = args.trajectory, args.length
    print('Meta lr : {}, loops : {}, trajectory : {}, length : {}'.format(meta_lr, inner_loops, trajectory, length))
    devices = [int(args.gpu) + i for i in range(0, trajectory)]
    paralleled_devices = devices[:trajectory]

    # fast_opts = None if not hasattr(meta_model, 'fast_opts') else meta_model.fast_opts
    if not hasattr(engine, 'loaders'):
        engine.loaders = [iter(CudaDataLoader(engine.get_loaders()[0][0], i, 'cuda:{}'.format(paralleled_devices[i]), queue_size=5)) for i in range(trajectory)]
        # engine.loaders = [(engine.get_loaders()[0][0]) for i in range(trajectory)]
        engine.paralleled_meta_model = ParallelUpdate(meta_model, meta_optimizers, meta_lr, paralleled_devices, trajectory, length)

    for it in range(inner_loops):
        try:
            engine.paralleled_meta_model(engine.loaders)
        except Exception as e:
            print(traceback.format_exc())
            print('Shared Memory Error/Deadlock may occurs. If this error still appears, try re-running the code from current checkpoint '
                  'and lower the number of workers! Upgrade pytorch version may also helps!')
            [l.stop() for l in engine.loaders]
            engine.loaders = [iter(CudaDataLoader(engine.get_loaders()[0][0], i, 'cuda:{}'.format(paralleled_devices[i]), queue_size=5)) for i in range(trajectory)]

    AveragedModel(start_epoch=0, device=device).update_bn(engine.loaders[0], epoch, iters=inner_loops // 2, model=meta_model)
    if epoch == args.num_epoch - 1:
        [l.stop() for l in engine.loaders]

    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()


@TrainFuncs.register('mvrml_content')
def MVRML_content(meta_model, train_data, meta_lr, epoch, args, engine, mode):
    assert args.loader != 'standard'
    device, meta_optimizers = engine.device, engine.optimizers
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    meta_lr = meta_optimizers.param_groups[0]['lr'] / 50

    trajectory, length = args.trajectory, args.length
    print('Meta lr : {}, loops : {}, path : {}, length : {}'.format(meta_lr, len(train_data), trajectory, length))

    iters = train_data

    if not hasattr(engine, 'ensemble_model'):
        ensemble_model = EnsembleModel()
        engine.ensemble_model = ensemble_model
        print('Init models')

    ensemble_model = engine.ensemble_model

    fast_opts = None if not hasattr(meta_model, 'fast_opts') else meta_model.fast_opts
    for it in range(len(train_data)):
        fast_models = []
        for i in range(trajectory):
            fast_model, fast_opts = init_network(meta_model, meta_lr, fast_opts, Adam=True, beta1=0.9, beta2=0.999)

            for j in range(length):
                data = iters.next()
                data = to(data, device)
                meta_train_data, meta_test_data = split_image_and_label(data, size=args.batch_size, loo=True)

                meta_train_out = fast_model.step(**meta_train_data)
                meta_train_loss = get_loss_and_acc(meta_train_out, running_loss, running_corrects)
                zero_and_update(fast_opts, meta_train_loss)

                meta_test_out = fast_model.step(**meta_test_data)
                meta_test_loss = get_loss_and_acc(meta_test_out, running_loss, running_corrects)

                meta_test_loss = meta_test_loss

                zero_and_update(fast_opts, meta_test_loss)

            fast_models.append(get_parameters(fast_model))

        update_meta_model(meta_model, fast_models, meta_optimizers, meta_lr=1)
        ensemble_model.update_model(meta_model)

    AveragedModel(start_epoch=0, device=device).update_bn(train_data, epoch, iters=len(train_data) // 2, model=meta_model)
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()



@EvalFuncs.register('mvrml_content')
def deepall_eval(model, eval_data, lr, epoch, args, engine, mode):
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    device = engine.device

    model.eval()

    with torch.no_grad():
        if isinstance(eval_data, (tuple, list)):
            eval_data = eval_data[0]
        for i, data_list in enumerate(eval_data):
            data_list = to(data_list, device)
            outputs = model(**data_list, epoch=epoch, step=len(eval_data) * epoch + i, engine=engine, train_mode='test')
            get_loss_and_acc(outputs, running_loss, running_corrects)

    ensemble_model = engine.ensemble_model

    if mode == 'eval':
        val_loss = running_loss.get_average_dicts()['main']
        # ensemble_model.add_model(model, val_loss)
        ensemble_model.update_loss(val_loss)

    if mode == 'test':
        # save_dict = {
        #     'models': ensemble_model.cached_models,
        #     'losses': ensemble_model.val_losses,
        # }
        # torch.save(save_dict, os.path.join(engine.path, 'models', 'Ensemble.pt'))

        scales = [0, 0.5, 1, 3, 5, 10, 20]
        models = []
        for i, s in enumerate(scales):
            avg_model = ensemble_model.get_avg_model(model, s)
            avg_model = avg_model.to(device)
            models.append(avg_model)

        # AveragedModel(start_epoch=0, device=device).update_bn(engine.source_train, epoch, iters=len(engine.source_train) // 2, model=avg_model)
        outs = [reset_model(m) for m in models]
        loader = engine.source_train
        with torch.no_grad():
            if not hasattr(loader, 'next'):
                loader = iter(loader)
            for i in range(len(engine.source_train) // 2):
                if hasattr(loader, 'next'):
                    data_list = to(loader.next(), device)
                else:
                    data_list = to(next(loader), device)
                [m(**data_list) for m in models]
        [recover_model(m, *o) for m, o in zip(models, outs)]

        with torch.no_grad():
            if isinstance(eval_data, (tuple, list)):
                eval_data = eval_data[0]
            for i, data_list in enumerate(eval_data):
                data_list = to(data_list, device)
                for m, s in zip(models, scales):
                    outputs = m(**data_list)
                    get_loss_and_acc(outputs, running_loss, running_corrects, prefix='Ensemble-{}_'.format(s))
        print('save to Ensemble.pt')
        states = [m.state_dict() for m in models]

        save_dict = {
            'models': ensemble_model.cached_models,
            'losses': ensemble_model.val_losses,
            'states': states
        }
        torch.save(save_dict, os.path.join(engine.path, 'models', 'Ensemble.pt'))

    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()

    return acc['main'], (loss, acc)
