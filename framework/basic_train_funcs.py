import torch

from framework.loss_and_acc import get_loss_and_acc
from framework.registry import EvalFuncs, TrainFuncs
from utils.tensor_utils import to, AverageMeterDict


@TrainFuncs.register('meta')
@TrainFuncs.register('deepall')
def deepall_train(model, train_data, lr, epoch, args, engine, mode):
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    optimizers, device = engine.optimizers, engine.device

    model.train()
    for i, data_list in enumerate(train_data):
        data_list = to(data_list, device)
        output_dicts = model(**data_list, epoch=epoch, step=len(train_data) * epoch + i, engine=engine, train_mode='train')
        total_loss = get_loss_and_acc(output_dicts, running_loss, running_corrects)
        if total_loss is not None:
            total_loss.backward()
        # zero_and_update(optimizers, total_loss)
        optimizers.step()
        optimizers.zero_grad()
    return running_loss.get_average_dicts(), running_corrects.get_average_dicts()


@EvalFuncs.register('meta')
@EvalFuncs.register('deepall')
def deepall_eval(model, eval_data, lr, epoch, args, engine, mode):
    running_loss, running_corrects = AverageMeterDict(), AverageMeterDict()
    device = engine.device

    model.eval()

    with torch.no_grad():
        for i, data_list in enumerate(eval_data):
            data_list = to(data_list, device)
            outputs = model(**data_list, epoch=epoch, step=len(eval_data) * epoch + i, engine=engine, train_mode='test')
            get_loss_and_acc(outputs, running_loss, running_corrects)
    loss, acc = running_loss.get_average_dicts(), running_corrects.get_average_dicts()
    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)

