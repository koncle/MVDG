import torch.optim
from tqdm import tqdm

from framework.loss_and_acc import get_loss_and_acc
from framework.registry import Datasets, EvalFuncs
from utils.tensor_utils import AverageMeterDict
from utils.tensor_utils import to


@EvalFuncs.register('mvp')
def MVP_eval_model(model, eval_data, lr, epoch, args, engine, mode):
    device = engine.device
    running_loss, running_corrects, shadow_running_corrects = AverageMeterDict(), AverageMeterDict(), AverageMeterDict()

    # State dicts
    if len(args.MVP_model_path) > 0:
        cur_domain = Datasets[args.dataset].Domains[args.exp_num[0]]
        model_path = args.MVP_model_path
        model.load_pretrained(model_path)
        print("Loaded models from {}".format(model_path))

    with torch.no_grad():
        model.eval()  # eval mode for normal test
        for i, data_list in enumerate(tqdm(eval_data)):
            test_data, test_label, aug_data = data_list['x'], data_list['label'], data_list['mvp']
            test_data, aug_data, test_label = to([test_data, aug_data, test_label], device)

            # original acc
            outputs = model.step(test_data, test_label)
            logits = outputs['logits']
            _ = get_loss_and_acc(outputs, running_loss, running_corrects)

            # MVP acc
            N, aug_n, C, H, W = aug_data.shape
            aug_data = aug_data.reshape(-1, C, H, W)
            aug_label = test_label.unsqueeze(1).repeat(aug_n, 1).reshape(-1)
            outputs2 = model.step(aug_data, aug_label)
            logits2 = outputs2['logits']
            mean_logits = logits2.reshape(N, aug_n, -1).mean(1)

            outputs2 = {'MVP': {'acc_type': 'acc', 'pred': mean_logits, 'target': test_label},}
            _ = get_loss_and_acc(outputs2, running_loss, running_corrects)

    loss = running_loss.get_average_dicts()
    acc = running_corrects.get_average_dicts()
    if 'main' in acc:
        return acc['MVP'], (loss, acc)
    else:
        return 0, (loss, acc)
