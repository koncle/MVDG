import copy
import os

import torch.optim
from tqdm import tqdm

from framework.loss_and_acc import get_loss_and_acc
from framework.registry import Datasets, EvalFuncs
from utils.tensor_utils import AverageMeterDict
from utils.tensor_utils import to
from utils.visualize import show_graphs
from models.meta_util import *

@torch.no_grad()
def mvp_pred(model, aug_data, test_label):
    N, aug_n, C, H, W = aug_data.shape
    aug_data = aug_data.reshape(-1, C, H, W)
    aug_label = test_label.unsqueeze(0).repeat(aug_n, 1).reshape(-1)
    outputs2 = model.step(aug_data, aug_label)
    logits2 = outputs2['logits']
    mean_logits = logits2.reshape(N, aug_n, -1).mean(1)
    changed = N - ((logits2.reshape(N, aug_n, -1).argmax(-1)== logits2.reshape(N, aug_n, -1).argmax(-1)[:, 0:1]).sum(-1) == aug_n).sum().float().item()
    outputs2 = {'MVP': {'acc_type': 'acc', 'pred': mean_logits, 'target': test_label},  'logits':mean_logits}
    return outputs2, changed


@torch.no_grad()
def camvp_pred(logits, feats, model, aug_data, test_label, thresh=0.7):
    conf = logits.softmax(1).max(1)[0]
    idx = conf < thresh

    aug_data = aug_data[idx]
    aug_origin_feats = feats[idx]
    new_logits = copy.deepcopy(logits)
    if len(aug_data) > 0:
        # MVP acc
        N, aug_n, C, H, W = aug_data.shape
        # print(aug_n)
        aug_data = aug_data.reshape(-1, C, H, W)
        outputs2 = model.step(aug_data, None)
        logits2 = outputs2['logits'].reshape(N, aug_n, -1)
        feats2 =  outputs2['out'][0].reshape(N, aug_n, -1)
        # print(dist.shape)
        # print(dist)
        
        logits2 = torch.cat([logits[idx][:, None], logits2], 1)

        aug_conf = logits2.softmax(-1).max(-1, keepdims=True)[0]  # N x aug_n x 1
        # print(aug_conf)
        mask = (aug_conf >= conf[idx][:, None, None])
        
        
        dist = torch.cosine_similarity(aug_origin_feats[:, None], 
                                       torch.cat([aug_origin_feats[:, None], feats2],  dim=1), dim=-1)[:, :, None]
        # print(dist.mean())
        mask = dist > 0.9
        # mask = aug_conf > 0.1
        
        # print(mask.shape[1], mask.sum()/mask.shape.numel())
        # mask = (aug_conf >= 0.6)
        
        # cur_idx = (mask1.sum(1) == 0)[:, 0]
        # mask2[cur_idx] = mask1[cur_idx]
        # mask = mask2

        mean_logits = (logits2 * mask).sum(1) / mask.sum(1)

        new_logits[idx] = mean_logits

    outputs2 = {'CaMVP': {'acc_type': 'acc', 'pred': new_logits, 'target': test_label}, 'logits':new_logits}
    return outputs2


@EvalFuncs.register('mvp')
def MVP_eval_model(model, eval_data, lr, epoch, args, engine, mode):
    running_loss, running_corrects, shadow_running_corrects = AverageMeterDict(), AverageMeterDict(), AverageMeterDict()

    device = engine.device

    path = os.path.join(engine.path, 'models', 'Ensemble.pt')
    models = torch.load(path, map_location='cpu')

    # ensemble_model = EnsembleModel()
    # ensemble_model.cached_models = models['models']
    # ensemble_model.val_losses = models['losses']
    #
    # model = ensemble_model.get_avg_model(model, loss_scale=10).to(device)
    model.load_state_dict(models['states'][-2])
    model.eval()

    # # State dicts
    # if len(args.MVP_model_path) > 0:
    #     cur_domain = Datasets[args.dataset].Domains[args.exp_num[0]]
    #     model_path = args.MVP_model_path
    #     model.load_pretrained(model_path)
    #     print("Loaded models from {}".format(model_path))

    # correctness when confidence > 0.9
    prob = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1]
    correct_all = [0 for i in range(len(prob) - 1)]
    correct_num = [0 for i in range(len(prob) - 1)]

    after_correct_all = [0 for i in range(len(prob) - 1)]
    after_correct_num = [0 for i in range(len(prob) - 1)]

    all_num = 0
    diff_conf = []
    timer = Timer(verbose=False)
    t = 0
    change_num = 0
    with torch.no_grad():
        model.eval()  # eval mode for normal test
        for i, data_list in enumerate(tqdm(eval_data)):
            test_data, test_label, aug_data = data_list['x'], data_list['label'], data_list['aug_x']
            test_data, aug_data, test_label = to([test_data, aug_data, test_label], device)
            
            # original acc
            with timer:
                outputs = model.step(test_data, test_label)
                original_logits = outputs['logits']
                _ = get_loss_and_acc(outputs, running_loss, running_corrects)

                outputs2, change_n = mvp_pred(model, aug_data, test_label)
                _ = get_loss_and_acc(outputs2, running_loss, running_corrects)

                # logits = outputs['logits']
                # outputs3 = camvp_pred(logits, model, aug_data, test_label)
                # _ = get_loss_and_acc(outputs3, running_loss, running_corrects)

                # change_num += change_n
                all_num += len(original_logits)
            s = timer.get_last_duration()
            t += s
    print("Time : ", t)
    # change_rate = change_num / all_num
    # print('change rate', change_rate)
    # print(diff_conf)
    loss = running_loss.get_average_dicts()
    acc = running_corrects.get_average_dicts()
    # correctness = [correct_num[i] / correct_all[i] if correct_all[i] != 0 else 0 for i in range(len(prob)-1)]
    # percentage = [correct_num[i] / all_num for i in range(len(prob)-1)]
    # all_percentage = [correct_all[i] / all_num for i in range(len(prob)-1)]
    #
    # after_correctness = [after_correct_num[i] / after_correct_all[i] if after_correct_all[i] != 0 else 0 for i in range(len(prob)-1)]
    # after_percentage = [after_correct_num[i] / all_num for i in range(len(prob)-1)]
    # after_all_percentage = [after_correct_all[i] / all_num for i in range(len(prob)-1)]
    #
    # print('======================')
    # print('all_num', all_num)
    # print('correctness', correctness)
    # print('percentage', percentage)
    # print('all_percentage', all_percentage)
    # print('======================')
    # print('after_correctness', after_correctness)
    # print('after_percentage', after_percentage)
    # print('after_all_percentage', after_all_percentage)
    # print('======================')
    # return change_rate, (loss, acc)
    return acc['MVP'], (loss, acc)
    # if 'MVP' in acc:
    #     return acc['MVP'], (loss, acc)
    # else:
    #     return acc['main'], (loss, acc)


@EvalFuncs.register('camvp')
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
            test_data, test_label, aug_data = data_list['x'], data_list['label'], data_list['aug_x']
            test_data, aug_data, test_label = to([test_data, aug_data, test_label], device)

            # show_graphs(aug_data[0], filename='test_weak.pdf')
            # return 
        
            # original acc
            outputs = model.step(test_data, test_label)
            logits = outputs['logits']
            feats = outputs['out'][0]
            _ = get_loss_and_acc(outputs, running_loss, running_corrects)
            outputs2 = camvp_pred(logits, feats, model, aug_data, test_label)
            _ = get_loss_and_acc(outputs2, running_loss, running_corrects)

    loss = running_loss.get_average_dicts()
    acc = running_corrects.get_average_dicts()
    if 'main' in acc:
        return acc['main'], (loss, acc)
    else:
        return 0, (loss, acc)
