import copy
import itertools

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from framework.loss_and_acc import get_loss_and_acc
from models.meta_util import get_parameters, put_parameters, AveragedModel
from framework.registry import Datasets
from main import get_default_parser
from framework.engine import GenericEngine
from utils.tensor_utils import *
import torch


class LossSurfaceVisualization():
    """
    https://github.com/timgaripov/dnn-mode-connectivity/blob/master/plane_plot.py
    loss surface requires 3 different trained models
    """

    def __init__(self, args, model_paths):
        self.args = get_default_parser().parse_args(args)
        engine = GenericEngine(self.args, 0)
        self.models_params = []
        self.loaders = {
            'train': engine.source_train,
            'val': engine.source_val,
            'test': engine.target_test
        }
        self.device = engine.device
        self.model = engine.model

        self.model_params = [self.get_parameters(self.load_pretrained(p)) for p in model_paths]
        self.get_basis(self.model_params)

    def load_pretrained(self, p):
        self.model.load_pretrained(p, absolute=True)
        return self.model

    def get_basis(self, model_params):
        self.origin, self.base_x, self.base_y, self.dx, self.dy = self.get_orthonormal_basis(model_params)
        print('computed basis : ', self.dx, self.dy)
        self.coefficients = []
        for param in model_params:
            co = self.proj_weight_to_unit(param)
            self.coefficients.append(co)
        avg_param = (model_params[0] + model_params[1] + model_params[2]) / 3
        self.coefficients.append(self.proj_weight_to_unit(avg_param))
        print(self.coefficients)

    def proj(self, from_vec, to_vec):
        length = from_vec.dot(to_vec) / to_vec.norm(p=2)
        return length

    def proj_weight_to_unit(self, weight):
        l1 = self.proj(weight - self.origin, self.base_x)
        l2 = self.proj(weight - self.origin, self.base_y)
        return l1, l2

    def get_parameters(self, model):
        param_dict = get_parameters(model)[0]
        all_param = torch.cat([p.view(-1) for p in param_dict.values()])
        return all_param.clone()

    def get_orthonormal_basis(self, model_params):
        """
        given vectors v1, v2
        the orthonormal basis can be obtained by setting b1 = v1, b2 = v1xv2xv1 = v2-<v1, v2>/||v1||**2 * v2
        """
        p1, p2, p3 = model_params[:3]
        v2, v3 = (p2 - p1).view(-1), (p3 - p1).view(-1)
        dx = v2.norm(p=2)
        x = v2 / dx
        y = v3 - v2.dot(v3) / v2.norm(p=2) ** 2 * v2
        dy = y.norm(p=2)
        y = y / dy
        return p1, x, y, dx, dy

    def put_new_param(self, new_param, model):
        old_params = get_parameters(model)[0]
        shaped_param, offset = {}, 0
        for name, param in old_params.items():
            shape, num = param.shape, param.shape.numel()
            shaped_param[name] = new_param[offset:offset + num].view(*shape).clone()
            offset += num
        put_parameters(model, shaped_param, None)

    @torch.no_grad()
    def collect_loss(self, model, loader):
        running_loss, running_acc = AverageMeterDict(), AverageMeterDict()
        self.model.eval()
        for data in loader:
            data = to(data, self.device)
            outputs = model.step(**data)
            loss = get_loss_and_acc(outputs, running_loss, running_acc)
        # print(running_acc.get_average_dicts())
        return running_loss.get_average_dicts()['main']

    def compute_loss_surface(self, mode):
        loader = self.loaders[mode]
        length, margin = 5, 0.5
        x_list = np.linspace(-margin, 1 + margin, length)
        y_list = np.linspace(-margin, 1 + margin, length)
        avg_util = AveragedModel(start_epoch=0, device=self.device)
        stat = []
        combinations = list(itertools.product(x_list, y_list))
        # combinations = []
        # combinations.extend(self.coefficients)

        print('combination num : ', combinations)

        for x, y in tqdm(combinations):
            x, y = x * self.dx, y * self.dy
            new_param = self.origin + x * self.base_x + y * self.base_y
            self.put_new_param(new_param, self.model)
            avg_util.update_bn(self.loaders['train'], 30, meta=False, model=self.model)
            loss = self.collect_loss(self.model, loader)
            print((x, y, loss))
            stat.append((x, y, loss))
        return stat

    def compute_flatness(self, parameter, mode):
        avg_util = AveragedModel(start_epoch=0, device=self.device)

        self.put_new_param(parameter, self.model)
        avg_util.update_bn(self.loaders['train'], 30, meta=False, iters=20, model=self.model)
        origin_loss = self.collect_loss(self.model, self.loaders[mode])
        print('origin loss', origin_loss)
        out = []
        times = 10
        for gamma in tqdm(list(range(2, 40, 2))):
            avg_loss = 0
            for _ in range(times):
                delta = torch.randn_like(parameter)
                delta = delta / delta.norm() * gamma
                new_param = parameter + delta
                self.put_new_param(new_param, self.model)
                loss = self.collect_loss(self.model, self.loaders[mode])
                avg_loss += loss
                # print(' '*10, loss)
            avg_loss = avg_loss / times - origin_loss
            print('{} : {}'.format(gamma, avg_loss, origin_loss))
            out.append(avg_loss)
        return out


def plot_contour(values, coeff, title, epoch, mode):
    all_values = values
    a = int(np.sqrt(len(values)))
    values = np.array(values)[:a ** 2]
    plt.contourf(values[:, 0].reshape(a, a)[:, 0], values[:, 1].reshape(a, a)[0], values[:, 2].reshape(a, a), levels=10, extend='both')
    shapes = ['o', 'o', 'o', '*', 'v', 'v', 's', 's', '*', '*']
    # assert len(all_values[a**2:]) == len(shapes)
    for v, shape in zip(coeff, shapes):
        plt.scatter(v[0], v[1], marker=shape, color='yellow', s=60)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    plt.clim(values[:, -1].min(), values[:, -1].max())
    plt.title(title, size=25)
    plt.xticks([]), plt.yticks([])
    plt.savefig('flat_{}-{}.pdf'.format(epoch, mode), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dataset = 'PACS'
    datasets = Datasets[dataset]
    domains = datasets.Domains

    d, gpu, epoch, it, mode = 1, 3, 5, 2, 'test'

    args = ['--do-train=True', '--gpu={}'.format(gpu), '--batch-size=128', '--workers=8',
            '--save-path=../test', '--exp-num={}'.format(d), '--dataset={}'.format(dataset), '--model=erm', '--num-epoch=30']

    root1 = root2 = root3 = 'resnet18/{}0'.format(domains[d])
    paths = [
        '{}/models/model_best.pt'.format(root1),
        '{}/models/model_best.pt'.format(root2),
        '{}/models/model_best.pt'.format(root3),
    ]

    vis = LossSurfaceVisualization(args, paths)
    out = vis.compute_flatness(vis.model_params[0], mode='val')
    print(out)

