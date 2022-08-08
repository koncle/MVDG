from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from framework.log import read_file
from framework.registry import Datasets
from utils.visualize import *
from utils.visualize import get_ticks_by_num

sns.set(style="whitegrid")

PACS = {'0': 'dog', '1': 'elephant', '2': 'giraffe', '3': 'guitar', '4': 'horse', '5': 'house', '6': 'person'}


def plot_confidence(confidence, i, j):
    text_size = 20
    # sns.set(style=None)
    pacs = list(PACS.values())
    # bar_plot([confidence], pacs)

    plt.ylim([0, 100])

    # ax1 = plt.axes()
    x = range(len(pacs))

    plt.bar(x, confidence)
    plt.xticks(x, pacs, size=12)
    # ax1.grid(False)
    # plt.axis('off')f
    # ax1.axes.get_xaxis().set_visible(False)
    # ax1.axes.get_yaxis().set_visible(False)
    # plt.xticks(size=text_size)
    plt.yticks(size=text_size)
    plt.savefig('conf_{}-{}.pdf'.format(i, j), bbox_inches='tight')
    plt.show()


def plot_changed_confidence():
    L = [
        [[18, 0, 0, 0, 82, 0, 0],
         [98, 1.4, 0, 0, 0.6, 0, 0]],

        [[82, 11, 0, 0, 7, 0, 0],
         [24, 54, 0, 0, 19, 0, 3]],

        [[1, 43, 1.5, 0.5, 63, 0.5, 0.5],
         [0, 94, 0, 0, 6, 0, 0]],
    ]

    for i, a in enumerate(L):
        for j in range(2):
            plot_confidence(a[j], i + 1, j + 1)


# plot_changed_confidence()


def get_task_num_graph():
    task_num_data = {
        'x': list(range(1, 7)),
        'val': [82.34, 82.84, 83.97, 84.95, 85.00, 84.79],
        'last': [82.72, 83.12, 84.62, 85.31, 85.15, 85.24]
    }
    # for k, v in task_num_data.items():
    #     if k != 'x':
    #         plt.plot(task_num_data['x'], v, label=k)
    # plt.legend()
    # plt.savefig("task_num.pdf")

    text_size = 'x-large'
    plt.rc('legend', fontsize=14)
    val, last = task_num_data['val'], task_num_data['last']
    bar_plot([val], x_axis_names=['{}'.format(i + 1) for i in range(len(val))], y_label='Acc', x_label='Tasks',
             ylim=[82, 85.5], width_of_all_col=0.5, save_name='task.pdf', figsize=(8, 6), title_size='xx-large',
             plot_text=True
             )


def get_longer_training_graph():
    label = ['Art', 'Cartoon', 'Photo', 'Sketch']
    x = list(range(4))
    y1 = [80.59, 76.23, 94.91, 77.65]
    y2 = [80.64, 77.22, 94.24, 77.41]  # 82.38]

    plt.rc('legend', fontsize=14)
    bar_plot([y1, y2], label, y_label='Acc', x_label='Domains', bar_names=['normal learning', 'longer training'],
             ylim=[70, 100], width_of_all_col=0.8, offset_between_bars=0.03, save_name='longer_training.pdf', figsize=(8, 6),
             title_size=15, label_size=24, tick_size=20, legend_size=18,
             plot_text=True,
             )


def get_tta_bs_graph():
    Normal_size = 24
    small_size = 22
    plt.rc('axes', titlesize=Normal_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=Normal_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)  # legend fontsize
    plt.rc('figure', titlesize=Normal_size)  # fontsize of the figure title
    TTA_bs_data = {
        'x': [str(2 ** i) for i in range(7)],
        'deepall': [79.21, 80.24, 80.49, 80.67, 80.79, 80.76, 80.81],
        # 'ours': [84.95, 85.10, 85.50, 85.61, 85.45, 85.90, 85.64]
        'ours': [85.49, 85.98, 86.33, 86.44, 86.44, 86.56, 86.52]
    }
    data = TTA_bs_data

    get_twin_fig(data['x'], data['deepall'], data['ours'], labels=['DeepAll', 'Ours'],
                 x_range=[79, 81.2], y_range=[85.4, 86.7], tick_num=5, legend_loc='upper left',
                 y_name=['DeepAll', 'Ours'], x_name='Augmented images', figsize=(7, 5), save_name='tta_with_time.pdf')

    # times = [2, 3, 4, 8, 14, 32, 64]
    # bar_plot([times], x_aixs_names=[1, 2, 4, 8, 16, 32, 64], offset_between_bars=0.2, save_name='time-TTA-bs.pdf',
    #          y_label='test time (s)', x_label='the # of augmented images')


def gaussian(xdata, ydata):
    from sklearn.gaussian_process import GaussianProcessRegressor
    xdata, ydata = np.array(xdata), np.array(ydata)
    # 计算高斯过程回归，使其符合 fit 数据点
    gp = GaussianProcessRegressor()
    gp.fit(xdata[:, np.newaxis], ydata)

    xfit = np.linspace(1, 30, 1000)
    yfit, std = gp.predict(xfit[:, np.newaxis], return_std=True)
    dyfit = 2 * std  # 两倍sigma ~ 95% 确定区域
    plt.plot(xdata, ydata, 'or')
    plt.plot(xfit, yfit, '-', color='gray')

    plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2)
    plt.show()


def plot_confidence_interval(x, y, label, color=None, ax=None):
    x = np.array(x)
    y = np.array(y) * 100
    print(y)
    std = np.std(y, axis=0)
    y = np.mean(y, axis=0)
    if ax is None:
        plt.plot(x, y, color=color, linewidth=2)
        plt.fill_between(x, (y - std), (y + std), alpha=.3, label=label, color=color)
    else:
        ax.plot(x, y, color=color, linewidth=2)
        ax.fill_between(x, (y - std), (y + std), alpha=.3, label=label, color=color)


def compare_BN_effect():
    Normal_size = 14
    small_size = 14
    plt.rc('axes', titlesize=Normal_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=Normal_size)  # fontsize of the x and y labels

    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)  # legend fontsize
    # plt.rc('figure', titlesize=Normal_size)  # fontsize of the figure title

    path1 = '/data/zj/PycharmProjects/TTA/exp/PACS_mvrml_parallel_step3_rand'
    path2 = '/data/zj/PycharmProjects/TTA/exp/PACS_mvrml_parallel_step3_length3_noUpdateBN'
    path1, path2 = Path(path1), Path(path2)
    domains = Datasets['PACS'].Domains
    times = 3
    print(domains)
    ds = ['Photo', 'Art', 'Cartoon', 'Sketch']
    for i, d in enumerate(domains):
        x_lists, y_lists_1, y_lists_2 = [], [], []
        for t in range(times):
            f1, f2 = path1 / (d + str(t)) / 'target_test.txt', path2 / (d + str(t)) / 'target_test.txt'
            acc_list1, acc_list2 = read_file(f1)[0], read_file(f2)[0]
            # plt.plot(range(1, len(acc_list1)+1), acc_list1, '-',  label='NoUpdate', color='#5A9BD5', linewidth=2)
            # plt.plot(range(1, len(acc_list2)+1), acc_list2, '-', label='Update', color='#FF9966', linewidth=2)
            # plt.xlim([0, 31])
            # plt.show()
            # sns.relplot()

            y_lists_1.append(acc_list1)
            y_lists_2.append(acc_list2)
        x_lists = list(range(len(y_lists_1[0])))
        a = np.stack([y_lists_1, y_lists_2], 0) * 100
        spacing = (a.max() - a.min()) / 7
        # spacing = 0.04 # This can be your user specified spacing.

        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        minorLocator = MultipleLocator(spacing)
        ax1.yaxis.set_major_locator(minorLocator)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # No decimal places
        text_size = 'xx-large'
        plot_confidence_interval(x_lists, y_lists_1, label='Update', color='#5A9BD5', ax=ax1)
        plot_confidence_interval(x_lists, y_lists_2, label='None', color='#FF9966', ax=ax1)
        plt.xlabel('Epochs', size=text_size)
        plt.ylabel('Acc', size=text_size)
        plt.xticks(size=text_size)
        plt.yticks(size=text_size)
        plt.legend(loc='lower right', fontsize='x-large')
        plt.title(ds[i], size=text_size)
        plt.savefig('BN_{}.pdf'.format(d), bbox_inches='tight')
        plt.show()


def plot_number_of_tasks():
    val = [82.09, 84.49, 85.89, 85.94]  # , 85.59]
    bar_plot([val], x_axis_names=['{}'.format(i + 1) for i in range(len(val))], y_label='Acc', x_label='Tasks',
             ylim=[81, 86.5], width_of_all_col=0.6, save_name='task.pdf', figsize=(8, 6),
             title_size=24, tick_size=26, label_size=28, ytick_format_n=1,
             plot_text=True)


def plot_number_of_trajectories():
    val = [83.68, 85.36, 85.89, 86.11]  # , 85.60]
    yticks = get_ticks_by_num([82, 86.5], tick_space=0.8)
    bar_plot([val], x_axis_names=['{}'.format(i + 1) for i in range(len(val))], y_label='Acc', x_label='Trajectories', yticks=yticks,
             ylim=[82, 86.5], width_of_all_col=0.6, save_name='trajectory.pdf', figsize=(8, 6),
             title_size=24, tick_size=26, label_size=28, ytick_format_n=1,
             plot_text=True)


def plot_flatness():
    domains = Datasets['PACS'].Domains
    ERM_val = [
        [0.0015, 0.0032, 0.0063, 0.0096, 0.0226, 0.0335, 0.0506, 0.0545, 0.0715, 0.1017, 0.1382, 0.1993, 0.2452, 0.2862, 0.5018, 0.6407, 0.7176, 0.8162, 0.8630],
        [0.0007, 0.0019, 0.0079, 0.0140, 0.0254, 0.0335, 0.0465, 0.0583, 0.0674, 0.1075, 0.1562, 0.2040, 0.2781, 0.3039, 0.5613, 0.7135, 0.7172, 0.8898, 0.8524],
        [0.0003, 0.0029, 0.0040, 0.0096, 0.0122, 0.0251, 0.0468, 0.0396, 0.0594, 0.0693, 0.1120, 0.1856, 0.2153, 0.2794, 0.5207, 0.6370, 0.5393, 0.7352, 0.8198],
        [0.0001, 0.0024, 0.0073, 0.0079, 0.0082, 0.0149, 0.0324, 0.0367, 0.0420, 0.0552, 0.0911, 0.1337, 0.1830, 0.2040, 0.2734, 0.4253, 0.3782, 0.4876, 0.4702]
    ]
    MLDG_val = [
        [0.0010, 0.0016, 0.0023, 0.0062, 0.0109, 0.0150, 0.0175, 0.0310, 0.0427, 0.0489, 0.0714, 0.0867, 0.1525, 0.1610, 0.2059, 0.2754, 0.3765, 0.2658, 0.4129],
        [0.0008, 0.0005, 0.0053, 0.0052, 0.0074, 0.0151, 0.0149, 0.0203, 0.0351, 0.0379, 0.0552, 0.0592, 0.0937, 0.1262, 0.1706, 0.2066, 0.3163, 0.2178, 0.3254],
        [0.0006, 0.0001, 0.0019, 0.0075, 0.0035, 0.0186, 0.0160, 0.0292, 0.0403, 0.0394, 0.0609, 0.0848, 0.1172, 0.1388, 0.1816, 0.2169, 0.3193, 0.2638, 0.3508],
        [0.0003, 0.0028, 0.0055, 0.0011, 0.0089, 0.0120, 0.0158, 0.0314, 0.0309, 0.0376, 0.0688, 0.0783, 0.1081, 0.1356, 0.1379, 0.2089, 0.2533, 0.2162, 0.2643]
    ]
    MVRML_val = [
        [0.0004, 0.0002, 0.0014, 0.0031, 0.0040, 0.0089, 0.0088, 0.0135, 0.0186, 0.0236, 0.0308, 0.0389, 0.0593, 0.0668, 0.0827, 0.1079, 0.1180, 0.1296, 0.1811],
        [0.0003, 0.0005, 0.0001, 0.0019, 0.0036, 0.0020, 0.0038, 0.0080, 0.0076, 0.0172, 0.0198, 0.0188, 0.0297, 0.0345, 0.0547, 0.0678, 0.0939, 0.0933, 0.1280],
        [0.0004, 0.0003, 0.0012, 0.0030, 0.0055, 0.0076, 0.0082, 0.0107, 0.0163, 0.0189, 0.0289, 0.0392, 0.0494, 0.0700, 0.0855, 0.1156, 0.1171, 0.1267, 0.1792],
        [0.0001, 0.0014, 0.0028, 0.0010, 0.0026, 0.0089, 0.0108, 0.0142, 0.0170, 0.0233, 0.0308, 0.0415, 0.0508, 0.0613, 0.0660, 0.1143, 0.1208, 0.1358, 0.1416]
    ]

    ERM_test = [
        [0.0010, 0.0012, 0.0028, 0.0084, 0.0131, 0.0259, 0.0348, 0.0434, 0.0430, 0.0777, 0.0926, 0.1519, 0.2124, 0.1930, 0.2665, 0.5092, 0.4034, 0.4829, 0.5128],
        [-0.0010, 0.0181, 0.0041, 0.0221, 0.0390, 0.0539, 0.0273, 0.0563, 0.1338, 0.2556, 0.2758, 0.2951, 0.3962, 0.6538, 0.5696, 1.1512, 0.8986, 1.0510, 1.1355],
        [0.0096, 0.0065, -0.0066, -0.0022, 0.0637, 0.0540, 0.0632, 0.1148, 0.1094, 0.1914, 0.3018, 0.4353, 0.5600, 0.5908, 0.7612, 0.9605, 0.6056, 0.9732, 0.8314],
        [-0.0055, 0.0081, 0.0528, 0.0301, -0.0190, 0.0318, 0.1457, 0.0329, 0.0590, 0.1640, 0.2801, 0.3201, 0.3241, 0.3072, 0.5272, 0.4750, 0.8311, 0.6904, 0.8373],
    ]
    MLDG_test = [
        [0.0009, 0.0016, 0.0061, 0.0089, 0.0129, 0.0247, 0.0374, 0.0408, 0.0500, 0.0561, 0.1008, 0.1307, 0.1815, 0.1767, 0.2018, 0.2433, 0.3556, 0.3071, 0.3705],
        [0.0007, 0.0145, 0.0115, 0.0262, 0.0252, 0.0125, 0.0432, 0.1045, 0.1026, 0.2042, 0.1796, 0.1851, 0.2650, 0.3094, 0.3474, 0.5074, 0.6441, 0.5749, 0.7738],
        [0.0041, -0.0010, 0.0049, -0.0074, 0.0413, 0.0345, 0.0328, 0.0458, 0.1065, 0.0523, 0.1521, 0.1954, 0.3254, 0.3764, 0.3619, 0.4128, 0.4421, 0.4469, 0.3717],
        [0.0049, 0.0079, 0.0121, 0.0195, -0.0004, 0.0763, 0.0404, 0.0911, 0.0871, 0.0908, 0.1960, 0.1603, 0.1679, 0.2959, 0.2583, 0.3526, 0.4299, 0.3324, 0.5674]
    ]
    MVRML_test = [
        [0.0003, 0.0010, 0.0046, 0.0079, 0.0056, 0.0108, 0.0211, 0.0202, 0.0278, 0.0354, 0.0563, 0.0715, 0.1009, 0.1021, 0.1228, 0.1430, 0.1721, 0.2199, 0.2703],
        [-0.0006, 0.0061, 0.0048, 0.0103, 0.0113, 0.0171, 0.0287, 0.0575, 0.0456, 0.0860, 0.0958, 0.0887, 0.1436, 0.1337, 0.2217, 0.2640, 0.3370, 0.2917, 0.4243],
        [0.0039, -0.0001, 0.0020, -0.0126, 0.0250, 0.0166, 0.0267, 0.0512, 0.0717, 0.0456, 0.0983, 0.1507, 0.2266, 0.2511, 0.2863, 0.3488, 0.2665, 0.3159, 0.3160],
        [-0.0006, 0.0044, -0.0030, 0.0141, -0.0015, 0.0363, 0.0247, 0.0389, 0.0251, 0.0364, 0.0804, 0.0704, 0.1039, 0.1194, 0.1422, 0.1914, 0.2578, 0.2230, 0.2774]
    ]

    def numpy_ewma_vectorized_v2(data, window):
        data = np.array(data)
        alpha = 2 / (window + 1.0)
        alpha_rev = 1 - alpha
        n = data.shape[0]

        pows = alpha_rev ** (np.arange(n + 1))

        scale_arr = 1 / pows[:-1]
        offset = data[0] * pows[1:]
        pw0 = alpha * alpha_rev ** (n - 1)

        mult = data * pw0 * scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums * scale_arr[::-1]
        return out

    vals = [ERM_val, MLDG_val, MVRML_val]
    tests = [ERM_test, MLDG_test, MVRML_test]
    # tests = pre
    # for v in tests:
    #     for x in v:
    #         print(len(x))
    titles = ['Photo', 'Art', 'Cartoon', 'Sketch']
    names = ['ERM', 'Reptile', 'MVRML']
    colors = ['#2CBDFE', '#47DBCD', '#9D2EC5']
    x = [i * 2 for i in range(len(tests[0][0]) + 1)]
    text_size = 26
    length = 20
    for d in range(0, 4):
        for m, l, c in zip(vals, names, colors):
            plt.plot(x[:length], np.abs([0] + m[d][:length]), alpha=1, color=None, label=l, linewidth=3)
            data = numpy_ewma_vectorized_v2(np.abs(m[d][:length]), 10)
            # plt.plot(x[:length], data, label=l, color=c)
            # plot_confidence_interval(x[:length], m[d][:length], label=l, color='#5A9BD5')
        # plt.xticks(list(range(40)))
        plt.legend(fontsize=22)
        plt.title(titles[d], size=text_size)
        plt.xticks(size=text_size)
        plt.yticks(size=text_size)
        # plt.xlabel('Distance', size=text_size)  # 横坐标名字
        # plt.ylabel('Sharpness', size=text_size)  # 纵坐标名字
        plt.savefig('flat_{}_train.pdf'.format(domains[d]), bbox_inches='tight')
        plt.show()


# plot_number_of_tasks()
# plot_number_of_trajectories()

# plot_flatness()

# get_longer_training_graph()
# get_task_num_graph()
# compare_BN_effect()
# get_tta_bs_graph()

def draw_loss_lines():
    p = np.linspace(0, 1, 100)
    em_loss = - p * np.log(p)
    slr_loss = - p * np.log(p / (1 - p))
    emt_loss = - p * np.log(p)
