import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from avalanche.benchmarks.classic import SplitMNIST, PermutedMNIST
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.models import SimpleMLP
from avalanche.training import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC, ClassBalancedBuffer
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, AGEMPlugin

from pytorch_ood.detector import OpenMax, EnergyBased, Entropy
from pytorch_ood.utils import OODMetrics

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from utils.fm import init_logger, set_all_seed
from utils.viz import show_batch
from utils.data import count_sample_in_experience, count_sample_in_dataset, count_sample_in_stream, \
    get_full_dataset_experience

from avalanche_ood_plugins.detectors import OODDetectorPlugin


def train_and_eval(train_stream, test_stream, cl_strategy, seed=0, device='cpu'):
    N_EXPERIENCES = len(train_stream)

    set_all_seed(seed)

    # train stream loop, after every train exp test on all the test exps
    conf_matrix = np.zeros(shape=(N_EXPERIENCES, N_EXPERIENCES))
    ood_matrix = np.zeros(shape=(N_EXPERIENCES, N_EXPERIENCES))
    for i, train_exp in enumerate(train_stream):
        # tr_loader = DataLoader(train_exp.dataset, batch_size=10, shuffle=True)
        # x, y, _ = next(iter(tr_loader))
        # show_batch(x)
        set_all_seed(seed)
        cl_strategy.train(train_exp)
        # if i > 0:
        #     cumulative_dataset = cumulative_dataset + train_exp.dataset
        # else:
        #     cumulative_dataset = train_exp.dataset
        #
        # train_loader = DataLoader(cumulative_dataset, shuffle=True, batch_size=35)

        # # todo: this must be included as a plugin
        # train_loader = DataLoader(train_exp.dataset, batch_size=32, shuffle=True)
        # detector.fit(train_loader, device=device)

        for j, test_exp in enumerate(test_stream):
            ##################################################################
            # Accuracy on test exp j when model is trained on train exp i
            ##################################################################
            results = cl_strategy.eval(test_exp)
            conf_matrix[i, j] = results[f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00{j}"]

            ##################################################################
            # OOD score on test exp j when model is trained on train exp i
            ##################################################################
            set_all_seed(seed)
            test_loader = DataLoader(test_exp.dataset, batch_size=32, shuffle=True)
            ood_scores = torch.tensor([]).to(device)
            for x, y, *_ in test_loader:
                detector = cl_strategy.plugins[0].detector
                ood_score_k = detector.predict(x)
                # ood_score_k = detector.predict_features(torch.randn(y.shape[0], 10)*100 + 200)
                ood_scores = torch.cat((ood_scores, ood_score_k))
            ood_matrix[i, j] = ood_scores.mean()

    conf_matrix = np.round(conf_matrix, decimals=4)
    ood_matrix = np.round(ood_matrix, decimals=4)

    return conf_matrix, ood_matrix

def main():
    N_EXPERIENCES = 10
    EPOCHS = 1
    SEED = 0
    test_only = False

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_all_seed(SEED)

    ##################################################################
    # CL Benchmark Creation
    ##################################################################
    benchmark = SplitMNIST(n_experiences=N_EXPERIENCES,
                           shuffle=False, dataset_root='datasets')
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream


    ##################################################################
    # Model, optimizer, criterion
    ##################################################################
    model = SimpleMLP(num_classes=10, hidden_size=50)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()


    ##################################################################
    # OOD detectors
    ##################################################################
    # detector = OpenMax(cl_strategy.model, tailsize=25, alpha=5, euclid_weight=0.5)
    # detector = EnergyBased(model)
    detector_list = [
        OpenMax(model, tailsize=10, alpha=10, euclid_weight=0.5),
        EnergyBased(model),
        Entropy(model)
    ]
    detector_names = ['OpenMax', 'EnergyBased', 'Entropy']

    conf_matrix_list, ood_matrix_list = [], []
    for detector in detector_list:
        ##################################################################
        # CL Strategy
        ##################################################################
        replay = ReplayPlugin(mem_size=200)
        # replay = AGEMPlugin(patterns_per_experience=10, sample_size=100)
        detector_plugin = OODDetectorPlugin(detector)
        cl_strategy = Naive(
            model, optimizer, criterion, train_mb_size=32, train_epochs=EPOCHS,
            eval_mb_size=32, device=DEVICE,
            plugins=[detector_plugin, replay])
        # cl_strategy = Cumulative(
        #     model, optimizer, criterion, train_mb_size=32, train_epochs=EPOCHS,
        #     eval_mb_size=32, device=DEVICE,
        #     plugins=[detector_plugin])

        conf_matrix, ood_matrix = train_and_eval(train_stream, test_stream, cl_strategy, SEED, DEVICE)
        conf_matrix_list.append(conf_matrix)
        ood_matrix_list.append(ood_matrix)


    # Plot
    nrows, ncols = 1, len(detector_list) + 1
    fdim = 7
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols*fdim, nrows*fdim),
                            squeeze=False)

    titles = ['Accuracy'] + detector_names
    sns.heatmap(conf_matrix_list[0], annot=True, fmt='g', ax=axs[0, 0],
                cmap='summer', cbar=False, vmin=0, vmax=1,
                xticklabels=np.arange(N_EXPERIENCES), yticklabels=np.arange(N_EXPERIENCES))
    axs[0, 0].set_title(titles[0])
    axs[0, 0].axis('equal')
    axs[0, 0].set_xlabel("test exp id")
    axs[0, 0].set_ylabel("train exp id")

    for k, ood_matrix in enumerate(ood_matrix_list):
        col = k + 1
        sns.heatmap(ood_matrix, annot=True, fmt='g', ax=axs[0, col],
                    cmap='summer', cbar=False,
                    xticklabels=np.arange(N_EXPERIENCES), yticklabels=np.arange(N_EXPERIENCES))
        axs[0, col].set_title(titles[col])
        axs[0, col].axis('equal')
        axs[0, col].set_xlabel("test exp id")
        axs[0, col].set_ylabel("train exp id")

    fig.tight_layout()
    fig.show()
    fig.savefig('experiments/correct_.pdf')
    print("")

if __name__ == '__main__':
    main()

