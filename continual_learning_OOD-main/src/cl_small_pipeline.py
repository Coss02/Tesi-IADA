import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from avalanche.benchmarks.classic import SplitMNIST
# from utils.data import MySplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC, ClassBalancedBuffer

from utils.fm import set_all_seed

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.fm import init_logger, my_load, my_save
from utils.viz import show_batch
from time import time

# Config
CL_STRATEGY_DICT = {
    'Naive': {'class': Naive,
              'hparams': {}
              },

    'Cumulative': {'class': Cumulative,
                   'hparams': {}
                   },

    'EWC': {'class': EWC,
            'hparams': {'ewc_lambda': 1000}
            },

    'Replay': {'class': Replay,
               'hparams': {'mem_size': 100}
               },

    'LwF': {'class': LwF,
            'hparams': {'alpha': 1,
            'temperature': 2}},

}



def main():
    # cl_strategy_names = ['Naive', 'Replay', 'EWC', 'LwF']
    cl_strategy_names = ('Naive',)
    N_EXPERIENCES = 5
    SEED = 0
    test_only = False

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = init_logger(root='experiments')

    # CL Benchmark Creation
    benchmark = SplitMNIST(n_experiences=N_EXPERIENCES,
                           shuffle=False, dataset_root='datasets')
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    if not test_only:
        for cl_id, cl_strategy_name in enumerate(cl_strategy_names):
            # Retrieve CL strategy information
            cl_strategy_class = CL_STRATEGY_DICT[cl_strategy_name]['class']
            cl_strategy_hparams = CL_STRATEGY_DICT[cl_strategy_name]['hparams']

            logger.info(f"CL strategy: {cl_strategy_name}, hparams: {cl_strategy_hparams}")
            set_all_seed(SEED)



            # model
            model = SimpleMLP(num_classes=10, hidden_size=50)

            # Prepare for training & testing
            optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
            criterion = CrossEntropyLoss()

            cl_strategy = cl_strategy_class(
                model, optimizer, criterion, train_mb_size=32, train_epochs=1,
                eval_mb_size=32, device=DEVICE,
                **cl_strategy_hparams)

            start = time()
            # train stream loop, after every train exp test on all the test exps
            results_list = []
            for train_exp in train_stream:
                tr_loader = DataLoader(train_exp.dataset, batch_size=10, shuffle=True)
                x, y, _ = next(iter(tr_loader))
                show_batch(x)
                cl_strategy.train(train_exp)
                results = cl_strategy.eval(test_stream)
                results_list.append(results)
            end = time()
            exe_time = end - start

            # todo: salvare anche eventuali iperparametri, epochs, batch size, seed, modello ecc
            my_save(results_list, f"experiments/{cl_strategy_name}_results.gz")
            logger.debug(f"This run took {exe_time} seconds.")

    # Plot
    nrows, ncols = 1, len(cl_strategy_names)
    fdim = 5
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols*fdim, nrows*fdim),
                            squeeze=False)
    for cl_id, cl_strategy_name in enumerate(cl_strategy_names):
        cl_strategy_hparams = CL_STRATEGY_DICT[cl_strategy_name]['hparams']
        results_list = my_load(f"experiments/{cl_strategy_name}_results.gz")
        # Gather results in a matrix
        conf_matrix = np.zeros(shape=(N_EXPERIENCES, N_EXPERIENCES))
        for i, results in enumerate(results_list):
            for j in range(N_EXPERIENCES):
                experience_id = j
                conf_matrix[i, j] = results[f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00{experience_id}"]
        conf_matrix = np.round(conf_matrix*100, decimals=2)
        sns.heatmap(conf_matrix, annot=True, fmt='g', ax=axs[0, cl_id],
                        cmap='summer', cbar=False, vmin=0, vmax=100,
                        xticklabels=[], yticklabels=[])
        axs[0, cl_id].set_title(f"{cl_strategy_name}, {cl_strategy_hparams}")

    fig.tight_layout()
    fig.show()
    fig.savefig('experiments/conf_matrix.pdf')

if __name__ == '__main__':
    main()

