import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from avalanche.benchmarks.datasets import MNIST
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset


from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.models import SimpleMLP
from avalanche.training import Naive, Cumulative, ICaRL, EWC, Replay
from avalanche.training.plugins import (
    ReplayPlugin,
    EWCPlugin,
    AGEMPlugin,
    EvaluationPlugin,
)
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics,
    confusion_matrix_metrics,
    disk_usage_metrics,
)
from avalanche.logging import InteractiveLogger

from pytorch_ood.detector import OpenMax, EnergyBased, Entropy
from pytorch_ood.utils import OODMetrics

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy

from utils.fm import init_logger, set_all_seed
from utils.viz import show_batch
from utils.data import (
    count_sample_in_experience,
    count_sample_in_dataset,
    count_sample_in_stream,
    get_full_dataset_experience,
)

# mpl.rcParams['mathtext.fontset'] = 'stix'
# mpl.rcParams['font.size'] = 18
# mpl.rcParams['font.family'] = 'STIXGeneral'
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['mathtext.fontset'] = 'stix'

from avalanche_ood_plugins.detectors import OODDetectorPlugin


def load_corrupted_mnist(seed):
    test_images = np.load("./datasets/brightness02/test/test_images.npy")
    test_labels = np.load("./datasets/brightness02/test/test_labels.npy")
    train_images = np.load("./datasets/brightness02/train/train_images.npy")
    train_labels = np.load("./datasets/brightness02/train/train_labels.npy")

    # Convert NumPy arrays to PyTorch tensors
    test_images_tensor = torch.tensor(test_images)
    test_labels_tensor = torch.tensor(test_labels)
    train_images_tensor = torch.tensor(train_images)
    train_labels_tensor = torch.tensor(train_labels)

    # Create TensorDataset
    test = TensorDataset(test_images_tensor, test_labels_tensor)
    train = TensorDataset(train_images_tensor, train_labels_tensor)

    scenario = nc_benchmark(
        train,
        test,
        n_experiences=10,
        shuffle=False,
        seed=seed,
        task_labels=False,
    )
    return scenario


def load_data(batch_size=64, seed = 0):
    # Location to save/load the MNIST dataset
    datadir = default_dataset_location("mnist")

    # Load the non-corrupted MNIST dataset
    train_MNIST = MNIST(datadir, train=True, download=True)
    test_MNIST = MNIST(datadir, train=False, download=True)

    # Extract train and test data/labels
    train_data = train_MNIST.data.float() / 255  # Normalize data to [0, 1]
    train_labels = train_MNIST.targets
    test_data = test_MNIST.data.float() / 255  # Normalize data to [0, 1]
    test_labels = test_MNIST.targets

    # Load corrupted data and labels
    c_test_images = (
        np.load("./datasets/brightness02/test/test_images.npy").astype(np.float32) / 255
    )  # Normalize
    c_test_labels = np.load("./datasets/brightness02/test/test_labels.npy")
    # c_train_images = (
    #     np.load("./brightness/train/train_images.npy").astype(np.float32) / 255
    # )  # Normalize
    # c_train_labels = np.load("./brightness/train/train_labels.npy")

    # Convert NumPy arrays to tensors and remove channel dimension for images
    c_test_images_tensor = remove_channel_dimension(torch.tensor(c_test_images))
    # c_train_images_tensor = remove_channel_dimension(torch.tensor(c_train_images))

    # Apply the specified mapping to the corrupted labels
    def map_labels(labels):
        return torch.tensor([10 if label == 0 else label+10 for label in labels])

    c_test_labels_tensor = map_labels(c_test_labels)
    # c_train_labels_tensor = map_labels(c_train_labels)

    # Combine non-corrupted and corrupted data
    combined_test_data = torch.cat(
        [test_data, c_test_images_tensor], dim=0
    )  # Add channel dimension
    combined_test_labels = torch.cat([test_labels, c_test_labels_tensor], dim=0)

    # combined_train_data = torch.cat(
    #     [train_data, c_train_images_tensor], dim=0
    # )  # Add channel dimension
    # combined_train_labels = torch.cat([train_labels, c_train_labels_tensor], dim=0)

    # Create TensorDataset objects
    # train_dataset = TensorDataset(combined_train_data, combined_train_labels)

    combined_test_dataset = TensorDataset(combined_test_data, combined_test_labels)

    # Create a train Dataset
    train_dataset = TensorDataset(train_data, train_labels)
    # Create DataLoader objects
    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataLoader = DataLoader(
        combined_test_dataset, batch_size=batch_size, shuffle=False
    )

    # Set the desired order of the classes
    desired_order = [0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 19]

    # Create the CL scenario
    scenario = nc_benchmark(
        combined_test_dataset,
        combined_test_dataset,
        n_experiences=10,
        shuffle=False,
        seed=seed,
        fixed_class_order=desired_order,
        task_labels=True,
    )

    return train_dataLoader, test_dataLoader, scenario 


def map_mixed_test(labels):
    return torch.tensor([-1 if label > 9 else label for label in labels])

def map_normal_test(labels, i, j):
    return torch.tensor([-1 if j > i else label for label in labels])


def train_and_eval(
    train_stream, normal_test_stream, mixed_test_stream , cl_strategy, seed=0, device="cpu"):
    N_EXPERIENCES = len(train_stream)

    set_all_seed(seed)
    acc_k_matrix = np.zeros(shape=(N_EXPERIENCES, 1))  # Accuracy K Matrix
    forgetting_k_matrix = np.zeros(shape=(N_EXPERIENCES, 1))  # Forgetting K Matrix
    acc_ij_matrix = np.zeros(
        shape=(N_EXPERIENCES, N_EXPERIENCES)
    )  # Accuracy I,J Matrix
    forgetting_ij_matrix = np.zeros(
        shape=(N_EXPERIENCES, N_EXPERIENCES)
    )  # Forgetting I,J Matrix
    ood_matrix = np.zeros(shape=(N_EXPERIENCES, N_EXPERIENCES))  # OOD Matrix

    OOD_AUROC_matrix = np.zeros(shape=(N_EXPERIENCES,N_EXPERIENCES)) # OOD AUROC Matrix
    OOD_AUPR_OUT_matrix = np.zeros(shape=(N_EXPERIENCES,N_EXPERIENCES)) # OOD AUPR-OUT Matrix
    OOD_FPR95TPR_matrix = np.zeros(shape=(N_EXPERIENCES,N_EXPERIENCES)) # OOD FPR95TPR Matrix
    OOD_AUPR_IN_matrix = np.zeros(shape=(N_EXPERIENCES,N_EXPERIENCES)) # OOD AUPR-IN Matrix
    IID_AUPR_OUT_matrix = np.zeros(shape=(N_EXPERIENCES,N_EXPERIENCES)) # IID AUPR-OUT Matrix
    IID_FPR95TPR_matrix = np.zeros(shape=(N_EXPERIENCES,N_EXPERIENCES)) # IID FPR95TPR Matrix
    IID_AUPR_IN_matrix = np.zeros(shape=(N_EXPERIENCES,N_EXPERIENCES)) # IID AUPR-IN Matrix
    IID_AUROC_matrix = np.zeros(shape=(N_EXPERIENCES,N_EXPERIENCES)) # IID AUROC Matrix

    OOD_metrics_large = OODMetrics()
    IID_metrics_large = OODMetrics()

    for i, train_exp in enumerate(train_stream):
        set_all_seed(seed)
        cl_strategy.train(train_exp)
        results = cl_strategy.eval(normal_test_stream)
        acc_k_matrix[i, 0] = results[
            f"Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000"
        ]
        forgetting_k_matrix[i, 0] = results[f"StreamForgetting/eval_phase/test_stream"]
        print(results)

        # Evaluation and OOD Scores on IID Data
        IID_metrics_narrow = OODMetrics()
        for j, test_exp in enumerate(normal_test_stream):
            acc_ij_matrix[i, j] = results[
                f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{str(j).zfill(3)}"
            ]
            if j < i:
                forgetting_ij_matrix[i, j] = results[
                    f"ExperienceForgetting/eval_phase/test_stream/Task000/Exp{str(j).zfill(3)}"
                ]

            # OOD Detection
            set_all_seed(seed)
            test_loader = DataLoader(test_exp.dataset, batch_size=32, shuffle=True)
            ood_scores = torch.tensor([]).to(device)
            # OOD Scores computing
            for batch in test_loader:
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, y, *_ = batch

                for plugin in cl_strategy.plugins:
                    if isinstance(plugin, OODDetectorPlugin):
                        detector = plugin.detector
                    y = map_normal_test(y,i,j)
                    ood_score_k = detector.predict(x)
                    ood_scores = torch.cat((ood_scores, ood_score_k))
                    IID_metrics_narrow.update(detector(x),y)
                    IID_metrics_large.update(detector(x),y)

            ood_matrix[i, j] = ood_scores.mean()
            if (j > i):
                dict_IID_M_narrow = IID_metrics_narrow.compute()
                IID_AUROC_matrix[i, j] = dict_IID_M_narrow['AUROC']
                IID_AUPR_IN_matrix[i, j] = dict_IID_M_narrow['AUPR-IN']
                IID_AUPR_OUT_matrix[i, j] = dict_IID_M_narrow['AUPR-OUT']
                IID_FPR95TPR_matrix[i, j] = dict_IID_M_narrow["FPR95TPR"]

        for z, c_test_exp in enumerate(mixed_test_stream):
            set_all_seed(seed)
            test_loader = DataLoader(c_test_exp.dataset, batch_size=32, shuffle=True)
            OOD_metrics_narrow = OODMetrics()

            for batch in test_loader:
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, y, *_ = batch

                for plugin in cl_strategy.plugins:
                    if isinstance(plugin, OODDetectorPlugin):
                        detector = plugin.detector
                    y = map_mixed_test(y)
                    OOD_metrics_narrow.update(detector(x), y)
                    OOD_metrics_large.update(detector(x), y)
            dict_OOD_M_narrow = OOD_metrics_narrow.compute()
            OOD_AUROC_matrix[i, z] = dict_OOD_M_narrow['AUROC']
            OOD_AUPR_IN_matrix[i, z] = dict_OOD_M_narrow['AUPR-IN']
            OOD_AUPR_OUT_matrix[i, z] = dict_OOD_M_narrow['AUPR-OUT']
            OOD_FPR95TPR_matrix[i, z] = dict_OOD_M_narrow["FPR95TPR"]

    print("Future IID OOD metrics ", IID_metrics_large.compute())
    print("Corrupted images OOD metrics ",OOD_metrics_large.compute())
    IID_AUROC_matrix = np.round(IID_AUROC_matrix, decimals=4)
    OOD_AUROC_matrix = np.round(OOD_AUROC_matrix, decimals=4)
    acc_ij_matrix = np.round(acc_ij_matrix, decimals=4)
    ood_matrix = np.round(ood_matrix, decimals=4)
    acc_k_matrix = np.round(acc_k_matrix, decimals=4)
    forgetting_k_matrix = np.round(forgetting_k_matrix, decimals=4)
    forgetting_ij_matrix = np.round(forgetting_ij_matrix, decimals=4)

    return (
        acc_ij_matrix,
        ood_matrix,
        acc_k_matrix,
        forgetting_k_matrix,
        forgetting_ij_matrix,
        OOD_AUROC_matrix,
        OOD_AUPR_IN_matrix,
        OOD_AUPR_OUT_matrix,
        OOD_FPR95TPR_matrix,
        IID_AUROC_matrix,
        IID_AUPR_IN_matrix,
        IID_AUPR_OUT_matrix,
        IID_FPR95TPR_matrix,
    )


def remove_channel_dimension(dataset):
    modified_data = []
    for data in dataset:
        modified_data.append(data.squeeze(-1))  # Remove the last channel dimension
    return torch.stack(modified_data)  # Stack the modified data and return as a tensor


def load_complete_dataset(seed=0):
    datadir = default_dataset_location("mnist")
    train_MNIST = MNIST(datadir, train=True, download=False)
    test_MNIST = MNIST(datadir, train=False, download=False)

    # Extracting train data/labels and test data/labels
    train_data = train_MNIST.data
    train_labels = train_MNIST.targets
    test_data = test_MNIST.data
    test_labels = test_MNIST.targets

    c_test_images = np.load("./datasets/brightness02/test/test_images.npy")
    c_test_labels = np.load("./datasets/brightness02/test/test_labels.npy")
    c_train_images = np.load("./datasets/brightness02/train/train_images.npy")
    c_train_labels = np.load("./datasets/brightness02/train/train_labels.npy")

    # Convert NumPy arrays to PyTorch tensors
    test_images_tensor = torch.tensor(test_data)
    test_labels_tensor = torch.tensor(test_labels)
    train_images_tensor = torch.tensor(train_data)
    train_labels_tensor = torch.tensor(train_labels)

    c_test_images_tensor = torch.tensor(c_test_images)
    c_test_labels_tensor = torch.tensor(c_test_labels)
    c_train_images_tensor = torch.tensor(c_train_images)
    c_train_labels_tensor = torch.tensor(c_train_labels)

    c_test_images_tensor = remove_channel_dimension(c_test_images_tensor)
    c_train_images_tensor = remove_channel_dimension(c_train_images_tensor)

    combined_train_data = torch.cat([train_images_tensor, c_train_images_tensor], dim=0)
    combined_train_labels = torch.cat(
        [train_labels_tensor, c_train_labels_tensor], dim=0
    )
    combined_test_data = torch.cat([test_images_tensor, c_test_images_tensor], dim=0)
    combined_test_labels = torch.cat([test_labels_tensor, c_test_labels_tensor], dim=0)

    combined_train = TensorDataset(combined_train_data, combined_train_labels)
    combined_test = TensorDataset(combined_test_data, combined_test_labels)

    print(combined_test)

    scenario = nc_benchmark(
        combined_train,
        combined_test,
        n_experiences=10,
        shuffle=False,
        seed=seed,
        task_labels=False,
    )
    return scenario


def main():
    N_EXPERIENCES = 10
    EPOCHS = 1
    SEED = 0

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_all_seed(SEED)

    ##################################################################
    # CL Benchmark Creation
    ##################################################################
    benchmark = SplitMNIST(
        n_experiences=N_EXPERIENCES, shuffle=False, dataset_root="datasets"
    )
    print(benchmark.original_classes_in_exp)
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    ##################################################################
    # load Train DataLoader (IID) Test DataLoader (IID+OOD) Scenario (contains only the test_stream, the train one is just a copy of the test)
    ##################################################################
    normal_train_dataLoader, combined_test_dataLoader, combined_scenario = load_data()

    ##################################################################
    # Model, optimizer, criterion
    ##################################################################
    model = SimpleMLP(num_classes=10, hidden_size=50)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    ##################################################################
    # Instantiate CL and OOD methods as Avalanche plugins
    ##################################################################
    # instantiate CL strategy plugin
    replay = ReplayPlugin(
        mem_size=50
    )  # classic Avalanche plugin that implements Replay strategy
    # ewc = EWCPlugin(ewc_lambda=1)

    # instantiate OOD strategy plugin
    detector = OpenMax(model, tailsize=50, alpha=10, euclid_weight=0.5)
    detector_plugin = OODDetectorPlugin(detector)
    ##################################################################
    # Mount everything in a single Avalanche CL strategy
    ##################################################################
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True, trained_experience=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )
    cl_strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=32,
        train_epochs=EPOCHS,
        eval_mb_size=32,
        device=DEVICE,
        plugins=[detector_plugin, replay],
        evaluator=eval_plugin,
    )

    (
        acc_ij_matrix,
        ood_matrix,
        acc_k_matrix,
        forgetting_k_matrix,
        forgetting_ij_matrix,
        OOD_AUROC_matrix,
        OOD_AUPR_IN_matrix,
        OOD_AUPR_OUT_matrix,
        OOD_FPR95TPR_matrix,
        IID_AUROC_matrix,
        IID_AUPR_IN_matrix,
        IID_AUPR_OUT_matrix,
        IID_FPR95TPR_matrix,
    ) = train_and_eval(
        train_stream,
        test_stream,
        combined_scenario.test_stream,
        cl_strategy,
        SEED,
        DEVICE,
    )

    ##################################################################
    # Visualize results
    ##################################################################
    nrows, ncols = 1, 5
    fdim = 13
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * fdim, nrows * fdim), squeeze=False
    )
    
    # Now we create a new figure for the OOD and IID AUROC Matrices
    fig_ood, axs_ood = plt.subplots(nrows, 2, figsize=(2 * fdim, nrows * fdim), squeeze=False)
    titles_ood = ["OOD AUROC Matrix", "IID AUROC Matrix"]
    
    titles = [
        "Experience Accuracy",
        "OOD score",
        "Experience Forgetting",
        "Stream Accuracy",
        "Stream Forgetting",
    ]

    # Plot (0, 0)
    sns.heatmap(
        acc_ij_matrix,
        annot=True,
        fmt="g",
        ax=axs[0, 0],
        cmap="summer",
        cbar=False,
        vmin=0,
        vmax=1,
        xticklabels=np.arange(N_EXPERIENCES),
        yticklabels=np.arange(N_EXPERIENCES),
    )
    axs[0, 0].set_title(titles[0])
    axs[0, 0].axis("equal")
    axs[0, 0].set_xlabel("test exp id")
    axs[0, 0].set_ylabel("train exp id")

    # Plot (0, 1)
    sns.heatmap(
        ood_matrix,
        annot=True,
        fmt="g",
        ax=axs[0, 1],
        cmap="summer",
        cbar=False,
        xticklabels=np.arange(N_EXPERIENCES),
        yticklabels=np.arange(N_EXPERIENCES),
    )
    axs[0, 1].set_title(titles[1])
    axs[0, 1].axis("equal")
    axs[0, 1].set_xlabel("test exp id")
    axs[0, 1].set_ylabel("train exp id")

    # Plot (0, 2)
    sns.heatmap(
        forgetting_ij_matrix,
        annot=True,
        fmt="g",
        ax=axs[0, 2],
        cmap="summer",
        cbar=False,
        xticklabels=np.arange(N_EXPERIENCES),
        yticklabels=np.arange(N_EXPERIENCES),
    )
    axs[0, 2].set_title(titles[2])
    axs[0, 2].axis("equal")
    axs[0, 2].set_xlabel("test exp id")
    axs[0, 2].set_ylabel("train exp id")

    # Plot (0, 3)
    sns.heatmap(
        acc_k_matrix,
        annot=True,
        fmt="g",
        ax=axs[0, 3],
        cmap="summer",
        cbar=False,
        xticklabels=np.arange(N_EXPERIENCES),
    )
    axs[0, 3].set_title(titles[3])
    axs[0, 3].axis("equal")
    axs[0, 3].set_ylabel("train exp id")

    # Plot (0, 4)
    sns.heatmap(
        forgetting_k_matrix,
        annot=True,
        fmt="g",
        ax=axs[0, 4],
        cmap="summer",
        cbar=False,
        xticklabels=np.arange(N_EXPERIENCES),
    )
    axs[0, 4].set_title(titles[4])
    axs[0, 4].axis("equal")
    axs[0, 4].set_ylabel("train exp id")
    
    sns.heatmap(
        OOD_AUROC_matrix,
        annot=True,
        fmt="g",
        ax=axs_ood[0, 0],
        cmap="summer",
        cbar=False,
        xticklabels=np.arange(N_EXPERIENCES),
    )
    axs_ood[0, 0].set_title(titles_ood[0])
    axs_ood[0, 0].axis("equal")
    axs_ood[0, 0].set_ylabel("train exp id")
    axs_ood[0, 0].set_xlabel("Corrupted test exp id")
    
    sns.heatmap(
        IID_AUROC_matrix,
        annot=True,
        fmt="g",
        ax=axs_ood[0, 1],
        cmap="summer",
        cbar=False,
        xticklabels=np.arange(N_EXPERIENCES),
    )
    axs_ood[0, 1].set_title(titles_ood[1])
    axs_ood[0, 1].axis("equal")
    axs_ood[0, 1].set_ylabel("train exp id")
    axs_ood[0, 1].set_xlabel("test exp id")

    # Refine, show and save figure
    fig.tight_layout()
    fig.show()
    os.makedirs(
        "experiments", exist_ok=True
    )  # if experiments folder does not exist create it
    fig.savefig("experiments/results.pdf")
    
    fig_ood.tight_layout()
    fig_ood.show()
    fig_ood.savefig("experiments/OOD_metrics.pdf")
    
    
    print("")


if __name__ == "__main__":
    main()
