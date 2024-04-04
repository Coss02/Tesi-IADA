"""
OpenMax
==============================

:class:`OpenMax <pytorch_ood.detector.OpenMax>` was originally proposed
for Open Set Recognition but can be adapted for Out-of-Distribution tasks.

.. warning:: OpenMax requires ``libmr`` to be installed, which is broken at the moment. You can only use it
   by installing ``cython`` and ``numpy``, and ``libmr`` manually afterwards.


"""
import torch.cuda
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST

from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import OpenMax
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

from utils.fm import my_load, my_save
from tqdm import tqdm

fix_random_seed(123)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# %%
# Setup preprocessing and data
trans = WideResNet.transform_for("cifar10-pt")
from torchvision.transforms import Compose
out_trans = Compose(trans.transforms[:-1])
dataset_train = CIFAR10(root="datasets", train=True, download=True, transform=trans)
dataset_in_test = CIFAR10(root="datasets", train=False, download=True, transform=trans)
dataset_out_test = CIFAR10(root="datasets", train=False, download=True, transform=out_trans, target_transform=ToUnknown())

train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)

# create data loaders
test_iid_loader = DataLoader(dataset_in_test, batch_size=128)
test_ood_loader = DataLoader(dataset_out_test, batch_size=128)
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128)

# %%
# Stage 1: Create DNN pre-trained on CIFAR 10
model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device).eval()

# # %%
# # Stage 2: Create and Fit OpenMax
# detector = OpenMax(model, tailsize=25, alpha=5, euclid_weight=0.5)
# detector.fit(train_loader, device=device)
# my_save(detector, 'models/openmax_detector_cifar10.gz')

detector = my_load('models/openmax_detector_cifar10.gz')

ood_scores = torch.tensor([])
for x, y in tqdm(test_iid_loader):
    ood_score = detector(x.to(device))
    ood_scores = torch.cat((ood_scores, ood_score))
print(f"OOD mean score, IID test: {ood_scores.mean()} +/- {ood_scores.std()}")

ood_scores = torch.tensor([])
for x, y in tqdm(test_ood_loader):
    ood_score = detector(x.to(device))
    ood_scores = torch.cat((ood_scores, ood_score))
print(f"OOD mean score, OOD test: {ood_scores.mean()} +/- {ood_scores.std()}")

# Stage 3: Evaluate Detectors
metrics = OODMetrics()

for x, y in tqdm(test_loader):
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())
