from torchvision.transforms import Normalize
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


class InvNormalize(Normalize):
    def __init__(self, normalizer):
        inv_mean = [-mean / std for mean, std in list(zip(normalizer.mean, normalizer.std))]
        inv_std = [1 / std for std in normalizer.std]
        super().__init__(inv_mean, inv_std)

def _tensor_to_show(img, transforms=None):
    if transforms is not None:
        for transform in transforms.transforms:
            if isinstance(transform, Normalize):
                normalizer = transform
                break
        inverse_transform = InvNormalize(normalizer)
        img = inverse_transform(img)

    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


def imshow(img, ax=None, transforms=None, figsize=(10, 20), path=None):
    npimg = _tensor_to_show(img, transforms)
    if ax is None:
        plt.figure(figsize=figsize)
        plt.imshow(npimg, interpolation=None)
    else:
        ax.imshow(npimg, interpolation=None)
    if path is not None:
        plt.savefig(path)


def show_batch(x, transforms=None, figsize=(10, 20)):
    imshow(make_grid(x.cpu().detach(), nrow=5),
           transforms=transforms, figsize=figsize)
    plt.axis('off')
    plt.show()


def create_legend(ax, figsize=(10, 0.5)):
    # create legend
    h, l = ax.get_legend_handles_labels()
    legend_dict = dict(zip(l, h))
    legend_fig = plt.figure(figsize=figsize)

    legend_fig.legend(legend_dict.values(), legend_dict.keys(), loc='center',
                      ncol=len(legend_dict.values()), frameon=False)
    legend_fig.tight_layout()

    return legend_fig
