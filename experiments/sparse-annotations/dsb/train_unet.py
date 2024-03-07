import numpy as np

from skimage.segmentation import relabel_sequential

import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_dsb_loader


ROOT = "/home/anwai/data/dsb"


class SparseAnnotationsLabelTrafo:
    def __init__(
        self,
        n_instances: int
    ):
        self.n_instances = n_instances

    def __call__(
        self,
        labels: np.ndarray
    ):
        # compute the total number of instances (exclude the background)
        label_ids = np.unique(labels)[1:]

        # now, let's select n number of instance to make use of sparse annotation-based segmentation
        chosen_ids = np.random.choice(label_ids, size=min(len(label_ids), self.n_instances), replace=False)

        # let's get the chosen instances
        labels[~(np.isin(labels, chosen_ids))] = 0
        labels = relabel_sequential(labels)[0]

        return labels


def get_dataloaders():
    patch_shape = (1, 256, 256)
    label_transform = SparseAnnotationsLabelTrafo(n_instances=3)

    train_loader = get_dsb_loader(
        path=ROOT,
        split="train",
        patch_shape=patch_shape,
        batch_size=2,
        download=True,
        label_transform=label_transform
    )
    val_loader = get_dsb_loader(
        path=ROOT,
        split="test",
        patch_shape=patch_shape,
        batch_size=1,
        download=True
    )
    return train_loader, val_loader


def _train_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders()

    model = UNet2d(in_channels=1, out_channels=1, initial_features=64, final_activation="Sigmoid")

    loss = torch_em.loss.DiceLoss()

    trainer = torch_em.default_segmentation_trainer(
        name="dsb-sparse-annotations",
        save_root="/home/anwai/models/torch-em/sparse-annotations",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=device,
        mixed_precision=True,
        compile_model=False,
        log_image_interval=10
    )
    trainer.fit(iterations=int(1e4))


def main():
    _train_unet()


if __name__ == "__main__":
    main()
