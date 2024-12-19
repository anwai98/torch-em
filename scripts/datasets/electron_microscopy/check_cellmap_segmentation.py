import os

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_cellmap_segmentation_loader


def check_cellmap_segmentation():
    ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

    loader = get_cellmap_segmentation_loader(
        path=os.path.join(ROOT, "cellmap"),
        patch_shape=(8, 256, 256),
        sampler=MinInstanceSampler(),
        batch_size=2,
        download=False,
    )

    check_loader(loader, 8)


if __name__ == "__main__":
    check_cellmap_segmentation()
