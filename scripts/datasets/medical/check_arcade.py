from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.medical import get_arcade_loader


ROOT = "/media/anwai/ANWAI/data/arcade"


def check_arcade():
    loader = get_arcade_loader(
        path=ROOT,
        split="test",
        patch_shape=(256, 256),
        batch_size=2,
        download=True,
        task="syntax",
        resize_inputs=False,
        sampler=MinInstanceSampler(),
    )
    check_loader(loader, 8)


if __name__ == "__main__":
    check_arcade()