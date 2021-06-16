import torch_em
from torch_em.model import AnisotropicUNet
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask
from torch_em.data.datasets import get_platynereis_nuclei_loader
from torch_em.util import parser_helper

OFFSETS = [
    [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    [-4, 0, 0], [0, -4, 0], [0, 0, -4],
    [-8, 0, 0], [0, -8, 0], [0, 0, -8],
    [-16, 0, 0], [0, -16, 0], [0, 0, -16]
]


def get_model():
    model = AnisotropicUNet(
        scale_factors=4*[[2, 2, 2]],
        in_channels=1,
        out_channels=len(OFFSETS) + 1,
        initial_features=32,
        gain=2,
        final_activation='Sigmoid'
    )
    return model


def get_loader(path, is_train, n_samples):
    batch_size = 1
    patch_shape = [48, 256, 256]
    if is_train:
        sample_ids = [1, 3, 6, 7, 8, 9, 10, 11, 12]
    else:
        sample_ids = [2, 4]
    loader = get_platynereis_nuclei_loader(
        path, patch_shape, sample_ids,
        offsets=OFFSETS,
        batch_size=batch_size,
        n_samples=n_samples,
        download=True,
        shuffle=True,
        num_workers=8*batch_size,
    )
    return loader


def train_affinities(args):
    model = get_model()
    train_loader = get_loader(args.input, True, n_samples=1000)
    val_loader = get_loader(args.input, False, n_samples=100)
    loss = LossWrapper(loss=DiceLoss(), transform=ApplyAndRemoveMask())

    name = "affinity_model"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=50,
        optimizer_kwargs={"weight_decay": 0.0005}
    )

    if args.from_checkpoint:
        trainer.fit(args.n_iterations, 'latest')
    else:
        trainer.fit(args.n_iterations)


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    train_affinities(args)