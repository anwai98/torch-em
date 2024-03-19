import os
import argparse
import numpy as np

import torch
import torch_em
from torch_em.model import UNETR
from torch_em.data.datasets import get_cremi_loader
from torch_em.data import MinInstanceSampler
from torch_em.loss import DiceLoss, LossWrapper, ApplyAndRemoveMask, DiceBasedDistanceLoss


ROOT = "/scratch/usr/nimanwai"

OFFSETS = [
    [-1, 0], [0, -1],
    [-3, 0], [0, -3],
    [-9, 0], [0, -9],
    [-27, 0], [0, -27]
]

MODELS = {
    "vit_t": "/scratch/usr/nimanwai/models/segment-anything/checkpoints/vit_t_mobile_sam.pth",
    "vit_b": "/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
}


def get_loaders(args, patch_shape=(1, 512, 512)):
    if args.distances:
        label_trafo = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True
        )
    else:
        label_trafo = None

    train_rois = {"A": np.s_[0:75, :, :], "B": np.s_[0:75, :, :], "C": np.s_[0:75, :, :]}
    val_rois = {"A": np.s_[75:100, :, :], "B": np.s_[75:100, :, :], "C": np.s_[75:100, :, :]}

    sampler = MinInstanceSampler()

    train_loader = get_cremi_loader(
        path=args.input,
        patch_shape=patch_shape,
        batch_size=2,
        label_transform=label_trafo,
        rois=train_rois,
        sampler=sampler,
        ndim=2,
        label_dtype=torch.float32,
        defect_augmentation_kwargs=None,
        boundaries=args.boundaries,
        offsets=OFFSETS if args.affinities else None,
        num_workers=16,
    )
    val_loader = get_cremi_loader(
        path=args.input,
        patch_shape=patch_shape,
        batch_size=1,
        label_transform=label_trafo,
        rois=val_rois,
        sampler=sampler,
        ndim=2,
        label_dtype=torch.float32,
        defect_augmentation_kwargs=None,
        boundaries=args.boundaries,
        offsets=OFFSETS if args.affinities else None,
        num_workers=16,
    )

    return train_loader, val_loader


def get_output_channels(args):
    if args.boundaries:
        output_channels = 1
    elif args.distances:
        output_channels = 3
    elif args.affinities:
        output_channels = len(OFFSETS)

    return output_channels


def get_loss_function(args):
    if args.affinities:
        loss = LossWrapper(
            loss=DiceLoss(),
            transform=ApplyAndRemoveMask(masking_method="multiply")
        )
    elif args.distances:
        loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)

    else:
        loss = DiceLoss()

    return loss


def get_save_root(args):
    # experiment_type
    if args.boundaries:
        experiment_type = "boundaries"
    elif args.affinities:
        experiment_type = "affinities"
    elif args.distances:
        experiment_type = "distances"
    else:
        raise ValueError

    model_name = args.model_type

    # saving the model checkpoints
    save_root = os.path.join(
        args.save_root, "pretrained" if args.pretrained else "scratch", experiment_type, model_name
    )
    return save_root


def run_cremi_unetr_training(args, device):
    # the dataloaders for cremi dataset
    train_loader, val_loader = get_loaders(args)

    output_channels = get_output_channels(args)

    # the UNETR model
    model = UNETR(
        encoder=args.model_type,
        out_channels=output_channels,
        use_sam_stats=args.pretrained,
        encoder_checkpoint=MODELS[args.model_type] if args.pretrained else None,
        final_activation="Sigmoid"
    )
    model.to(device)

    save_root = get_save_root(args)

    # loss function
    loss = get_loss_function(args)

    trainer = torch_em.default_segmentation_trainer(
        name="cremi-unetr",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False
    )

    trainer.fit(args.iterations)


def run_cremi_unetr_inference(args):
    raise NotImplementedError


def main(args):
    assert (args.boundaries + args.affinities + args.distances) == 1

    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        run_cremi_unetr_training(args, device)

    if args.predict:
        run_cremi_unetr_inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=os.path.join(ROOT, "data", "cremi"))
    parser.add_argument("--iterations", type=int, default=int(1e5))
    parser.add_argument("-s", "--save_root", type=str, default=os.path.join(ROOT, "experiments", "vimunet"))
    parser.add_argument("-m", "--model_type", type=str, default="vit_b")

    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")

    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--affinities", action="store_true")
    parser.add_argument("--distances", action="store_true")

    args = parser.parse_args()
    main(args)
