"""
"""

import os

import torch_em

from .. import util


CHOICES = [
    'jrc_cos7-1a', 'jrc_cos7-1b', 'jrc_ctl-id8-1', 'jrc_fly-mb-1a', 'jrc_fly-vnc-1', 'jrc_hela-2',
    'jrc_hela-3', 'jrc_jurkat-1', 'jrc_macrophage-2', 'jrc_mus-heart-1', 'jrc_mus-kidney', 'jrc_mus-kidney-3',
    'jrc_mus-kidney-glomerulus-2', 'jrc_mus-liver', 'jrc_mus-liver-3', 'jrc_mus-liver-zon-1', 'jrc_mus-liver-zon-2',
    'jrc_mus-nacc-1', 'jrc_sum159-1', 'jrc_sum159-4', 'jrc_ut21-1413-003', 'jrc_zf-cardiac-1'
]


def get_cellmap_segmentation_paths(path, download):
    for choice in CHOICES:
        volume_path = os.path.join(path, choice, f"{choice}.zarr")
        assert os.path.exists(volume_path), volume_path

        import zarr
        with zarr.open(volume_path, "r") as f:
            breakpoint()
            raw = f['recon-1/em/fibsem-uint8']
            labels = f['recon-1/labels/groundtruth']

            breakpoint()

        breakpoint()

    return volume_paths


def get_cellmap_segmentation_dataset(path, patch_shape, download, **kwargs):
    volume_paths = get_cellmap_segmentation_paths(path, download)

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="",
        label_paths=volume_paths,
        label_key="",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        **kwargs
    )


def get_cellmap_segmentation_loader(path, batch_size, patch_shape, download, **kwargs):
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cellmap_segmentation_dataset(path, patch_shape, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
