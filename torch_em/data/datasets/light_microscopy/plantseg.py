"""This dataset contains confocal and lightsheet microscopy images of plant cells
with annotations for cell and nucleus segmentation.

The dataset part of the publication https://doi.org/10.7554/eLife.57613.
Please cite it if you use this dataset in your research.
"""

import os
from glob import glob
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader

import torch_em

from .. import util


URLS = {
    "root": {
        "train": "https://files.de-1.osf.io/v1/resources/9x3g2/providers/osfstorage/?zip=",
        "val": "https://files.de-1.osf.io/v1/resources/vs6gb/providers/osfstorage/?zip=",
        "test": "https://files.de-1.osf.io/v1/resources/tn4xj/providers/osfstorage/?zip=",
    },
    "nuclei": {
        "train": "https://files.de-1.osf.io/v1/resources/thxzn/providers/osfstorage/?zip=",
    },
    "ovules": {
        "train": "https://files.de-1.osf.io/v1/resources/x9yns/providers/osfstorage/?zip=",
        "val": "https://files.de-1.osf.io/v1/resources/xp5uf/providers/osfstorage/?zip=",
        "test": "https://files.de-1.osf.io/v1/resources/8jz7e/providers/osfstorage/?zip=",
    }
}

# FIXME somehow the checksums are not reliably, this is a bit weird.
CHECKSUMS = {
    "root": {
        "train": None, "val": None, "test": None
        # "train": "f72e9525ff716ef14b70ab1318efd4bf303bbf9e0772bf2981a2db6e22a75794",
        # "val": "987280d9a56828c840e508422786431dcc3603e0ba4814aa06e7bf4424efcd9e",
        # "test": "ad71b8b9d20effba85fb5e1b42594ae35939d1a0cf905f3403789fc9e6afbc58",
    },
    "nuclei": {
        "train": None
        # "train": "9d19ddb61373e2a97effb6cf8bd8baae5f8a50f87024273070903ea8b1160396",
    },
    "ovules": {
        "train": None, "val": None, "test": None
        # "train": "70379673f1ab1866df6eb09d5ce11db7d3166d6d15b53a9c8b47376f04bae413",
        # "val": "872f516cb76879c30782d9a76d52df95236770a866f75365902c60c37b14fa36",
        # "test": "a7272f6ad1d765af6d121e20f436ac4f3609f1a90b1cb2346aa938d8c52800b9",
    }
}

CROPPING_VOLUMES = {
    # root (train)
    "Movie2_T00006_crop_gt.h5": slice(4, None),
    "Movie2_T00008_crop_gt.h5": slice(None, -18),
    "Movie2_T00010_crop_gt.h5": slice(None, -32),
    "Movie2_T00012_crop_gt.h5": slice(None, -39),
    "Movie2_T00014_crop_gt.h5": slice(None, -40),
    "Movie2_T00016_crop_gt.h5": slice(None, -42),
    # root (test)
    "Movie2_T00020_crop_gt.h5": slice(None, -50),
    # ovules (train)
    "N_487_ds2x.h5": slice(17, None),
    "N_535_ds2x.h5": slice(None, -1),
    "N_534_ds2x.h5": slice(None, -1),
    "N_451_ds2x.h5": slice(None, -1),
    "N_425_ds2x.h5": slice(None, -1),
    # ovules (val)
    "N_420_ds2x.h5": slice(None, -1),
}

# The resolution previous used for the resizing
# I have removed this feature since it was not reliable,
# but leaving this here for reference
# (also implementing resizing would be a good idea,
#  but more general and not for each dataset individually)
# NATIVE_RESOLUTION = (0.235, 0.075, 0.075)


def _fix_inconsistent_volumes(data_path, name, split):
    import h5py

    file_paths = glob(os.path.join(data_path, "*.h5"))
    if name not in ["root", "ovules"] and split not in ["train", "val"]:
        return

    for vol_path in tqdm(file_paths, desc="Fixing inconsistencies in volumes"):
        fname = os.path.basename(vol_path)

        # avoid duplicated volumes in 'train' and 'test'.
        if fname == "Movie1_t00045_crop_gt.h5" and (name == "root" and split == "train"):
            os.remove(vol_path)
            continue

        if fname not in CROPPING_VOLUMES:
            continue

        with h5py.File(vol_path, "r+") as f:
            raw, labels = f["raw"], f["label"]

            crop_slices = CROPPING_VOLUMES[fname]
            resized_raw, resized_labels = raw[:][crop_slices], labels[:][crop_slices]

            cropped_shape = resized_raw.shape
            raw.resize(cropped_shape)
            labels.resize(cropped_shape)

            raw[...] = resized_raw
            labels[...] = resized_labels


def get_plantseg_data(path: Union[os.PathLike, str], name: str, split: str, download: bool = False) -> str:
    """Download the PlantSeg training data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the data to load. Either 'root', 'nuclei' or 'ovules'.
        split: The split to download. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        The filepath to the training data.
    """
    url = URLS[name][split]
    checksum = CHECKSUMS[name][split]
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, f"{name}_{split}")
    if os.path.exists(out_path):
        return out_path
    tmp_path = os.path.join(path, f"{name}_{split}.zip")
    util.download_source(tmp_path, url, download, checksum)
    util.unzip(tmp_path, out_path, remove=True)
    _fix_inconsistent_volumes(out_path, name, split)
    return out_path


def get_plantseg_paths(
    path: Union[os.PathLike, str],
    name: str,
    split: str,
    download: bool = False
) -> List[str]:
    """Get paths to the PlantSeg data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the data to load. Either 'root', 'nuclei' or 'ovules'.
        split: The split to download. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the data.
    """
    data_path = get_plantseg_data(path, name, split, download)
    file_paths = sorted(glob(os.path.join(data_path, "*.h5")))
    return file_paths


def get_plantseg_dataset(
    path: Union[os.PathLike, str],
    name: str,
    split: str,
    patch_shape: Tuple[int, int, int],
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> Dataset:
    """Get the PlantSeg dataset for segmenting nuclei or cells.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the data to load. Either 'root', 'nuclei' or 'ovules'.
        split: The split to download. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    assert len(patch_shape) == 3

    file_paths = get_plantseg_paths(path, name, split, download)

    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=binary, binary=binary, boundaries=boundaries,
        offsets=offsets, binary_is_exclusive=False
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=file_paths,
        raw_key="raw",
        label_paths=file_paths,
        label_key="label",
        patch_shape=patch_shape,
        **kwargs
    )


# TODO add support for ignore label, key: "/label_with_ignore"
def get_plantseg_loader(
    path: Union[os.PathLike, str],
    name: str,
    split: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    **kwargs,
) -> DataLoader:
    """Get the PlantSeg dataloader for segmenting nuclei or cells.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        name: The name of the data to load. Either 'root', 'nuclei' or 'ovules'.
        split: The split to download. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
       The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_plantseg_dataset(
        path, name, split, patch_shape, download=download, offsets=offsets,
        boundaries=boundaries, binary=binary, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
