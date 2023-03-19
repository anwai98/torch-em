import os
import unittest
from shutil import rmtree

import h5py
import numpy as np
import torch

from torch_em import default_segmentation_loader
from torch_em.loss import DiceLoss
from torch_em.model import UNet2d
from torch_em.trainer import DefaultTrainer

try:
    import onnx
except ImportError:
    onnx = None


class ExpandChannels:
    def __init__(self, n_channels):
        self.n_channels = n_channels

    def __call__(self, labels):
        labels = np.concatenate([labels[None]] * self.n_channels, axis=0)
        return labels


class TestModelzoo(unittest.TestCase):
    data_path = "./data.h5"
    checkpoint_folder = "./checkpoints"
    save_folder = "./zoo_export"
    name = "test"

    def setUp(self):
        shape = (8, 128, 128)
        chunks = (1, 128, 128)
        with h5py.File(self.data_path, "w") as f:
            f.create_dataset("raw", data=np.random.rand(*shape),
                             chunks=chunks)
            f.create_dataset("labels", data=np.random.randint(0, 32, size=shape),
                             chunks=chunks)

    def tearDown(self):
        if os.path.exists(self.checkpoint_folder):
            rmtree(self.checkpoint_folder)
        if os.path.exists(self.save_folder):
            rmtree(self.save_folder)
        if os.path.exists(self.data_path):
            os.remove(self.data_path)

    def _create_checkpoint(self, n_channels):
        if n_channels > 1:
            label_transform = ExpandChannels(n_channels)
        else:
            label_transform = None
        label_transform = label_transform

        loader = default_segmentation_loader(
            raw_paths=self.data_path, raw_key="raw",
            label_paths=self.data_path, label_key="labels",
            batch_size=1, patch_shape=(1, 128, 128), ndim=2,
            label_transform=label_transform,
        )
        model = UNet2d(in_channels=1, out_channels=n_channels,
                       depth=2, initial_features=4, norm="BatchNorm")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        trainer = DefaultTrainer(
            name=self.name, train_loader=loader, val_loader=loader,
            model=model, loss=DiceLoss(), metric=DiceLoss(),
            optimizer=optimizer, device=torch.device("cpu"),
            mixed_precision=False, logger=None, compile_model=False,
        )
        trainer.fit(10)

    def _test_export(self, n_channels):
        from torch_em.util.modelzoo import export_bioimageio_model
        self._create_checkpoint(n_channels)

        success = export_bioimageio_model(
            os.path.join(self.checkpoint_folder, self.name),
            self.save_folder,
            input_data=np.random.rand(128, 128).astype("float32"),
            input_optional_parameters=False

        )
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.save_folder))
        self.assertTrue(os.path.exists(os.path.join(self.save_folder, "rdf.yaml")))

    def test_export_single_channel(self):
        self._test_export(1)

    def test_export_multi_channel(self):
        self._test_export(4)

    def test_add_weights_torchscript(self):
        from torch_em.util.modelzoo import add_weight_formats
        self._test_export(1)
        additional_formats = ["torchscript"]
        add_weight_formats(self.save_folder, additional_formats)
        self.assertTrue(os.path.join(self.save_folder, "weigths-torchscript.pt"))

    @unittest.skipIf(onnx is None, "Needs onnx")
    def test_add_weights_onnx(self):
        from torch_em.util.modelzoo import add_weight_formats
        self._test_export(1)
        additional_formats = ["onnx"]
        add_weight_formats(self.save_folder, additional_formats)
        self.assertTrue(os.path.join(self.save_folder, "weigths.onnx"))


if __name__ == "__main__":
    unittest.main()
