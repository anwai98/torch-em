from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unetr import SingleDeconv2DBlock
from .unet import ConvBlock2d, Upsampler2d
from .vit import get_vision_transformer, ViT_MAE, ViT_Sam

try:
    from micro_sam.util import get_sam_model
except ImportError:
    get_sam_model = None


class ExplicitDecoder(nn.Module):
    def __init__(
        self,
        features,
        scale_factors,
        conv_block_impl,
        sampler_impl,
        **conv_block_kwargs
    ):
        super().__init__()
        if len(features) != len(scale_factors) + 1:
            raise ValueError("Incompatible number of features {len(features)} and scale_factors {len(scale_factors)}")

        conv_kwargs = [conv_block_kwargs] * len(scale_factors)

        self.blocks = nn.ModuleList(
            [conv_block_impl(inc, outc, **kwargs)
             for inc, outc, kwargs in zip(features[:-1], features[1:], conv_kwargs)]
        )
        self.samplers = nn.ModuleList(
            [sampler_impl(factor, inc, inc) for factor, inc
             in zip(scale_factors, features[:-1])]
        )

        self.in_channels = features[0]
        self.out_channels = features[-1]

    def __len__(self):
        return len(self.blocks)

    def forward(self, x):
        for block, sampler in zip(self.blocks, self.samplers):
            x = sampler(x)
            x = block(x)

        return x


class VNETR(nn.Module):

    def _load_encoder_from_checkpoint(self, backbone, encoder, checkpoint):

        if isinstance(checkpoint, str):
            if backbone == "sam":
                # If we have a SAM encoder, then we first try to load the full SAM Model
                # (using micro_sam) and otherwise fall back on directly loading the encoder state
                # from the checkpoint
                try:
                    _, model = get_sam_model(
                        model_type=encoder,
                        checkpoint_path=checkpoint,
                        return_sam=True
                    )
                    encoder_state = model.image_encoder.state_dict()
                except Exception:
                    # If we have a MAE encoder, then we directly load the encoder state
                    # from the checkpoint.
                    encoder_state = torch.load(checkpoint)

            elif backbone == "mae":
                # vit initialization hints from:
                #     - https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L233-L242
                encoder_state = torch.load(checkpoint)["model"]
                encoder_state = OrderedDict({
                    k: v for k, v in encoder_state.items()
                    if (k != "mask_token" and not k.startswith("decoder"))
                })

                # let's remove the `head` from our current encoder (as the MAE pretrained don't expect it)
                current_encoder_state = self.encoder.state_dict()
                if ("head.weight" in current_encoder_state) and ("head.bias" in current_encoder_state):
                    del self.encoder.head

        else:
            encoder_state = checkpoint

        self.encoder.load_state_dict(encoder_state)

    def __init__(
        self,
        img_size: int = 1024,
        backbone: str = "sam",
        encoder: str = "vit_B",
        out_channels: int = 1,
        use_sam_stats: bool = False,
        use_mae_stats: bool = False,
        encoder_checkpoint: Optional[Union[str, OrderedDict]] = None,
        final_activation: Optional[Union[str, nn.Module]] = None
    ) -> None:
        super().__init__()

        self.use_sam_stats = use_sam_stats
        self.use_mae_stats = use_mae_stats

        print(f"Using {encoder} from {backbone.upper()}")
        self.encoder = get_vision_transformer(img_size=img_size, backbone=backbone, model=encoder)
        if encoder_checkpoint is not None:
            self._load_encoder_from_checkpoint(backbone, encoder, encoder_checkpoint)

        # parameters for the decoder network
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]
        self.out_channels = out_channels

        self.decoder = ExplicitDecoder(
            features=features_decoder,
            scale_factors=scale_factors[::-1],
            conv_block_impl=ConvBlock2d,
            sampler_impl=Upsampler2d
        )

        self.base = ConvBlock2d(self.encoder.embed_dim, features_decoder[0])

        self.out_conv = nn.Conv2d(features_decoder[-1], out_channels, 1)

        self.deconv = SingleDeconv2DBlock(features_decoder[-1], features_decoder[-1])

        self.decoder_head = ConvBlock2d(features_decoder[-1], features_decoder[-1])
        self.final_activation = self._get_activation(final_activation)

    def _get_activation(self, activation):
        return_activation = None
        if activation is None:
            return None
        if isinstance(activation, nn.Module):
            return activation
        if isinstance(activation, str):
            return_activation = getattr(nn, activation, None)
        if return_activation is None:
            raise ValueError(f"Invalid activation: {activation}")
        return return_activation()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.use_sam_stats:
            pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
            pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)
        elif self.use_mae_stats:
            # TODO: add mean std from mae experiments (or open up arguments for this)
            raise NotImplementedError
        else:
            pixel_mean = torch.Tensor([0.0, 0.0, 0.0]).view(-1, 1, 1).to(device)
            pixel_std = torch.Tensor([1.0, 1.0, 1.0]).view(-1, 1, 1).to(device)

        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = self.encoder.img_size - h
        padw = self.encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.encoder.img_size, self.encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, x):
        org_shape = x.shape[-2:]

        # backbone used for reshaping inputs to the desired "encoder" shape
        x = torch.stack([self.preprocess(e) for e in x], dim=0)

        if type(self.encoder) in [ViT_MAE, ViT_Sam]:
            x, _ = self.encoder(x)
        else:
            x = self.encoder(x)

        x = self.base(x)

        x = self.decoder(x)

        x = self.deconv(x)

        x = self.decoder_head(x)

        x = self.out_conv(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        x = self.postprocess_masks(x, org_shape, org_shape)
        return x
