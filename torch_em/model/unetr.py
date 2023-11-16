import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from functools import partial
from torch_em.model.unet import Decoder, ConvBlock2d, Upsampler2d

# we catch ImportErrors here because segment_anything, micro_sam and timm should
# only be optional dependencies for torch_em
try:
    from segment_anything.modeling import ImageEncoderViT
    _sam_import_success = True
except ImportError:
    ImageEncoderViT = object
    _sam_import_success = False

try:
    from micro_sam.util import get_sam_model
except ImportError:
    get_sam_model = None

try:
    import timm.models.vision_transformer as timm_vit
    _timm_import_success = True
except ImportError:
    timm_vit.VisionTransformer = object
    _timm_import_success = False


class ViT_Sam(ImageEncoderViT):
    """Vision Transformer derived from the Segment Anything Codebase (https://arxiv.org/abs/2304.02643):
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
    """
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 768,
        global_attn_indexes: Tuple[int, ...] = ...,
        **kwargs
    ) -> None:
        if not _sam_import_success:
            raise RuntimeError(
                "The vision transformer backend can only be initialized if segment anything is installed."
                "Please install segment anything from https://github.com/facebookresearch/segment-anything."
                "and then rerun your code."
            )

        super().__init__(embed_dim=embed_dim, **kwargs)
        self.chunks_for_projection = global_attn_indexes
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        list_from_encoder = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.chunks_for_projection:
                list_from_encoder.append(x)

        x = x.permute(0, 3, 1, 2)
        list_from_encoder = [e.permute(0, 3, 1, 2) for e in list_from_encoder]
        return x, list_from_encoder[:3]


class ViT_MAE(timm_vit.VisionTransformer):
    """Vision Transformer derived from the Masked Auto Encoder Codebase (https://arxiv.org/abs/2111.06377)
    https://github.com/facebookresearch/mae/blob/main/models_vit.py#L20-L53
    """
    def __init__(
            self,
            in_chans=3,
            depth=12,
            **kwargs
    ):
        if not _timm_import_success:
            raise RuntimeError(
                "The vision transformer backend can only be initialized if timm is installed."
                "Please install timm (using conda/mamba) for using https://github.com/facebookresearch/mae/."
                "and then rerun your code"
            )
        super().__init__(depth=depth, **kwargs)
        self.in_chans = in_chans
        self.depth = depth
        self.img_size = self.patch_embed.img_size[0]

    def convert_to_expected_dim(self, inputs_):
        inputs_ = inputs_[:, 1:, :]  # removing the class tokens
        # reshape the outputs to desired shape (N x H*W X C -> N x H x W x C)
        rdim = inputs_.shape[1]
        dshape = int(rdim ** 0.5)  # finding the square root of the outputs for obtaining the patch shape
        inputs_ = torch.unflatten(inputs_, 1, (dshape, dshape))
        inputs_ = inputs_.permute(0, 3, 1, 2)
        return inputs_

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # chunks obtained for getting the projections for conjuctions with upsampling blocks
        _chunks = int(self.depth / 4)
        chunks_for_projection = [_chunks - 1, 2*_chunks - 1, 3*_chunks - 1, 4*_chunks - 1]

        list_from_encoder = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in chunks_for_projection:
                list_from_encoder.append(self.convert_to_expected_dim(x))

        x = self.convert_to_expected_dim(x)
        return x, list_from_encoder[:3]

    def forward(self, x):
        x, list_from_encoder = self.forward_features(x)
        return x, list_from_encoder


class UNETR(nn.Module):
    def __init__(
        self,
        backbone="sam",
        encoder="vit_b",
        decoder=None,
        out_channels=1,
        use_sam_stats=True,
        use_mae_stats=False,
        encoder_checkpoint_path=None
    ) -> None:
        super().__init__()

        self.use_sam_stats = use_sam_stats
        self.use_mae_stats = use_mae_stats

        if backbone == "sam":
            if encoder == "vit_b":
                self.encoder = ViT_Sam(
                    depth=12, embed_dim=768, img_size=1024,  mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),  # type: ignore
                    num_heads=12, patch_size=16, qkv_bias=True, use_rel_pos=True,
                    global_attn_indexes=[2, 5, 8, 11],  # type: ignore
                    window_size=14, out_chans=256,
                )
            elif encoder == "vit_l":
                self.encoder = ViT_Sam(
                    depth=24, embed_dim=1024, img_size=1024, mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),  # type: ignore
                    num_heads=16, patch_size=16, qkv_bias=True, use_rel_pos=True,
                    global_attn_indexes=[5, 11, 17, 23],  # type: ignore
                    window_size=14,  out_chans=256
                )
            elif encoder == "vit_h":
                self.encoder = ViT_Sam(
                    depth=32, embed_dim=1280, img_size=1024, mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),  # type: ignore
                    num_heads=16, patch_size=16, qkv_bias=True, use_rel_pos=True,
                    global_attn_indexes=[7, 15, 23, 31],  # type: ignore
                    window_size=14, out_chans=256
                )

            else:
                raise ValueError(f"{encoder} is not supported by SAM. Currently vit_b, vit_l, vit_h are supported.")

        elif backbone == "mae":
            self.use_sam_preprocessing = False
            if encoder == "vit_b":
                self.encoder = ViT_MAE(
                    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                )
            elif encoder == "vit_l":
                self.encoder = ViT_MAE(
                    patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                )
            elif encoder == "vit_h":
                self.encoder = ViT_MAE(
                    patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                )
            else:
                raise ValueError(f"{encoder} is not supported by MAE. Currently vit_b, vit_l, vit_h are supported.")

        else:
            raise ValueError("The UNETR supported backbones are `sam` or `mae`. Please choose either of the two")

        if backbone == "sam":
            _, model = get_sam_model(
                model_type=encoder,
                checkpoint_path=encoder_checkpoint_path,
                return_sam=True
            )
            for param1, param2 in zip(model.parameters(), self.encoder.parameters()):
                param2.data = param1

        # TODO: ini MAE weights in vit mae

        # parameters for the decoder network
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]
        self.out_channels = out_channels

        if decoder is None:
            self.decoder = Decoder(
                features=features_decoder,
                scale_factors=scale_factors[::-1],
                conv_block_impl=ConvBlock2d,
                sampler_impl=Upsampler2d
            )
        else:
            self.decoder = decoder

        self.z_inputs = ConvBlock2d(self.encoder.in_chans, features_decoder[-1])

        self.base = ConvBlock2d(self.encoder.embed_dim, features_decoder[0])
        self.out_conv = nn.Conv2d(features_decoder[-1], out_channels, 1)
        self.final_activation = nn.Sigmoid()

        self.deconv1 = Deconv2DBlock(self.encoder.embed_dim, features_decoder[0])
        self.deconv2 = Deconv2DBlock(features_decoder[0], features_decoder[1])
        self.deconv3 = Deconv2DBlock(features_decoder[1], features_decoder[2])

        self.deconv4 = SingleDeconv2DBlock(features_decoder[-1], features_decoder[-1])

        self.decoder_head = ConvBlock2d(2*features_decoder[-1], features_decoder[-1])

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.use_sam_stats:
            pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
            pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)
        elif self.use_mae_stats:
            # TODO: add mean std from mae experiments (or open up arguments for this)
            raise NotImplementedError
        else:
            pixel_mean = 0
            pixel_std = 1

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

        z0 = self.z_inputs(x)

        z12, from_encoder = self.encoder(x)
        x = self.base(z12)

        from_encoder = from_encoder[::-1]
        z9 = self.deconv1(from_encoder[0])

        z6 = self.deconv1(from_encoder[1])
        z6 = self.deconv2(z6)

        z3 = self.deconv1(from_encoder[2])
        z3 = self.deconv2(z3)
        z3 = self.deconv3(z3)

        updated_from_encoder = [z9, z6, z3]
        x = self.decoder(x, encoder_inputs=updated_from_encoder)
        x = self.deconv4(x)
        x = torch.cat([x, z0], dim=1)

        x = self.decoder_head(x)

        x = self.out_conv(x)
        x = self.final_activation(x)

        x = self.postprocess_masks(x, org_shape, org_shape)
        return x


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
