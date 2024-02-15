import torch
import torch.nn as nn


class CustomUNet2d(nn.Module):
    def encoder_block(self, in_feats, out_feats, pool):
        layers = [nn.MaxPool2d(2)] if pool else []
        layers.extend([
            nn.Conv2d(in_feats, out_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ])
        return nn.Sequential(*layers)

    def decoder_block(self, in_feats, out_feats):
        return nn.Sequential(*[
            nn.Conv2d(in_feats, out_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ])

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        in_features = [in_channels, 64, 128, 256]
        out_features = [64, 128, 256, 512]
        self.encoders = nn.ModuleList([
            self.encoder_block(in_feats, out_feats, pool=level > 0)
            for level, (in_feats, out_feats) in enumerate(zip(in_features, out_features))
        ])
        self.base = self.encoder_block(512, 1024, pool=True)

        in_features = [1024, 512, 256, 128]
        out_features = [512, 256, 128, 64]
        self.decoders = nn.ModuleList([
            self.decoder_block(in_feats, out_feats)
            for in_feats, out_feats in zip(in_features, out_features)
        ])
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose2d(in_feats, out_feats, 2, stride=2)
            for in_feats, out_feats in zip(in_features, out_features)
        ])
        self.out_conv = nn.Conv2d(out_features[-1], out_channels, 1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        from_encoder = []
        for encoder in self.encoders:
            x = encoder(x)
            from_encoder.append(x)
        x = self.base(x)
        from_encoder = from_encoder[::-1]
        for decoder, upsampler, from_enc in zip(self.decoders, self.upsamplers, from_encoder):
            x = decoder(torch.cat([
                from_enc, upsampler(x)
            ], dim=1))
        x = self.out_conv(x)
        x = self.final_activation(x)
        return x
