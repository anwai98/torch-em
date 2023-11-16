import torch
from torch_em.model import UNETR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    backbone="mae", encoder="vit_b", out_channels=1, use_sam_stats=False
)
model.to(device)

x = torch.randn(1, 3, 1024, 1024).to(device=device)

y = model(x)
print(y.shape)
