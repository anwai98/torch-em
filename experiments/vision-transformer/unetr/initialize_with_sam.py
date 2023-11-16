import torch
from torch_em.model import UNETR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    backbone="mae", encoder="vit_b", out_channels=1
)
model.to(device)

x = torch.randn(1, 3, 224, 224).to(device=device)

y = model(x)
print(y.shape)
