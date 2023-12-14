import torch

from torch_em.model import VNETR


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VNETR(
        img_size=1024,
        backbone="sam",
        encoder="vit_b",
        out_channels=1
    )
    model.to(device)

    x = torch.randn(1, 1, 512, 512).to(device=device)
    y = model(x)

    print(y.shape)

    print("VNETR Model successfully created")


if __name__ == "__main__":
    main()
