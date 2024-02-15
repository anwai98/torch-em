import torch
import torch_em
from torch_em.model import UNet2d
from torch_em.data.datasets import get_covid_if_loader

from common import CustomUNet2d


def train_boundaries(args, model, loss):
    patch_shape = (512, 512)
    # use the first 5 images for validation
    train_loader = get_covid_if_loader(
        args.input, patch_shape, sample_range=(5, None),
        download=True, boundaries=True, batch_size=args.batch_size
    )
    val_loader = get_covid_if_loader(
        args.input, patch_shape, sample_range=(0, 5),
        boundaries=True, batch_size=args.batch_size
    )
    trainer = torch_em.default_segmentation_trainer(
        name="covid-if-boundary-model",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=torch.device("cuda"),
        mixed_precision=True,
        log_image_interval=50,
        compile_model=False
    )
    trainer.fit(iterations=args.n_iterations)


def predict_boundaries(args, model):
    checkpoint_path = "checkpoints/covid-if-boundary-model/best.pt"
    breakpoint()
    model_state = torch.load(checkpoint_path, map_location="cpu")["model_state"]
    model.load_state_dict(model_state)
    model.to("cuda")

    _loader = get_covid_if_loader(
        args.input, (512, 512), sample_range=(0, 5), boundaries=True, batch_size=1
    )

    for x, y in _loader:
        true_fg, true_bd = y.squeeze()
        outputs = model(x.to("cuda"))
        # outputs = torch.sigmoid(outputs)
        fg, bd = outputs.detach().cpu().squeeze().numpy()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 5)

        ax[0].imshow(x.squeeze(), cmap="gray")
        ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(true_bd)
        ax[1].set_title("True BD")
        ax[1].axis("off")

        ax[2].imshow(bd)
        ax[2].set_title("Predicted BD")
        ax[2].axis("off")

        ax[3].imshow(true_fg)
        ax[3].set_title("True FG")
        ax[3].axis("off")

        ax[4].imshow(fg)
        ax[4].set_title("Predicted FG")
        ax[4].axis("off")

        plt.tight_layout()
        plt.savefig("plot.png")
        plt.close()

        breakpoint()


def main(args):
    if args.use_torch_em:
        model = UNet2d(
            in_channels=1,
            out_channels=2,
            initial_features=64,
            final_activation="Sigmoid"
        )
        loss = torch_em.loss.DiceLoss()
    else:
        model = CustomUNet2d(
            in_channels=1,
            out_channels=2
        )
        loss = torch_em.loss.DiceLoss()

    if args.train:
        train_boundaries(args, model, loss)

    if args.predict:
        predict_boundaries(args, model)


if __name__ == '__main__':
    parser = torch_em.util.parser_helper(
        default_batch_size=8
    )
    parser.add_argument("--use_torch_em", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    main(args)
