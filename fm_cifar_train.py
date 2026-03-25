import torch
import torch.nn.functional as F
import argparse
import csv
import os
import torchvision
from torchvision import transforms
from tqdm import tqdm

from diffusers import UNet2DModel
from diffusers.models.attention_processor import AttnProcessor2_0


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flow Matching UNet on CIFAR10.")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size-train", type=int, default=128)
    parser.add_argument("--batch-size-test", type=int, default=128)
    parser.add_argument(
        "--no-shuffle-train", action="store_false", dest="shuffle_train",
    )
    parser.add_argument(
        "--no-shuffle-test", action="store_false", dest="shuffle_test",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every-n", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-every-n", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--experiment-name", type=str, default="fm_default")
    parser.add_argument(
        "--num-timesteps", type=int, default=1000,
        help="Number of discrete timestep bins (t in [0,1] is scaled to [0, T-1]).",
    )
    parser.set_defaults(shuffle_train=True, shuffle_test=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    DEVICE = args.device
    T = args.num_timesteps

    scale = transforms.Lambda(lambda x: x * 2 - 1)
    clip  = transforms.Lambda(lambda x: x.clip(-1.0, 1.0))

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=args.download,
        transform=transforms.Compose([transforms.ToTensor(), scale, clip]),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=args.download,
        transform=transforms.Compose([transforms.ToTensor(), scale, clip]),
    )

    checkpoint_exp_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    log_exp_dir        = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(checkpoint_exp_dir, exist_ok=True)
    os.makedirs(log_exp_dir, exist_ok=True)
    losses_csv_path = os.path.join(log_exp_dir, "losses.csv")

    with open(losses_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "loss"])

    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        attention_head_dim=4,
        norm_num_groups=32,
        mid_block_type="UNetMidBlock2D",
    ).to(DEVICE)

    def set_attn_processors(model, processor):
        for _, module in model.named_modules():
            if hasattr(module, "set_processor"):
                module.set_processor(processor)
            elif hasattr(module, "processor") and "Attention" in module.__class__.__name__:
                module.processor = processor

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        set_attn_processors(model, AttnProcessor2_0())
        print("Using AttnProcessor2_0")
    else:
        from diffusers.models.attention_processor import AttnProcessor
        set_attn_processors(model, AttnProcessor())

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size_train, shuffle=args.shuffle_train,
    )

    step = 0
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        epoch_loss_rows = []

        for x0, _ in pbar:
            x0 = x0.to(DEVICE)                                 # clean image
            x1 = torch.randn_like(x0)                          # noise

            # Continuous t ~ U(0, 1), discretised for the timestep embedding
            # t=0: clean image, t=1: noise
            t_cont = torch.rand(x0.shape[0], device=DEVICE)    # [B]
            t_int  = (t_cont * (T - 1)).long()                 # [B], in [0, T-1]

            # Linear interpolation: x_t = (1-t)*x0 + t*x1
            t_bcast = t_cont.view(-1, 1, 1, 1)
            x_t = (1 - t_bcast) * x0 + t_bcast * x1

            # Target velocity (data→noise direction): v* = x1 - x0
            v_target = x1 - x0

            optimizer.zero_grad()
            v_pred = model(x_t, t_int).sample
            loss   = F.mse_loss(v_pred, v_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

            if step % args.log_every_n == 0:
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            epoch_loss_rows.append([epoch + 1, step, float(loss.item())])
            step += 1

        with open(losses_csv_path, "a", newline="") as f:
            csv.writer(f).writerows(epoch_loss_rows)

        if (epoch + 1) % args.save_every_n == 0:
            checkpoint_path = os.path.join(checkpoint_exp_dir, f"epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                checkpoint_path,
            )
