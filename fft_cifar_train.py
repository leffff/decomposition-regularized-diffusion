import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as TD
import torch.optim as optim
import argparse
import csv
from torch.distributions import Normal, Bernoulli, Independent
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

from collections import defaultdict
from tqdm.notebook import tqdm
from typing import List, Union
import gc
import os
import numpy as np
# from sklearn.datasets import make_regression
# from sklearn.metrics import r2_score
# from IPython.display import Image, display
# from IPython.core.display import HTML
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns

# import matplotlib.animation as animation
# from torch.linalg import svd

from diffusers import UNet2DModel
from diffusers.models.attention_processor import AttnProcessor2_0


def fft_forward_process(image, t):
    # image: [C, H, W] or [B, C, H, W] — fft2 operates on last 2 dims
    fft = torch.fft.fft2(image)
    shifted_fft = torch.fft.fftshift(fft, dim=(-2, -1))

    H, W = image.shape[-2:]
    h_center, w_center = H // 2, W // 2

    h_low  = max(0, h_center - t)
    h_high = min(H, h_center + t)
    w_low  = max(0, w_center - t)
    w_high = min(W, w_center + t)

    # mask shape matches [C, H, W] or [B, C, H, W] automatically
    mask = torch.zeros_like(shifted_fft, dtype=torch.bool)
    mask[..., h_low:h_high, w_low:w_high] = True
    shifted_fft = shifted_fft * mask

    unshifted_fft = torch.fft.ifftshift(shifted_fft, dim=(-2, -1))
    restored = torch.fft.ifft2(unshifted_fft).real
    return restored


def parse_args():
    parser = argparse.ArgumentParser(description="Train FFT-based UNet on CIFAR10.")
    parser.add_argument("--data-root", type=str, default=".", help="Dataset root directory.")
    parser.add_argument("--download", action="store_true", help="Download CIFAR10 if missing.")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size-train", type=int, default=128, help="Train batch size.")
    parser.add_argument("--batch-size-test", type=int, default=128, help="Test batch size.")
    parser.add_argument(
        "--no-shuffle-train",
        action="store_false",
        dest="shuffle_train",
        help="Disable shuffling for train loader.",
    )
    parser.add_argument(
        "--no-shuffle-test",
        action="store_false",
        dest="shuffle_test",
        help="Disable shuffling for test loader.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--base-ch", type=int, default=128, help="UNet base channels.")
    parser.add_argument("--time-emb-dim", type=int, default=128, help="Time embedding dimension.")
    parser.add_argument("--in-channels", type=int, default=3, help="Input/output channel count.")
    parser.add_argument("--high", type=int, default=32, help="Upper bound for FFT step sampling.")
    parser.add_argument("--log-every-n", type=int, default=10, help="Progress log frequency.")
    parser.add_argument(
        "--sampling-decay",
        type=float,
        default=0.5,
        help="Exponential bias in timestep sampling.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument(
        "--save-every-n",
        type=int,
        default=10,
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Base directory for checkpoints.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Base directory for logs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="default",
        help="Experiment name used as subdirectory in log/checkpoint dirs.",
    )
    parser.set_defaults(shuffle_train=True, shuffle_test=True)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    clip = transforms.Lambda(lambda x: x.clip(-1.0, 1.0))
    scale = transforms.Lambda(lambda x: x * 2 - 1)
    unscale = transforms.Lambda(lambda x: (x + 1) / 2)

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=args.download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            scale,
            clip,
        ])
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=args.download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            scale,
            clip,
        ])
    )

    losses = []

    log_every_n = args.log_every_n
    num_epoches = args.num_epochs
    step = 0
    HIGH = args.high

    DEVICE = args.device

    DEVICE = "cuda:0"

    checkpoint_exp_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    log_exp_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(checkpoint_exp_dir, exist_ok=True)
    os.makedirs(log_exp_dir, exist_ok=True)
    losses_csv_path = os.path.join(log_exp_dir, "losses.csv")

    with open(losses_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss"])

    # model = SimpleUNet(in_channels=3, base_ch=128, time_emb_dim=128).to(DEVICE)
    model = UNet2DModel(
        # num_class_embeds=10,
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
    )

    # Move to device first
    model = model.to(DEVICE)

    # 🔥 Manual method to enable FlashAttention-2 (works across diffusers versions)
    def set_attn_processors(model, processor):
        """Recursively set attention processors for all attention modules in the model."""
        for name, module in model.named_modules():
            if hasattr(module, "set_processor"):
                module.set_processor(processor)
            # For older diffusers versions where attention modules have 'processor' attribute directly
            elif hasattr(module, "processor") and "Attention" in module.__class__.__name__:
                module.processor = processor

    # Check if we can use AttnProcessor2_0 (PyTorch 2.0+ with efficient SDPA)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        processor = AttnProcessor2_0()
        print("✅ Using AttnProcessor2_0 (PyTorch 2.0 scaled_dot_product_attention)")
    else:
        from diffusers.models.attention_processor import AttnProcessor
        processor = AttnProcessor()
        print("⚠️ PyTorch 2.0 not available. Using default attention processor.")

    # Apply the processor to all attention modules
    set_attn_processors(model, processor)

    # Also ensure we're using bfloat16 if available and beneficial
    if torch.cuda.is_bf16_supported():
        use_bf16 = True
        print("✅ bfloat16 is supported on this device")
    else:
        use_bf16 = False
        print("⚠️ bfloat16 not supported, using float32")


    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"Number of parameters: {sum([p.numel() for p in model.parameters()]) / 1e6:.2f}M")


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        shuffle=args.shuffle_train,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=args.shuffle_test,
    )

    for epoch in range(num_epoches):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epoches}')
        epoch_loss_rows = []
        for images, labels in pbar:
            batch_src = []
            batch_dst = []

            # rec_step = torch.randint(size=(images.shape[0],), low=1, high=HIGH - 1)

            # Bias sampling towards earlier steps using exponential distribution
            weights = torch.exp(-args.sampling_decay * torch.arange(1, HIGH - 1, dtype=torch.float))
            weights = weights / weights.sum()
            rec_step = torch.multinomial(weights, images.shape[0], replacement=True) + 1
                    
            for i in range(images.shape[0]):
                our_sample = images[i]  # [C, H, W]
                
                src = fft_forward_process(our_sample, rec_step[i])
                
                dst = fft_forward_process(our_sample, rec_step[i] + 1)
        
                batch_src.append(src.unsqueeze(dim=0))
                batch_dst.append(dst.unsqueeze(dim=0))
                
            src = torch.cat(batch_src, dim=0).to(DEVICE)
            dst = torch.cat(batch_dst, dim=0).to(DEVICE)
            target = dst.to(DEVICE)
            rec_step = rec_step.to(DEVICE)

            optimizer.zero_grad()
            output = model(src, rec_step).sample
            
            loss = F.l1_loss(output, target)
        
            if step % log_every_n == 0:
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
        
            losses.append(loss.item())
            epoch_loss_rows.append([epoch + 1, step, float(loss.item())])
        
            step += 1

        with open(losses_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(epoch_loss_rows)

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