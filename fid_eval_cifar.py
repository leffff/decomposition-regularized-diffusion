import torch
import torchvision
from torchvision import transforms
import argparse
from tqdm import tqdm

from diffusers import UNet2DModel
from torchmetrics.image.fid import FrechetInceptionDistance


# ---------------------------------------------------------------------------
# FFT utilities
# ---------------------------------------------------------------------------

def fft_forward_process(image, t):
    fft = torch.fft.fft2(image)
    shifted_fft = torch.fft.fftshift(fft, dim=(-2, -1))

    H, W = image.shape[-2:]
    h_center, w_center = H // 2, W // 2
    h_low, h_high = max(0, h_center - t), min(H, h_center + t)
    w_low, w_high = max(0, w_center - t), min(W, w_center + t)

    mask = torch.zeros_like(shifted_fft, dtype=torch.bool)
    mask[..., h_low:h_high, w_low:w_high] = True
    shifted_fft = shifted_fft * mask

    unshifted_fft = torch.fft.ifftshift(shifted_fft, dim=(-2, -1))
    return torch.fft.ifft2(unshifted_fft).real


def extract_fft_latent(image, t):
    """Center-patch complex FFT coefficients as a real vector (real+imag interleaved)."""
    fft = torch.fft.fftshift(torch.fft.fft2(image), dim=(-2, -1))
    H, W = image.shape[-2:]
    h_center, w_center = H // 2, W // 2
    h_low, h_high = max(0, h_center - t), min(H, h_center + t)
    w_low, w_high = max(0, w_center - t), min(W, w_center + t)

    latent = fft[..., h_low:h_high, w_low:w_high]
    return torch.view_as_real(latent).flatten()


def latent_to_image(latent_vec, image_shape=(3, 32, 32), t=1):
    C, H, W = image_shape
    h_center, w_center = H // 2, W // 2
    h_low, h_high = max(0, h_center - t), min(H, h_center + t)
    w_low, w_high = max(0, w_center - t), min(W, w_center + t)

    patch_h, patch_w = h_high - h_low, w_high - w_low
    latent = torch.view_as_complex(latent_vec.view(C, patch_h, patch_w, 2))

    full_fft_shifted = torch.zeros((C, H, W), dtype=torch.complex64, device=latent_vec.device)
    full_fft_shifted[..., h_low:h_high, w_low:w_high] = latent
    full_fft = torch.fft.ifftshift(full_fft_shifted, dim=(-2, -1))
    return torch.fft.ifft2(full_fft).real


# ---------------------------------------------------------------------------
# Prior fitting
# ---------------------------------------------------------------------------

def _collect_and_fit(dataset, n_batches, batch_size, extractor, eps=1e-5):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    samples = []
    for i, (imgs, _) in enumerate(tqdm(loader, desc="Fitting prior", total=n_batches)):
        for img in imgs:
            samples.append(extractor(img))
        if i + 1 >= n_batches:
            break
    samples = torch.stack(samples)
    mean = samples.mean(dim=0)
    centered = samples - mean
    cov = centered.T @ centered / (len(samples) - 1)
    cov = cov + eps * torch.eye(cov.shape[0])
    return mean, cov


def fit_default_prior(dataset, n_batches=100, batch_size=128):
    """Gaussian on pixel-space t=1 FFT-filtered images (flattened to [3072])."""
    mean, cov = _collect_and_fit(
        dataset, n_batches, batch_size,
        extractor=lambda img: fft_forward_process(img, 1).flatten(),
    )
    return mean.view(3, 32, 32), cov


def fit_latent_prior(dataset, n_batches=100, batch_size=128):
    """Gaussian on complex FFT latent vectors (real+imag flattened)."""
    return _collect_and_fit(
        dataset, n_batches, batch_size,
        extractor=lambda img: extract_fft_latent(img, t=1),
    )


# ---------------------------------------------------------------------------
# Sample generation — FFT model
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples_fft(model, num_samples, high, device, prior_mean, prior_cov,
                         prior_type="default", batch_size=128):
    model.eval()
    dist = torch.distributions.MultivariateNormal(prior_mean.flatten(), covariance_matrix=prior_cov)
    all_samples = []

    for start in tqdm(range(0, num_samples, batch_size), desc="Generating (FFT)"):
        bs = min(batch_size, num_samples - start)
        latents = dist.sample((bs,))

        if prior_type == "latent":
            x = torch.stack([latent_to_image(z, t=1) for z in latents]).to(device)
        else:
            x = latents.view(bs, 3, 32, 32).to(device)

        for t in range(1, high):
            t_tensor = torch.full((bs,), t, device=device, dtype=torch.long)
            x = model(x, t_tensor).sample.clamp(-1, 1)

        x = ((x + 1) / 2).clamp(0, 1)
        x = (x * 255).to(torch.uint8).cpu()
        all_samples.append(x)

    return torch.cat(all_samples, dim=0)


# ---------------------------------------------------------------------------
# Sample generation — FM model
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples_fm(model, num_samples, device, num_timesteps=1000,
                        fm_steps=100, batch_size=128):
    """Euler integration from t=1 (noise) backwards to t=0 (clean image).
    Convention matches training: x0=data, x1=noise, v* = x1 - x0."""
    model.eval()
    dt = 1.0 / fm_steps
    all_samples = []

    for start in tqdm(range(0, num_samples, batch_size), desc="Generating (FM)"):
        bs = min(batch_size, num_samples - start)
        x = torch.randn(bs, 3, 32, 32, device=device)   # start from noise at t=1

        for i in range(fm_steps):
            t_cont = 1.0 - i * dt                        # descend 1 → 0
            t_int  = torch.full((bs,), int(t_cont * (num_timesteps - 1)),
                                device=device, dtype=torch.long)
            v = model(x, t_int).sample
            x = x - dt * v                               # Euler step backwards

        x = x.clamp(-1, 1)
        x = ((x + 1) / 2).clamp(0, 1)
        x = (x * 255).to(torch.uint8).cpu()
        all_samples.append(x)

    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def generate_samples_fm_x0(model, num_samples, device, num_timesteps=1000,
                            fm_steps=100, batch_size=128):
    """Euler integration with x0-parametrization: model predicts x0 from x_t.
    Implied velocity: v = (x_t - x0_hat) / t."""
    model.eval()
    all_samples = []

    for start in tqdm(range(0, num_samples, batch_size), desc="Generating (FM-x0)"):
        bs = min(batch_size, num_samples - start)
        x = torch.randn(bs, 3, 32, 32, device=device)

        ts = torch.linspace(1.0, 1e-3, fm_steps + 1, device=device)
        for i in range(fm_steps):
            t = ts[i]
            t_next = ts[i + 1]
            dt = t - t_next

            t_int = torch.full((bs,), int(t * (num_timesteps - 1)),
                               device=device, dtype=torch.long)
            x0_hat = model(x, t_int).sample

            v_hat = (x - x0_hat) / (t + 1e-6)
            x = x - dt * v_hat

        x = x.clamp(-1, 1)
        x = ((x + 1) / 2).clamp(0, 1)
        x = (x * 255).to(torch.uint8).cpu()
        all_samples.append(x)

    return torch.cat(all_samples, dim=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FID of UNet vs CIFAR-10 train set.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="fft", choices=["fft", "fm", "fm_x0"],
                        help="'fft': FFT generative model; 'fm': Flow Matching (v-pred); 'fm_x0': Flow Matching (x0-pred).")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    # FFT-specific
    parser.add_argument("--high", type=int, default=32, help="[FFT] FFT steps (must match training).")
    parser.add_argument("--prior-batches", type=int, default=100,
                        help="[FFT] Dataset batches used to fit the t=1 Gaussian prior.")
    parser.add_argument("--prior-type", type=str, default="default", choices=["default", "latent"],
                        help="[FFT] 'default': pixel-space prior; 'latent': complex FFT coefficient prior.")
    # FM-specific
    parser.add_argument("--fm-steps", type=int, default=100,
                        help="[FM] Number of Euler steps for ODE integration.")
    parser.add_argument("--num-timesteps", type=int, default=1000,
                        help="[FM] Discrete timestep bins (must match training).")
                
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)

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
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}  |  Seed: {args.seed}")

    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=args.download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ]),
    )

    if args.model_type == "fft":
        print("Fitting Gaussian prior at t=1...")
        if args.prior_type == "latent":
            prior_mean, prior_cov = fit_latent_prior(dataset, args.prior_batches, args.batch_size)
        else:
            prior_mean, prior_cov = fit_default_prior(dataset, args.prior_batches, args.batch_size)

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    indices = torch.randperm(len(dataset))[:args.num_samples]
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, num_workers=4)

    print("Feeding real images to FID...")
    for imgs, _ in tqdm(loader, desc="Real"):
        imgs_uint8 = ((imgs + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).to(device)
        fid.update(imgs_uint8, real=True)

    if args.model_type == "fm":
        generated = generate_samples_fm(
            model, args.num_samples, device,
            num_timesteps=args.num_timesteps,
            fm_steps=args.fm_steps,
            batch_size=args.batch_size,
        )
    elif args.model_type == "fm_x0":
        generated = generate_samples_fm_x0(
            model, args.num_samples, device,
            num_timesteps=args.num_timesteps,
            fm_steps=args.fm_steps,
            batch_size=args.batch_size,
        )
    else:
        generated = generate_samples_fft(
            model, args.num_samples, args.high, device,
            prior_mean, prior_cov, args.prior_type, args.batch_size,
        )

    print("Feeding generated images to FID...")
    for start in tqdm(range(0, len(generated), args.batch_size), desc="Fake"):
        batch = generated[start:start + args.batch_size].to(device)
        fid.update(batch, real=False)

    score = fid.compute().item()
    print(f"\nFID ({args.num_samples} samples): {score:.4f}")


if __name__ == "__main__":
    main()
