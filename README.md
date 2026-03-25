# decomposition-regularized-diffusion

[![Paper](https://img.shields.io/badge/Paper-ICML%20Style-blue)](https://github.com/leffff/decomposition-regularized-diffusion)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

A novel family of generative models that replaces noise-based corruption with structured matrix decompositions. Instead of adding Gaussian noise, we progressively corrupt images by discarding information along principled axes using **Singular Value Decomposition (SVD)** and **Fast Fourier Transform (FFT)**.

## 📖 Overview

Traditional diffusion models corrupt images with Gaussian noise—a stochastic process with limited interpretability. DBGM introduces **deterministic, interpretable corruptions** that align with how information is naturally structured in images:

- **SVD Corruption**: Truncates singular values to remove low-rank components, preserving global structure while discarding fine details
- **FFT Corruption**: Zeroes out high frequencies in Fourier space, keeping low-frequency content (shapes, smooth variations) first

This yields highly structured generative trajectories where coarse features emerge before fine details—making the generation process more interpretable and controllable.

## 🚀 Key Features

- **Interpretable Forward Process**: Corrupt images via rank truncation (SVD) or frequency filtering (FFT)
- **Markovian & Non-Markovian Variants**: Support for both next-step prediction (Markovian) and clean-image prediction (Cold DBGM)
- **SNR-Aware Loss for Flow Matching**: Integrates decomposition-based objectives into Flow Matching without altering sampling
- **State-of-the-Art on MNIST**: Cold DBGM with FFT achieves FID 25.8 with only 15 NFE, outperforming standard Flow Matching

## 📁 Repository Structure
```
decomposition-regularized-diffusion/
├── fft_process_cold_diffusion.ipynb # FFT-based Cold DBGM (non-Markovian)
├── fft_process_markovian.ipynb # FFT-based Markovian DBGM
├── fm_mnist_fft_loss_v2.ipynb # Flow Matching + FFT SNR-Aware Loss
├── fm_mnist_svd_loss_v2.ipynb # Flow Matching + SVD SNR-Aware Loss
├── svd_process_cold_diffusion.ipynb # SVD-based Cold DBGM on MNIST
├── svd_process_cold_diffusion_cifar10.ipynb # SVD-based Cold DBGM on CIFAR-10
└── svd_process_markovian.ipynb # SVD-based Markovian DBGM
```


Reverting matrix decomposition for image generation
![noise](https://github.com/leffff/decomposition-regularized-diffusion/blob/main/assets/fwd_bwd.png)


![noise](https://github.com/leffff/decomposition-regularized-diffusion/blob/main/assets/noise_process.png)
Noise corruption

```python
def noise_truncate(x0, noise, t):
    return (1 - t) * x0 + (t) * noise
```

![svd](https://github.com/leffff/decomposition-regularized-diffusion/blob/main/assets/svd_process.png)
SVD corruption

```python
def svd_truncate(x, r):
    # x: [B, C, H, W]
    H, W = x.shape
    x_flat = x.view(H, W)

    U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)

    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    x_r = torch.matmul(U_r, torch.matmul(torch.diag_embed(S_r), Vh_r))
    x_r = x_r.view(H, W)
    return x_r
```

![fft](https://github.com/leffff/decomposition-regularized-diffusion/blob/main/assets/fft_process.png)
FFT corruption

```python
def fft_truncate(image, t):
    fft = torch.fft.fft2(image)
    shifted_fft = torch.fft.fftshift(fft, dim=(-2, -1))
    
    H, W = image.shape[-2:]
    h_center, w_center = H // 2, W // 2

    h_low = max(0, h_center - t)
    h_high = min(H, h_center + t)
    w_low = max(0, w_center - t)
    w_high = min(W, w_center + t)
    
    # Zero out everything *outside* the central square
    mask = torch.zeros_like(shifted_fft, dtype=torch.bool)
    mask[..., h_low:h_high, w_low:w_high] = True
    shifted_fft = shifted_fft * mask  # or use where / masked_fill
    
    # Inverse
    unshifted_fft = torch.fft.ifftshift(shifted_fft, dim=(-2, -1))
    restored = torch.fft.ifft2(unshifted_fft).real
    return restored
```

# SNR-Aware Loss
![](https://github.com/leffff/decomposition-regularized-diffusion/blob/main/assets/snr_aware_svd.png)
![](https://github.com/leffff/decomposition-regularized-diffusion/blob/main/assets/snr_aware_fft.png)
![](https://github.com/leffff/decomposition-regularized-diffusion/blob/main/assets/fid_sns_aware.png)

⚠️ Current Limitations
Conditional generation: Deterministic priors retain class information, making class-agnostic sampling difficult

Scaling to RGB: Complex priors on CIFAR-10 make prior distribution fitting (via GMM) challenging

SNR-Aware Loss on RGB: Selecting a single rank across multiple channels lacks theoretical foundation

🛠️ Requirements
Python 3.8+

PyTorch 1.10+

torchvision

numpy

scipy

matplotlib

📝 Citation
@article{novitskiy2024decomposition,
  title={Decomposition-Based Generative Models},
  author={Novitskiy, Lev and Varlamov, Alexander and Bogakovskaya, Sofiya},
  year={2026}
}

📧 Contact
[Lev Novitskiy](levnovitskiy@gmail.com)
Alexander Varlamov
Sofiya Bogakovskaya
