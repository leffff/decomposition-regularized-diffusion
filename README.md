# decomposition-regularized-diffusion

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
