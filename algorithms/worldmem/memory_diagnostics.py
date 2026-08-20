import torch
import torch.nn.functional as F


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def structural_similarity_per_image(prediction, target, data_range=1.0):
    """Return standard Gaussian-window SSIM for each NCHW image."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have matching NCHW shapes")

    prediction = prediction.float()
    target = target.float()
    channels = prediction.shape[1]
    minimum_side = min(prediction.shape[-2:])
    window_size = min(11, minimum_side)
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)
    sigma = 1.5 * window_size / 11
    window = _gaussian_window(
        window_size,
        sigma,
        channels,
        prediction.device,
        prediction.dtype,
    )
    padding = window_size // 2

    mu_pred = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target

    sigma_pred = (
        F.conv2d(prediction.square(), window, padding=padding, groups=channels)
        - mu_pred_sq
    ).clamp_min(0)
    sigma_target = (
        F.conv2d(target.square(), window, padding=padding, groups=channels)
        - mu_target_sq
    ).clamp_min(0)
    sigma_cross = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mu_cross
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (
        sigma_pred + sigma_target + c2
    )
    return (numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)).flatten(1).mean(1)


def lpips_per_image(model, prediction, target):
    """Use TorchMetrics' underlying LPIPS network without temporal accumulation."""
    if model is None:
        return torch.full(
            (prediction.shape[0],),
            float("nan"),
            device=prediction.device,
        )

    normalize = bool(getattr(model, "normalize", False))
    try:
        values = model.net(prediction, target, normalize=normalize)
        return values.detach().float().reshape(prediction.shape[0], -1).mean(1)
    except (AttributeError, TypeError):
        values = []
        for index in range(prediction.shape[0]):
            model.reset()
            model.update(prediction[index : index + 1], target[index : index + 1])
            values.append(model.compute().detach().float())
        model.reset()
        return torch.stack(values)


def image_quality_per_image(prediction, target, lpips_model=None, psnr_cap=100.0):
    """Compute metrics on [0, 1] NCHW images, retaining one value per image."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have matching NCHW shapes")

    prediction = prediction.detach().float().clamp(0, 1)
    target = target.detach().float().clamp(0, 1).to(prediction.device)
    mse = (prediction - target).square().flatten(1).mean(1)
    psnr = -10 * torch.log10(mse.clamp_min(10 ** (-float(psnr_cap) / 10)))
    psnr = psnr.clamp_max(float(psnr_cap))
    ssim = structural_similarity_per_image(prediction, target)
    lpips = lpips_per_image(lpips_model, prediction, target)
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
    }


def mean_quality(metrics):
    return {
        key: float(value.detach().float().mean().cpu().item())
        for key, value in metrics.items()
    }
