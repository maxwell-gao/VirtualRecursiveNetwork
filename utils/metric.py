
import math
import torch
import torch.nn.functional as F

try:
    import torchvision.ops
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def _finsler_randers(v, M, w, eps=1e-6):
    """Finsler-Randers metric: F(v) = sqrt(v^T M v) + w^T v"""
    M_2x2 = M.reshape(M.shape[0], 2, 2, *M.shape[-2:])
    norm = torch.sqrt(torch.einsum("bjrc,bijrc,birc->brc", v, M_2x2, v) + eps)
    drift = torch.einsum("birc,birc->brc", w, v)
    return norm + drift


def _sample_tangent_ball(batch_size, M, w, kh, kw, device, eps=1e-6):
    """Sample points on unit tangent ball using onion peeling."""
    out_shape = M.shape[-2:]
    u_list, s_list = [], []

    for k in range(kh // 2 + 1):
        if k == 0:
            theta = torch.zeros(1, device=device)
            u_list.append(torch.stack([torch.cos(theta), torch.sin(theta)], dim=1))
            s_list.append(torch.zeros(1, device=device))
        else:
            n = 8 * k
            theta = torch.linspace(0, 2 * math.pi * (1 - 1 / n), n, device=device)
            u_list.append(torch.stack([torch.cos(theta), torch.sin(theta)], dim=1))
            s_list.append(torch.full((n,), k / (kh // 2), device=device))

    u = torch.cat(u_list, dim=0)
    s = torch.cat(s_list)
    N = u.shape[0]

    # Ensure we have enough points for the kernel
    target_N = kh * kw
    if N < target_N:
         # Fill with center point (0,0) or some strategy if not enough points sampled
         # This onion peeling strategy is specific to generating roughly kh*kw points
         # for a square grid.
         pass
    
    # We truncate or pad to match kh*kw. 
    # The original implementation in patch.py seems to assume specific kernel sizes 
    # or that the onion peeling matches the number of patch points.
    # For a standard 3x3 conv, we need 9 points.
    # k=0: 1 point
    # k=1: 8 points
    # Total = 9. Perfect for 3x3.
    
    u_exp = u.view(1, N, 2, 1, 1).expand(batch_size, -1, -1, *out_shape)
    u_flat = u_exp.reshape(batch_size * N, 2, *out_shape)
    M_rep = M.repeat(N, 1, 1, 1)
    w_rep = w.repeat(N, 1, 1, 1)
    F_u = _finsler_randers(u_flat, M_rep, w_rep, eps).view(batch_size, N, *out_shape)

    y = u_exp / (F_u.unsqueeze(2) + eps) * s.view(1, N, 1, 1, 1)
    return y.view(batch_size, kh, kw, 2, *out_shape)

def get_metric_offsets(x, metric_net, kernel_size, eps_w=0.5):
    """
    Compute offsets for deform_conv2d using Finsler-Randers metric.
    
    Args:
        x: Input tensor (B, C, H, W)
        metric_net: Network to predict metric params (B, 7, H, W)
        kernel_size: Int, kernel size (e.g. 3)
        eps_w: Float, stability parameter for drift
    
    Returns:
        offsets: (B, 2*kernel_size*kernel_size, H, W)
    """
    B = x.shape[0]
    params = metric_net(x)
    
    # Build positive-definite M
    v = F.normalize(params[:, :2].permute(0, 2, 3, 1), dim=-1)
    eigvals = 2 * torch.sigmoid(params[:, 2:4].permute(0, 2, 3, 1))
    scale = 0.5 + 1.5 * torch.sigmoid(params[:, 4:5].permute(0, 2, 3, 1))
    eigvals = eigvals * scale

    R = torch.stack([v, torch.stack([-v[..., 1], v[..., 0]], -1)], -1)
    M = R @ torch.diag_embed(eigvals) @ R.transpose(-1, -2)
    M = M.permute(0, 3, 4, 1, 2).reshape(B, 4, *params.shape[-2:])

    # Build drift w
    w = params[:, 5:7]
    if eps_w < 1.0:
        w = w * (1 - eps_w) * torch.sigmoid(w.norm(dim=1, keepdim=True))
    else:
        w = torch.zeros_like(w)

    # Sample tangent ball and compute offsets
    # Only supports square kernels like 3x3 (k=1 onion layer) for now based on _sample_tangent_ball logic
    y = _sample_tangent_ball(B, M, w, kernel_size, kernel_size, x.device)

    P = kernel_size
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(-(P // 2), P // 2 + 1, device=x.device),
            torch.arange(-(P // 2), P // 2 + 1, device=x.device),
            indexing="ij",
        ),
        -1,
    ).float()

    offsets = (y.permute(0, 4, 5, 1, 2, 3).flip(-1) - grid).permute(
        0, 3, 4, 5, 1, 2
    )
    return offsets.reshape(B, -1, *params.shape[-2:])

