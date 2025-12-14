import torch
import numpy as np


def get_dis_target(
    y_true: torch.Tensor,
    step: int,
    total_steps: int,
    vocab_size: int,
    mask_token_id: int,
    ignore_label_id: int = -100,
    schedule: str = "linear",
    noise: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generate the target for Deep Improvement Supervision (DIS) at a specific step.

    Args:
        y_true: Ground truth tensor (B, L).
        step: Current supervision step (0 to total_steps - 1).
        total_steps: Total number of supervision steps.
        vocab_size: Size of the vocabulary (used for random replacement if needed).
        mask_token_id: ID of the mask token.
        ignore_label_id: ID to ignore in loss (usually -100).
        schedule: Type of schedule ("linear", "cosine", etc.). Currently only "linear" is supported.
        noise: Optional pre-generated noise tensor (B, L) for monotonic masking.

    Returns:
        y_target: The target tensor for the current step.
    """

    # Calculate mask rate based on schedule
    # We want mask_rate to go from 1.0 (step 0) to 0.0 (step total_steps)
    # Actually, step 0 should probably have some information, but let's follow the paper:
    # "decreasing noise schedule... so that y_dagger_N_sup = y_star"
    # If step goes from 0 to N-1:
    # step = N-1 -> mask_rate = 0.0 (Full target)
    # step = 0 -> mask_rate = high (e.g. 1.0 or 0.9)

    if schedule == "linear":
        # Linear decay from 1.0 to 0.0
        # At step=0, rate=1.0 (all masked) -> Model predicts from scratch?
        # At step=N-1, rate=0.0 (none masked) -> Model predicts full target
        # The paper says "intermediate targets... strictly closer to ground truth".
        # If we mask y_true, the "target" is the masked version?
        # No, the target for the *loss* is usually the full y_true, but the *input* to the next step is the prediction.
        # Wait, the paper says: "Each supervision step s is trained toward a step-specific intermediate target y_dagger_s".
        # And "y_dagger_s ~ q_beta(y_star)".
        # So the *label* for the loss is y_dagger_s.
        # If y_dagger_s is a masked version of y_star, then the model is trained to predict the *masked* sequence?
        # That implies the model should output "MASK" for masked tokens?
        # Usually in discrete diffusion (like MaskGIT), the model predicts the *original* token at the masked positions.
        # But here, the "target" itself is the intermediate state.
        # If the target is "MASK", the model learns to output "MASK".
        # Then in the next step, it takes "MASK" as input and predicts... "MASK" again?
        # No, the goal is to refine.
        # Let's re-read: "y_dagger_s ... strictly closer to y_star".
        # If y_dagger_s has MASK tokens, and y_star has real tokens.
        # Distance(MASK, real) > Distance(real, real).
        # So y_dagger_s should have *fewer* masks as s increases.
        # So at step s, the target `y_target` contains some real tokens and some MASK tokens.
        # The model output `y_hat` is compared to `y_target`.
        # So the model is indeed trained to output MASK tokens for the parts that are not yet "revealed".

        progress = (step + 1) / total_steps
        mask_rate = 1.0 - progress
    else:
        raise NotImplementedError(f"Schedule {schedule} not implemented")

    # Create mask
    # B, L
    B, L = y_true.shape
    device = y_true.device

    # Generate random mask if not provided
    if noise is None:
        rand_vals = torch.rand(B, L, device=device)
    else:
        rand_vals = noise

    # Mask where rand_vals < mask_rate
    # But we must respect ignore_label_id (padding)
    # Don't mask padding
    is_content = y_true != ignore_label_id

    mask = (rand_vals < mask_rate) & is_content

    y_target = y_true.clone()
    y_target[mask] = mask_token_id

    return y_target
