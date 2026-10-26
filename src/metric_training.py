"""Explicit supervised metric loss for a separate experimental depth-decoder trial."""
import torch


def metric_supervision_loss(inverse, target_m, valid, object_mask, teacher_inverse, scale):
    if inverse.shape != target_m.shape or inverse.shape != valid.shape or inverse.shape != object_mask.shape or inverse.shape != teacher_inverse.shape or valid.dtype != torch.bool or object_mask.dtype != torch.bool:
        raise ValueError("Aligned depth tensors and boolean masks required")
    if not 0 < scale < 1e6 or not torch.isfinite(inverse).all() or (inverse <= 0).any() or not torch.isfinite(teacher_inverse).all() or (teacher_inverse <= 0).any():
        raise ValueError("Finite positive inverse depth and metric scale required")
    if valid.sum().item() < 16 or not torch.isfinite(target_m[valid]).all() or (target_m[valid] <= 0).any():
        raise ValueError("Insufficient finite positive metric supervision")
    log_metric = torch.log(scale / inverse)
    pixel = (log_metric[valid] - torch.log(target_m[valid])).abs().mean()
    selected = valid & object_mask
    object_loss = (log_metric[selected] - torch.log(target_m[selected])).abs().mean() if selected.sum().item() >= 4 else inverse.new_zeros(())
    teacher = (torch.log(inverse) - torch.log(teacher_inverse.detach())).abs().mean()
    total = pixel + .5 * object_loss + .1 * teacher
    return total, dict(pixel=pixel, object=object_loss, teacher=teacher)

