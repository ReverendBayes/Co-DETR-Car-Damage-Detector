
import torch
import torch.nn.functional as F

def scylla_iou_loss(pred_boxes, target_boxes, reduction="mean", eps=1e-6):
    """
    Compute SCYLLA-IoU loss between predicted and target boxes.
    Boxes must be in xyxy format.

    Arguments:
    pred_boxes -- Tensor of shape (N, 4)
    target_boxes -- Tensor of shape (N, 4)
    reduction -- "mean" or "sum"

    Returns:
    Scalar tensor with loss value
    """
    # Ensure inputs are float
    pred_boxes = pred_boxes.float()
    target_boxes = target_boxes.float()

    # Extract coords
    x1_p, y1_p, x2_p, y2_p = pred_boxes.unbind(-1)
    x1_t, y1_t, x2_t, y2_t = target_boxes.unbind(-1)

    # Compute areas
    area_p = (x2_p - x1_p).clamp(min=0) * (y2_p - y1_p).clamp(min=0)
    area_t = (x2_t - x1_t).clamp(min=0) * (y2_t - y1_t).clamp(min=0)

    # Intersection
    x1 = torch.max(x1_p, x1_t)
    y1 = torch.max(y1_p, y1_t)
    x2 = torch.min(x2_p, x2_t)
    y2 = torch.min(y2_p, y2_t)
    inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    union = area_p + area_t - inter_area + eps
    iou = inter_area / union

    # Center distances
    cx_p = (x1_p + x2_p) / 2
    cy_p = (y1_p + y2_p) / 2
    cx_t = (x1_t + x2_t) / 2
    cy_t = (y1_t + y2_t) / 2
    center_dist = (cx_p - cx_t)**2 + (cy_p - cy_t)**2

    # Diagonal of smallest enclosing box
    enc_x1 = torch.min(x1_p, x1_t)
    enc_y1 = torch.min(y1_p, y1_t)
    enc_x2 = torch.max(x2_p, x2_t)
    enc_y2 = torch.max(y2_p, y2_t)
    enc_diag = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + eps

    # Distance penalty
    distance_cost = center_dist / enc_diag

    # Aspect ratio cost
    w_p = (x2_p - x1_p).clamp(min=eps)
    h_p = (y2_p - y1_p).clamp(min=eps)
    w_t = (x2_t - x1_t).clamp(min=eps)
    h_t = (y2_t - y1_t).clamp(min=eps)

    ar_p = w_p / h_p
    ar_t = w_t / h_t
    shape_cost = ((ar_p - ar_t)**2 / (ar_t + eps)**2)

    # Final SIoU loss (angle cost omitted for simplification)
    loss = 1 - iou + distance_cost + shape_cost

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
