import torch
import cv2
from utils.yaml_helper import read_yaml


COLORS  = [
        (128, 0, 0),    # Maroon
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        (192, 192, 192),# Silver
        (128, 128, 128),# Gray
        (0, 0, 128),    # Navy
        (255, 255, 0),  # Yellow
        (0, 128, 0),    # Dark Green
        (128, 0, 128),  # Purple
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (255, 192, 203),# Pink
        (255, 222, 173),# Navajo White
        (173, 216, 230),# Light Blue
        (240, 230, 140),# Khaki
    ]

def non_max_suppression(boxes, scores, iou_threshold):
    # Input validation
    if len(boxes) == 0 or len(scores) == 0:
        return []
    
    # Convert to tensor if not already
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    
    # Ensure boxes and scores have same first dimension
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"boxes and scores must have same length, got {boxes.shape[0]} and {scores.shape[0]}")

    # Handle single box case
    if boxes.shape[0] == 1:
        return [0]

    # Get coordinates
    x1 = boxes[:, 0] # x_min of all boxes -> (N,)
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by score
    _, order = scores.sort(0, descending=True)
    order = order.reshape(-1)  # Ensure order is 1D

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break
        
        #! box with highest score
        i = order[0].item()
        keep.append(i)

        # Compute IoU
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.max(torch.tensor(0.0), xx2 - xx1 + 1)
        h = torch.max(torch.tensor(0.0), yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than threshold
        mask = iou <= iou_threshold
        if not mask.any():
            break
            
        inds = mask.nonzero().reshape(-1)
        order = order[inds + 1]

    return keep

def remove_invalid_boxes(targets):
    valid_targets = []
    for target in targets:
        boxes = target['boxes']
        labels = target['labels']
        valid_indices = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])  # x1 < x2 and y1 < y2
        valid_boxes = boxes[valid_indices]
        valid_labels = labels[valid_indices]
        valid_targets.append({'boxes': valid_boxes, 'labels': valid_labels})
    return valid_targets


def draw_bbox(frame, id, x1, y1, x2, y2, conf, missing=None, type='detect', class_names=None):
    if type == "track":
        color = (0,254,255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f'ID: {id}, Score: {conf:.2f}'
        cv2.putText(frame, text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    elif type == "violate":
        color = (255,0,0) if missing else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if missing:
            text = f'ID {id} w NO:'
            cv2.putText(frame, text, (x1, y1-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            missing = ", ".join([str(class_names[m]) for m in missing])
            cv2.putText(frame, missing, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            text = f'ID {id} OKE'
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    elif type == "detect":
        # Detection boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[id], 2)
        cv2.putText(frame, f"{class_names[id]}: {conf:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[id], 2)