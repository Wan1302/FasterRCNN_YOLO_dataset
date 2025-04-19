import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse
import torch
from torchvision import models, transforms
import cv2
import os
import numpy as np
from PIL import Image

from models.FASTER_RCNN import FASTER_RCNN
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from utils.function import non_max_suppression
from utils.function import draw_bbox
from utils.yaml_helper import read_yaml

def inference(
        weights="weights/faster-rcnn.pt", 
        img_path="sample/1.jpg",
        class_names=None,
        detect_thresh=0.5,
        device="cpu"
        ):
    CLASS_NAMES = class_names
    DETECT_THRESH = detect_thresh
    print(img_path, CLASS_NAMES, DETECT_THRESH, device, weights)
    model = FASTER_RCNN(7)
    model.model.load_state_dict(torch.load(weights))
    model.model.to(device)
    model.eval()

    weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    normalize = weights.transforms()
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        normalize
    ])
    
        # check if img_path is a file path or an image
    if isinstance(img_path, str):
        image = cv2.imread(img_path) # cv2 format
        if image is None:
            raise FileNotFoundError(f"Image file '{img_path}' not found.")
    elif isinstance(img_path, np.ndarray):
        image = img_path
    else:
        raise ValueError("img_path must be a string (file path) or a numpy array (image).")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    image = cv2.resize(image, (640, 640))

    #!----- Detection
    with torch.no_grad():
        preds = model.model(image_tensor)
        preds = [{k: v.to(device) for k, v in t.items()} for t in preds]



    #!------------ Post-processing: filter low score, nms
    boxes = preds[0]['boxes']
    labels = preds[0]['labels'] - 1 # 0-based index for inference
    scores = preds[0]['scores']
    # print("Before NMS boxes:", boxes.shape)
    # print("Before NMS labels:", labels.shape)
    # print("Before NMS scores:", scores.shape)
    unique_labels = torch.unique(labels)
    final_boxes = []
    final_scores = []
    final_labels = []
    for label in unique_labels:
        class_mask = labels == label
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        keep = non_max_suppression(class_boxes, class_scores, iou_threshold=0.2)

        final_boxes.append(class_boxes[keep])
        final_scores.append(class_scores[keep])
        final_labels.append(torch.full((len(keep),), label, dtype=torch.int16))

# Concatenate results
    if final_boxes:
        boxes = torch.cat(final_boxes)
        scores = torch.cat(final_scores)
        labels = torch.cat(final_labels)
    else:
        boxes = torch.empty((0, 4), dtype=torch.float32)
        scores = torch.empty((0,), dtype=torch.float32)
        labels = torch.empty((0,), dtype=torch.int16)  
    # keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
    # keep = non_max_suppression(boxes, scores, labels, iou_threshold=0.8)

    # print("After NMS boxes:", boxes.shape)
    # print("After NMS labels:", labels.shape)
    # print("After NMS scores:", scores.shape)
    
    #----
    per_detections = []
    obj_detections = []
    image_detection = image.copy()
    for i in range(boxes.shape[0]):
        if scores[i] < DETECT_THRESH:
            continue
        box = boxes[i].cpu().numpy()
        label = labels[i].cpu().numpy()
        score = scores[i].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        class_id = int(label)
        draw_bbox(image_detection, class_id, x1, y1, x2, y2, score, type='detect', class_names=CLASS_NAMES)
        # Detections bbox format for tracker
        if CLASS_NAMES[class_id] == "Person": # only track person
            per_detections.append([x1, y1, x2, y2, score])
        else:
            obj_detections.append([x1, y1, x2, y2, score, class_id])

    return image_detection

if __name__ == "__main__":
    parse  = argparse.ArgumentParser(description="Parser for Faster-RCNN inference")
    parse.add_argument("--weights", type=str, default="weights/faster-rcnn.pt", help="Path to weights weights")
    parse.add_argument("--img_path", type=str, default="sample/1.jpg", help="Path to source image")
    parse.add_argument("--yaml_class", type=str, default="data/data-ppe.yaml", help="Path to class yaml")
    
    args = parse.parse_args()
    yaml_class = read_yaml(args.yaml_class)
    CLASS_NAMES = yaml_class["names"]
    DETECT_THRESH = 0.4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_detection = inference(
        device=device, weights=args.weights, 
        img_path=args.img_path, 
        class_names=CLASS_NAMES, 
        detect_thresh=DETECT_THRESH)
    
    image_detection = cv2.cvtColor(image_detection, cv2.COLOR_RGB2BGR)
    # Save the image
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num = 1

    while os.path.exists(os.path.join(output_dir, f"inference-{num}-frcnn.jpg")):
        num += 1
    output_path = os.path.join(output_dir, f"inference-{num}-frcnn.jpg")
    cv2.imwrite(output_path, image_detection)
    print(f"Saved inference result to {output_path}") 