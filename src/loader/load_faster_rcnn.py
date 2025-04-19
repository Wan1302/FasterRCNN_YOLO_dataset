import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models 
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights



class CustomDataset(Dataset):
    def __init__(self, root_dir, sub_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.sub_dir = sub_dir
        self.image_dir = os.path.join(root_dir, "images", sub_dir)
        self.label_dir = os.path.join(root_dir, "labels", sub_dir)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([x_center, y_center, width, height])
                    labels.append(int(class_id) + 1)

        boxes = torch.tensor(boxes, dtype=torch.float16)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
            if boxes.numel() > 0:
                boxes[:, 0] = boxes[:, 0] * w
                boxes[:, 1] = boxes[:, 1] * h
                boxes[:, 2] = boxes[:, 2] * w
                boxes[:, 3] = boxes[:, 3] * h

                x_center = boxes[:, 0]
                y_center = boxes[:, 1]
                width = boxes[:, 2]
                height = boxes[:, 3]
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

                # Resize bounding boxes to match the resized image
                new_size = image.size()[1:]  # (Channel, H, W) -> (H, W)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_size[1] / w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_size[0] / h) 
        
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor(idx)}
        return image, target
    
def get_preprocessed_data(data_path, sub_dir, args):
    # weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    normalize = weights.transforms()
    transform = None
    if bool(args.is_aug):
        transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.3),  # Add Gaussian Blur, random apply for random stuffs
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.ToTensor(),
            normalize
        ])
    data = CustomDataset(
        root_dir=data_path,
        sub_dir=sub_dir,
        transform=transform
    )
    return data