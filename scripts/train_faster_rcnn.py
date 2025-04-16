import sys
import os
import datetime
from tqdm import tqdm
import torch
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.FASTER_RCNN import FASTER_RCNN
from utils.metrics import compute_metrics
from utils.yaml_helper import read_yaml
from utils.data_helper import collate_fn
from loader.load_faster_rcnn import get_preprocessed_data
from parsers.parser_faster_rcnn import parse_args

args = parse_args()
yaml = read_yaml(args.yaml)
CLASS_NAMES = yaml["names"]

def collect_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Collecting predictions"):
            images = [img.to(device) for img in images]
            outputs = model.model(images)  # your wrapper
            # Convert to CPU for not running out of GPU memory
            batch_preds = []
            for out in outputs:
                batch_preds.append({
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu()
                })
            batch_targets = []
            for t in targets:
                batch_targets.append({
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu()
                })
            
            all_preds.append(batch_preds)
            all_targets.append(batch_targets)

    return all_preds, all_targets


def evaluate_model(model, val_loader, device):
    preds, targets = collect_predictions(model, val_loader, device)
    results = compute_metrics(preds, targets, CLASS_NAMES)
    return results

def train_model(train_loader, val_loader, device, num_classes=7, epochs=10, batch_size=1, eval_every=1):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    csv_file = os.path.join(log_dir, "result.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train/box_loss", "train/obj_loss", "train/cls_loss",
            "metrics/precision", "metrics/recall", "metrics/mAP_0.5",
            "val/box_loss", "val/obj_loss", "val/cls_loss", "learning_rate"
        ])
        
    #setup model
    model = FASTER_RCNN(num_classes)
    model.model.to(device)

    #* setup optimizer, maybe i need to add this in parser in the futrue
    for param in model.model.backbone.parameters():
        param.requires_grad = False
    params = [p for p in model.model.parameters() if p.requires_grad]
    LR = args.lr
    MOMENTUM = args.momentum
    WEIGHT_DECAY = args.weight_decay
    STEP_SIZE = args.step_size
    GAMMA = args.gamma
    optimizer = SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # Training log
    print("Training started.")
    print("Device: ", device)
    print("Number of classes: ", num_classes)
    print("Number of epochs: ", epochs)
    print("Batch size: ", batch_size)
    print("Dataset length: ", len(train_loader.dataset))
    print("Evaluation every: ", eval_every)

    best_map50 = 0.0
    best_model_wts = None

    for epoch in range(epochs):
        model.train()
        
        # setup losses
        epoch_losses = {
            'cls_loss': 0.0,
            'box_loss': 0.0,
            'obj_loss': 0.0
        }
        num_batches = len(train_loader)
        for i, (images, targets) in enumerate(train_loader, 1):
            #! TRAINING MODEL, return loss_dict = {'loss_classifier': loss_classifier, 'loss_box_reg': loss_box_reg, 'loss_objectness': loss_objectness}
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model.forward(images, targets)

            optimizer.zero_grad()
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            #!------

            # save losses for log
            loss_classifier = loss_dict['loss_classifier']
            loss_box_reg = loss_dict['loss_box_reg']
            loss_objectness = loss_dict['loss_objectness']

            # log epoch losses
            with open(os.path.join(log_dir, "training.log"), "a") as log_file:
                if (args.iter_every > 0) and (i % args.iter_every == 0 or i == len(train_loader) or i == 1):
                    print(
                        f"Epoch:[{epoch+1}/{epochs}]\t"
                        f"Iter:[{i}/{len(train_loader)}]\t"
                        f"cls_loss:{loss_classifier:.4f}\t"
                        f"box_loss:{loss_box_reg:.4f}\t"
                        f"obj_loss:{loss_objectness:.4f}"
                    )
                    log_file.write(f"Epoch:[{epoch+1}/{epochs}]\t"
                        f"Iter:[{i}/{len(train_loader)}]\t"
                        f"cls_loss:{loss_classifier:.4f}\t"
                        f"box_loss:{loss_box_reg:.4f}\t"
                        f"obj_loss:{loss_objectness:.4f}")
                    log_file.write(f"Targets: {targets}\n")


            # Update epoch losses
            epoch_losses['cls_loss'] += loss_dict['loss_classifier'].item()
            epoch_losses['box_loss'] += loss_dict['loss_box_reg'].item()
            epoch_losses['obj_loss'] += loss_dict['loss_objectness'].item()
        
        lr_scheduler.step()

        # Calculate average losses for epoch
        #! ====================== EVALUATION ================
        avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
        # if eval_every > 0 and (epoch + 1) % eval_every == 0:
        metrics = evaluate_model(model, val_loader, device)

        # ====================== VAL LOSS ================
        val_losses = {
            'cls_loss': 0.0,
            'box_loss': 0.0,
            'obj_loss': 0.0
        }
        val_batches = len(val_loader)
        model.model.train()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model.model(images, targets)

                val_losses['cls_loss'] += loss_dict['loss_classifier'].item()
                val_losses['box_loss'] += loss_dict['loss_box_reg'].item()
                val_losses['obj_loss'] += loss_dict['loss_objectness'].item()

        val_losses = {k: v/val_batches for k, v in val_losses.items()}

        # log avg losses
        print(
            f"================================= \n"
            f"[Evaluate epoch {epoch+1}] \n"
            f"Average cls_loss:{avg_losses['cls_loss']:.4f}\t"
            f"Average box_loss:{avg_losses['box_loss']:.4f}\t"
            f"Average obj_loss:{avg_losses['obj_loss']:.4f}"
        )
        # log metrics
        ap_per_class = metrics["ap50_per_class"]
        ar_per_class = metrics["ar50_per_class"]
        map50 = metrics["map50"]
        mar50 = metrics["mar50"]
        # Print AP@0.5 and AR@0.5 for each class
        print("\nPer-class AP/AR (IoU=0.5):")
        for idx, class_name in enumerate(CLASS_NAMES):
            ap50 = ap_per_class[idx].item()
            ar50 = ar_per_class[idx].item()
            # print(f"Class {idx} ({class_name}): AP@50={ap50:.4f}")
            print(f"Class {idx} ({class_name}): AP@50={ap50:.4f}, AR@50={ar50:.4f}")
        print(f"\nmAP@0.5 (all classes): {map50:.4f}")
        # print(f"mAR@0.5 (all classes): {mar50:.4f}")
        print("=================================")

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch+1,
                avg_losses['box_loss'],
                avg_losses['obj_loss'],
                avg_losses['cls_loss'],
                map50,
                mar50,
                map50,
                val_losses['box_loss'],
                val_losses['obj_loss'],
                val_losses['cls_loss'],
                optimizer.param_groups[0]['lr']
            ])

        if map50 > best_map50:
            best_map50 = map50
            best_model_wts = model.model.state_dict()
            print(f">>> New best model found at epoch {epoch+1} with mAP@0.5 = {map50:.4f}")

        # Save model at Epoch X
        # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # if not os.path.exists("weights"):
        #     os.makedirs("weights")
        # torch.save(model.model.state_dict(), f"weights/epoch{epoch}-faster-rcnn-{timestamp}.pt")


    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists("weights"):
        os.makedirs("weights")
    torch.save(model.model.state_dict(), f"weights/last-faster-rcnn-{timestamp}.pt")
    if best_model_wts is not None:
        torch.save(best_model_wts, f"weights/best-faster-rcnn.pt")
    print("Training complete.")

def main():
    FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.data_dir))
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    EVAL_EVERY = args.eval_every
    NUM_CLASSES = args.num_classes
    
    # Create dataset
    print("Loading dataset...")
    train_dataset = get_preprocessed_data(FOLDER_PATH, sub_dir="train", args=args)
    val_dataset = get_preprocessed_data(FOLDER_PATH, sub_dir="val", args=args)
    # Create Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(
        train_loader,
        val_loader,
        device,
        num_classes=NUM_CLASSES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        eval_every=EVAL_EVERY
    )

if __name__ == "__main__":
    main()