import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Parser for Faster R-CNN training")
    parser.add_argument("--data_dir", type=str, default="data/split", help="Path to dataset folder")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Frequency of evaluation")
    parser.add_argument("--iter_every", type=int, default=1, help="Frequency of iteration")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--yaml", type=str, default="data/data-ppe.yaml", help="Path to yaml file")
    # fine tune the model
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--step_size", type=int, default=3, help="Step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma")
    # pre processing data
    parser.add_argument("--resize", type=int, default=224, help="Resize image")
    parser.add_argument("--is_aug", type=float, default=False, help="is Augmentation?")

    return parser


def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    return args