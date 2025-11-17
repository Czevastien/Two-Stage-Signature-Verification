import os
import yaml

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torchvision.transforms as transforms

import custom_transform
from checkpoint_loader import load_checkpoint
from error_rate import ErrorRates

from triplet_dataset import OfflinePremadeTripletDataset
from triplet_network import TripletNetwork

import json

def generate_loaders(transform_sets, cfg):
    for transform_set in transform_sets:
        external_evaluation_dataset = OfflinePremadeTripletDataset(
            csv_path=cfg["testing_ext"]["csv"],
            img_dir=cfg["testing_ext"]["dir"],
            transform=transforms.Compose(transform_set),
        )
        external_evaluation_loader = torch.utils.data.DataLoader(
            external_evaluation_dataset,
            shuffle=False,
            num_workers=cfg["data_loader_n_worker"],
            batch_size=cfg["testing_ext"]["batch_size"],
        )
        yield external_evaluation_loader


def start():
    cfg = yaml.safe_load(open("./config.yaml"))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    transform_sets = [
        [
            transforms.Grayscale(1),
            custom_transform.ResizePad(size=180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ],
        [
            transforms.Grayscale(num_output_channels=1),
            custom_transform.ResizePad(size=180),
            transforms.RandomRotation(degrees=2),
            transforms.RandomAffine(degrees=0, translate=(0.01, 0.01)),
            transforms.ColorJitter(brightness=0.02, contrast=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ],
        [
            transforms.Grayscale(num_output_channels=1),
            custom_transform.ResizePad(size=180),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(
                degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ],
        [
            transforms.Grayscale(num_output_channels=1),
            custom_transform.ResizePad(size=180),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=5,
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.01, 0.05), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ],
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TripletNetwork(pretrained=False).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

    resume_path = r"D:\SHELA\for-archive\TRANSFORMS\HEAVY\model\epoch-18.pth"

    load_checkpoint(resume_path, net, device, optimizer, cfg)

    output = {}

    for index, loader in enumerate(generate_loaders(transform_sets=transform_sets, cfg=cfg)):
        print(f"Testing Model #{index}")
        metrics = ErrorRates(net, loader, device, debug_mode=False).calculate(
            threshold=0.5
        )
        output[f"TS-{index}"] = metrics.copy()
    
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    start()
