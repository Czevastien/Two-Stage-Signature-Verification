
from triplet_network import TripletNetwork
from triplet_dataset import OnlineRandomTripletDataset, OnlineHardTripletDataset
from triplet_loss import BatchHardTripletLoss, TripletLoss

import torchvision.transforms as transforms
import torch

import seed

def collate_fn(batch):
    # batch is a list of tuples: (pid, images, labels)
    pids, images, labels = zip(*batch)  # unzip into 3 tuples
    return list(pids), list(images), list(labels)

def generate_training_kit(device, n=1, cfg={}, sub_cfg=[], transform_sets=[]):
    """
    [
        transforms.Grayscale(num_output_channels=1),
        custom_transform.ResizePad(size=180),
        transforms.RandomRotation(degrees=3),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.05, scale=(0.01, 0.05)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
    """
    return_data = []

    for i in range(n):

        # ---- NETWORKS ----
        network = TripletNetwork().to(device)


        # ---- LOADERS ----
        # epoch 1-5
        random_training_dataset = OnlineRandomTripletDataset(
            csv_path=cfg[sub_cfg[i]]["training_random"]["csv"],
            img_dir=cfg[sub_cfg[i]]["training_random"]["dir"],
            transform=transforms.Compose(transform_sets[i]),
            sets_per_person=cfg[sub_cfg[i]]["training_random"]["sets_per_person"],
            forgery_negative_ratio=cfg[sub_cfg[i]]["training_random"]["forgery_negative_ratio"],
            external_negative_ratio=cfg[sub_cfg[i]]["training_random"]["external_negative_ratio"],
        )
        random_train_loader = torch.utils.data.DataLoader(
            random_training_dataset,
            shuffle=True,
            num_workers=cfg["data_loader_n_worker"],
            batch_size=cfg[sub_cfg[i]]["training_random"]["batch_size"],
            generator=seed.set_seed(101)
        )

        # epoch 6+
        hard_training_dataset = OnlineHardTripletDataset(
            csv_path=cfg[sub_cfg[i]]["training_hard"]["csv"],
            img_dir=cfg[sub_cfg[i]]["training_hard"]["dir"],
            transform=transforms.Compose(transform_sets[i]),
            sampler_size=cfg[sub_cfg[i]]["training_hard"]["epoch_multiplier"],
        )
        hard_train_loader = torch.utils.data.DataLoader(
            hard_training_dataset,
            shuffle=True,
            num_workers=cfg["data_loader_n_worker"],
            batch_size=cfg[sub_cfg[i]]["training_hard"]["batch_size"],
            collate_fn=collate_fn,
            generator=seed.set_seed(101)
        )
        
        # ---- CRITERION ----
        random_train_criterion = TripletLoss(margin=cfg["loss"]["margin"])
        hard_train_criterion = BatchHardTripletLoss(margin=cfg["loss"]["margin"])

        # ---- OPTIMIZER ----
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg["parameters"]["learning_rate"], weight_decay=cfg["parameters"]["weight_decay"])
    
        data = {
            "id": sub_cfg[i],
            "network": network,
            "loader": {
                "random": random_train_loader,
                "hard": hard_train_loader
            },
            "criterion": {
                "random": random_train_criterion,
                "hard": hard_train_criterion
            },
            "optimizer": optimizer
        }
        return_data.append(data)

    return return_data