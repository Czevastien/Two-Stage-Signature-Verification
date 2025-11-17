import os, time, datetime
import torch
import torchvision.transforms as transforms
import yaml

from triplet_dataset import (
    OfflinePremadeTripletDataset,
    OnlineHardTripletDataset,
    OnlineRandomTripletDataset,
)
from triplet_network import TripletNetwork
from triplet_loss import TripletLoss, BatchHardTripletLoss

import custom_transform

import train
import evaluate

from error_rate import ErrorRates
from checkpoint_loader import load_checkpoint
from record_manager import RecordManager

import json

import os

import math

import seed


def collate_fn(batch):
    # batch is a list of tuples: (pid, images, labels)
    pids, images, labels = zip(*batch)  # unzip into 3 tuples
    return list(pids), list(images), list(labels)


def generate_csv(cfg):
    from types import SimpleNamespace
    from csv_builder import CSVBuilder

    if not os.path.isfile(cfg["testing_ext"]["csv"]):
        config = SimpleNamespace()
        config.SETS_PER_PERSON = cfg["testing_ext"]["sets_per_person"]
        config.FORGERY_NEGATIVE_RATIO = cfg["testing_ext"]["forgery_negative_ratio"]
        config.EXTERNAL_NEGATIVE_RATIO = cfg["testing_ext"]["external_negative_ratio"]
        config.TARGET_CSV_FILENAME = cfg["testing_ext"]["csv"]
        config.DATASET_PATH = cfg["testing_ext"]["dir"]
        config.TYPE = cfg["testing_ext"]["type"]
        CSVBuilder(config).start()

    if not os.path.isfile(cfg["testing_int"]["csv"]):
        config = SimpleNamespace()
        config.SETS_PER_PERSON = cfg["testing_int"]["sets_per_person"]
        config.FORGERY_NEGATIVE_RATIO = cfg["testing_int"]["forgery_negative_ratio"]
        config.EXTERNAL_NEGATIVE_RATIO = cfg["testing_int"]["external_negative_ratio"]
        config.TARGET_CSV_FILENAME = cfg["testing_int"]["csv"]
        config.DATASET_PATH = cfg["testing_int"]["dir"]
        config.TYPE = cfg["testing_int"]["type"]
        CSVBuilder(config).start()

    if not os.path.isfile(cfg["training_hard"]["csv"]):
        config = SimpleNamespace()
        config.TARGET_CSV_FILENAME = cfg["training_hard"]["csv"]
        config.DATASET_PATH = cfg["training_hard"]["dir"]
        config.TYPE = cfg["training_hard"]["type"]
        CSVBuilder(config).start()

    if not os.path.isfile(cfg["training_random"]["csv"]):
        config = SimpleNamespace()
        config.TARGET_CSV_FILENAME = cfg["training_random"]["csv"]
        config.DATASET_PATH = cfg["training_random"]["dir"]
        config.TYPE = cfg["training_random"]["type"]
        CSVBuilder(config).start()


def start():
    global EER
    cfg = yaml.safe_load(open("./config.yaml"))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Config file:")

    print(
        "\n".join(
            "\t" + line
            for line in yaml.dump(
                cfg, sort_keys=False, default_flow_style=False
            ).splitlines()
        )
    )

    generate_csv(cfg)

    record_manager = RecordManager(cfg=cfg)

    hard_training_dataset = OnlineHardTripletDataset(
        csv_path=cfg["training_hard"]["csv"],
        img_dir=cfg["training_hard"]["dir"],
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                custom_transform.ResizePad(size=180),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
        cfg=cfg,
    )
    random_training_dataset = OnlineRandomTripletDataset(
        csv_path=cfg["training_random"]["csv"],
        img_dir=cfg["training_random"]["dir"],
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                custom_transform.ResizePad(size=180),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
        sets_per_person=cfg["training_random"]["sets_per_person"],
        forgery_negative_ratio=cfg["training_random"]["forgery_negative_ratio"],
        external_negative_ratio=cfg["training_random"]["external_negative_ratio"],
    )

    external_evaluation_dataset = OfflinePremadeTripletDataset(
        csv_path=cfg["testing_ext"]["csv"],
        img_dir=cfg["testing_ext"]["dir"],
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                custom_transform.ResizePad(size=180),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )

    internal_evaluation_dataset = OfflinePremadeTripletDataset(
        csv_path=cfg["testing_int"]["csv"],
        img_dir=cfg["testing_int"]["dir"],
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                custom_transform.ResizePad(size=180),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )

    hard_train_loader = torch.utils.data.DataLoader(
        hard_training_dataset,
        shuffle=True,
        num_workers=cfg["data_loader_n_worker"],
        batch_size=cfg["training_hard"]["batch_size"],
        collate_fn=collate_fn,
        generator=seed.set_seed(101)
    )

    random_train_loader = torch.utils.data.DataLoader(
        random_training_dataset,
        shuffle=True,
        num_workers=cfg["data_loader_n_worker"],
        batch_size=cfg["training_random"]["batch_size"],
        generator=seed.set_seed(101)
    )

    external_evaluation_loader = torch.utils.data.DataLoader(
        external_evaluation_dataset,
        shuffle=False,
        num_workers=cfg["data_loader_n_worker"],
        batch_size=cfg["testing_ext"]["batch_size"],
    )
    internal_evaluation_loader = torch.utils.data.DataLoader(
        internal_evaluation_dataset,
        shuffle=False,
        num_workers=cfg["data_loader_n_worker"],
        batch_size=cfg["testing_int"]["batch_size"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TripletNetwork(pretrained=False).to(device)
    eval_criterion = TripletLoss(
        margin=cfg["loss"]["margin"],
        pos_threshold=cfg["loss"]["pos_threshold"],
        neg_threshold=cfg["loss"]["neg_threshold"],
        punishment_scale=cfg["loss"]["punishment_scale"],
        pos_weight=cfg["loss"]["pos_weight"],
        neg_weight=cfg["loss"]["neg_weight"],
    )
    hard_train_criterion = BatchHardTripletLoss(margin=cfg["loss"]["margin"])
    random_train_criterion = TripletLoss(margin=cfg["loss"]["margin"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=1,
        eta_min=1e-6
    )


    resume_path = ""
    resume_latest = True

    if resume_latest and not resume_path:
        epoch_list = [
            int(x.split("-")[-1].strip(".pth"))
            for x in os.listdir(cfg["result"]["dir"])
            if ".pth" in x
        ]
        if epoch_list:
            resume_path = f"{cfg['result']['dir']}/epoch-{max(epoch_list)}.pth"

    if resume_path:
        start_epoch, _ = load_checkpoint(resume_path, net, device, optimizer, scheduler, cfg)
    else:
        start_epoch = 0

    epochs = cfg["epoch"]
    test_only = False

    for epoch in range(start_epoch + 1, epochs + 1):

        if test_only:
            # Set to True when testing results

            # Force all BatchNorm layers to always use batch stats, even in eval
            e_loss, e_losses, e_sims = evaluate.evaluate(
                device, optimizer, net, eval_criterion, external_evaluation_loader
            )
            metrics = ErrorRates(
                net, external_evaluation_loader, device, debug_mode=False
            ).calculate(threshold=0.5)
            metrics_internal = ErrorRates(
                net, internal_evaluation_loader, device, debug_mode=True
            ).calculate(threshold=0.5)
            result_data = {
                "loss": {
                    "evaluation-loss": e_loss,
                },
                "internal_metrics": metrics_internal,
                "metrics": metrics,
            }
            print(
                "\n".join(
                    "\t" + line
                    for line in yaml.dump(
                        result_data, sort_keys=False, default_flow_style=False
                    ).splitlines()
                )
            )
            break

        time_started = time.time()
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"\tTraining:")

        if epoch > 5:
            t_loss, t_losses = train.hard_train(
                device, optimizer, net, hard_train_criterion, hard_train_loader, cfg
            )
        else:
            t_loss, t_losses = train.random_train(
                device, optimizer, net, random_train_criterion, random_train_loader
            )

        print(f"\tEvaluating:")
        e_loss, e_losses, e_sims = evaluate.evaluate(
            device, optimizer, net, eval_criterion, external_evaluation_loader
        )

        print(f"\tCalculating metrics:")
        print(f"\t\tMetrics 0/2.", end="\r")
        metrics = ErrorRates(net, external_evaluation_loader, device).calculate(
            threshold=0.5
        )
        print(f"\t\tMetrics 1/2.", end="\r")
        metrics_internal = ErrorRates(
            net, internal_evaluation_loader, device
        ).calculate(threshold=0.5)
        print(f"\t\tMetrics calculated.")

        EER = metrics["error_rates"]["0.5"]["EER"]

        print(f"Epoch {epoch} trained. {' '*20}")

        scheduler.step()
        print(f"\tScheduler stepped. Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        result_data = {
            f"epoch-{epoch}": {
                "model_path": f"{cfg['result']['dir']}/epoch-{epoch}.pth",
                "time": {
                    "time_started": time_started,
                    "time_finished": time.time(),
                    "total_time": time.time() - time_started,
                },
                "loss": {
                    "training-loss": t_loss,
                    "evaluation-loss": e_loss,
                },
                "internal_metrics": metrics_internal,
                "metrics": metrics,
            }
        }

        record_manager.update_histogram_record(epoch=epoch, similarities=e_sims)
        record_manager.update_epoch_record(data=result_data)

        print(
            "\n".join(
                "\t" + line
                for line in yaml.dump(
                    result_data, sort_keys=False, default_flow_style=False
                ).splitlines()
            )
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": t_loss,
                "eval_loss": e_loss,
                "train_losses": t_losses,
                "eval_losses": e_losses,
                "scheduler_state_dict": scheduler.state_dict()
            },
            f"{cfg['result']['dir']}/epoch-{epoch}.pth",
        )
        print("Model Saved Successfully")


if __name__ == "__main__":

    start()

    # test12 = [1, 2, 3]

    # for i in range(100):
    #     print(test12[i%len(test12)])
