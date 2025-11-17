import torch


def load_checkpoint(path, model, device, optimizer=None, scheduler=None, cfg=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    start_epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    print(f"Loading {path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"\tOptimizer state loaded.")

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"\tScheduler state loaded.")

    print(
        f"\tCheckpoint loaded.\n\tResuming from Epoch {start_epoch} out of {cfg['epoch']} at [{path}]"
    )
    return start_epoch, loss
