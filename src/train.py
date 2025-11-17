import time
import torch
import matplotlib.pyplot as plt
import os


def hard_train(device, optimizer, net, criterion, loader, cfg):
    """
    Train one epoch with batch-hard triplet loss (handles variable number of images per person).

    Loader yields:
        pids: list of person IDs in this batch
        images_list: list of lists of images per person
        labels_list: list of lists of labels per person
    """
    net.train()
    losses = []

    time_started_epoch = time.time()

    chunk_size = cfg.get("training", {}).get(
        "sub_batch_size", 32
    )  # flexible sub-batch size

    for batch_idx, (pids_batch, images_list, labels_list) in enumerate(loader, start=1):
        # Flatten all images, labels, and pids across persons
        all_images = []
        all_labels = []
        all_pids = []

        for pid, imgs, lbls in zip(pids_batch, images_list, labels_list):
            all_images.extend(imgs)
            all_labels.extend(lbls)
            all_pids.extend([pid] * len(imgs))

        embeddings_list = []

        # Process images in sub-batches to avoid OOM
        for i in range(0, len(all_images), chunk_size):
            sub_images = torch.stack(all_images[i : i + chunk_size]).to(device)
            with torch.set_grad_enabled(True):
                sub_embeds = net.forward_once(sub_images)  # (sub_batch_size, D)
            embeddings_list.append(sub_embeds)

        # Concatenate all embeddings
        embeddings = torch.cat(embeddings_list, dim=0)
        labels_tensor = torch.tensor(all_labels, device=device)
        pids_tensor = torch.tensor(all_pids, device=device)

        # Compute loss and update
        optimizer.zero_grad()
        loss = criterion(embeddings, labels_tensor, pids_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(loss.item())

        # ETA
        elapsed = time.time() - time_started_epoch
        avg_time = elapsed / batch_idx
        remaining = len(loader) - batch_idx
        eta = avg_time * remaining

        print(
            f"\t\tBatch {batch_idx}/{len(loader)}\tBatch Loss: {loss.item():.4f}\tEpoch Loss: {torch.tensor(losses).mean().item():.4f}\tETA: {eta:.2f}s\tPIDs: {pids_batch}{' '*20}".replace(
                "\t", "        "
            ),
            end="\r",
        )

    print(
        f"\t\tSuccessfully finished training {len(loader)} batches at {torch.tensor(losses).mean().item()} loss.{' '*60}"
    )

    return torch.tensor(losses).mean().item(), losses


def random_train(device, optimizer, net, criterion, loader):
    """
    Train one epoch with online batch-hard triplet loss.
    """
    net.train()
    losses = []
    similarities = []

    time_started_epoch = time.time()

    for batch_idx, (idx, anc, pos, neg, debug) in enumerate(loader, start=1):
        # Move to device
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

        optimizer.zero_grad()

        # Forward pass
        out_a, out_p, out_n = net(anc, pos, neg)

        # Compute loss
        loss, similarity = criterion(out_a, out_p, out_n)

        # Backward pass and optimizer step
        if loss.item() > 1e-6:  # or some threshold
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
        else:
            # pass
            # TODO: Monitor the weighted result function and revert to this if cant work it out
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

        losses.append(loss.item())

        pos_sim, neg_sim = similarity

        similarities.append((pos_sim, neg_sim))

        elapsed = time.time() - time_started_epoch
        avg_time = elapsed / batch_idx
        remaining = len(loader) - batch_idx
        eta = avg_time * remaining

        print(
            f"\t\tBatch {batch_idx}/{len(loader)}\tLoss: {loss.item():.4f}\tETA: {eta:.2f}s{' '*20}",
            end="\r",
        )
    print(
        f"\t\tSuccessfully finished training {len(loader)} batches at {torch.tensor(losses).mean().item()} loss.{' '*60}"
    )

    return torch.tensor(losses).mean().item(), losses
