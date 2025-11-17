import torch
import time


def validity_check(debug_entry):
    validity = 0
    for a, p, n in list(zip(*debug_entry)):
        if any([x for x in (a, p) if a.split("_")[0] not in x]):
            validity += 1
        else:
            pass
            # validity+=1

        if a == p:
            validity += 1
        else:
            pass
            # validity+=1

        if (a.split("_")[0] in n) and ("_f" not in n):
            validity += 1
        else:
            pass
            # validity+=1

    return validity


def set_bn_to_train(module):
    """Keep only BatchNorm layers in train mode after net.eval()."""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()


def evaluate(device, optimizer, net, criterion, loader):
    """
    Evaluate one epoch with pre-generated triplets..
    """
    net.eval()
    net.apply(set_bn_to_train)  # <-- add this line

    losses = []
    similarities = []
    time_started_epoch = time.time()

    with torch.no_grad():
        for batch_idx, (anc, pos, neg, debug) in enumerate(loader, start=1):
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            out_a, out_p, out_n = net(anc, pos, neg)

            loss, similarity = criterion(out_a, out_p, out_n)
            losses.append(loss.item())
            similarities.append(similarity)

            elapsed = time.time() - time_started_epoch
            avg_time = elapsed / batch_idx
            remaining = len(loader) - batch_idx
            eta = avg_time * remaining

            print(
                f"\t\tBatch {batch_idx}/{len(loader)}\tLoss: {loss.item():.4f}\tETA: {eta:.2f}s\tErrors: {validity_check(debug)}{' '*20}",
                end="\r",
            )
        print(
            f"\t\tSuccessfully finished evaluating {len(loader)} batches at {torch.tensor(losses).mean().item()} loss.{' '*60}"
        )

    return torch.tensor(losses).mean().item(), losses, similarities
