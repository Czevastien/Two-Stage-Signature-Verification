import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(torch.nn.Module):
    """
    Triplet loss with cosine similarity and margin.
    NTS: Don't use Euclidean...

    Formula:
        L = average( max( cos_sim(a,n) - cos(a,p) + margin, 0 ) )

    Range:
        - Best case (perfect separation):
            cos(a,p) = 1, cos(a,n) = -1
            → -1 - 1 + margin = -2 + margin
            → Normalized to 0 (lowest possible loss)
        - Worst case (practically the opposite):
            cos(a,p) = -1, cos(a,n) = 1
            → 1 - (-1) + margin = 2 + margin (e.g. 2.5 if margin=0.5)

    Interpretation:
        - High loss → model is penalized (positives not close enough or negatives too close)
        - Low loss  → model is learning well (positives cluster together, negatives are pushed away)

    Where:
        - cos(a,p): cosine similarity between anchor & positive (should be high, close to 1)
        - cos(a,n): cosine similarity between anchor & negative (should be low, near 0 or negative)

    Cosine similarity meaning:
        -  1   → identical embeddings (perfect genuine match)
        -  0   → unrelated or no clear identical relationship
        - -1   → very different (perfect opposite of genuine match if a word for it exists lol)
    """

    def __init__(
        self,
        margin: float = 0.5,
        pos_threshold: float = 0.0,
        neg_threshold: float = 0.0,
        punishment_scale: float = 2.0,
        pos_weight=0.5,
        neg_weight=0.5,
    ):
        super().__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=1)
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.punishment_scale = punishment_scale
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, anchor, positive, negative):
        # Cosine similarities
        sim_pos = self.cos(anchor, positive)  # (B,)
        sim_neg = self.cos(anchor, negative)  # (B,)

        # Standard triplet loss
        base_loss = F.relu(sim_neg - sim_pos + self.margin)  # (B,)

        # Mistake scaler
        weights = torch.ones_like(base_loss)
        weights = torch.where(
            sim_pos < self.pos_threshold, self.punishment_scale, weights
        )
        weights = torch.where(
            sim_neg > self.neg_threshold, self.punishment_scale, weights
        )

        # Positive pull
        pos_loss = 1 - sim_pos  # want sim_pos → 1

        # Negative push
        neg_loss = F.relu(sim_neg + 0.2)  # want sim_neg → ≤ -0.2

        # Weighted loss
        losses = (
            (base_loss * weights)
            + (self.pos_weight * pos_loss)
            + (self.neg_weight * neg_loss)
        )

        return losses.mean(), (sim_pos.detach().tolist(), sim_neg.detach().tolist())


# class BatchHardTripletLoss(torch.nn.Module):
#     """
#     Batch-hard triplet loss using cosine similarity.
#     For each anchor:
#       - hardest positive = positive with lowest similarity (within same PID)
#       - hardest negative = negative with highest similarity (different PID)
#     """
#     def __init__(self, margin=0.5):
#         super().__init__()
#         self.margin = margin

#     def forward(self, embeddings, labels, pids):
#         """
#         embeddings: (N, D)
#         labels: (N,) - 0/1 for genuine/forgery
#         pids: (N,) - person IDs
#         """
#         N = embeddings.size(0)

#         # Compute cosine similarity matrix
#         sim_matrix = torch.matmul(
#             F.normalize(embeddings, p=2, dim=1),
#             F.normalize(embeddings, p=2, dim=1).t()
#         )  # (N, N)

#         loss_all = []

#         for i in range(N):
#             if labels[i] == 0:
#                 continue
#             anchor_pid = pids[i]

#             # mask positives and negatives
#             pos_mask = (pids == anchor_pid) & (labels == 1) & (torch.arange(N, device=pids.device) != i)
#             neg_mask = ((pids == anchor_pid) & (labels == 0)) | (pids != anchor_pid)

#             if pos_mask.sum() == 0 or neg_mask.sum() == 0:
#                 continue  # skip if no valid pairs

#             # hardest positive = lowest similarity within same PID
#             hardest_pos = sim_matrix[i][pos_mask].min()

#             # hardest negative = highest similarity outside PID
#             hardest_neg = sim_matrix[i][neg_mask].max()

#             # Triplet loss
#             loss = F.relu(hardest_neg - hardest_pos + self.margin)
#             loss_all.append(loss)

#         if len(loss_all) == 0:
#             return embeddings.new_tensor(0.0, requires_grad=True)

#         return torch.stack(loss_all).mean()


class BatchHardTripletLoss(torch.nn.Module):
    """
    Batch-hard triplet loss using cosine similarity.
    For each anchor:
      - hardest positive = positive with lowest similarity (within same PID)
      - hardest negative = negative with highest similarity (different PID)
    """

    def __init__(
        self,
        margin=0.5,
        pos_weight=0.5,
        neg_weight=0.5,
        punishment_scale=2.0,
        pos_threshold=0.6,
        neg_threshold=0.3,
    ):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.punishment_scale = punishment_scale
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, embeddings, labels, pids):
        """
        embeddings: (N, D)
        labels: (N,) - 0/1 (1=genuine, 0=forgery)
        pids:   (N,) - person IDs
        """
        N = embeddings.size(0)

        # normalize before similarity
        normed = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(normed, normed.t())  # (N, N)

        loss_all = []

        for i in range(N):
            if labels[i] == 0:  # skip anchors that are forgeries
                continue
            anchor_pid = pids[i]

            # positives = other genuines of same PID
            pos_mask = (
                (pids == anchor_pid)
                & (labels == 1)
                & (torch.arange(N, device=pids.device) != i)
            )

            # negatives = forgeries of same PID OR any sample from another PID
            forgery_mask = (pids == anchor_pid) & (labels == 0)
            cross_mask = pids != anchor_pid
            neg_mask = forgery_mask | cross_mask

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            # hardest positive = lowest similarity among same PID genuines
            hardest_pos = sim_matrix[i][pos_mask].min()

            neg_sims = sim_matrix[i][neg_mask]
            semi_hard_neg_mask = (neg_sims < hardest_pos) & (
                neg_sims > hardest_pos - self.margin
            )

            # hardest negative = highest similarity among forgery/cross negatives
            # hardest_neg = sim_matrix[i][neg_mask].max()
            if semi_hard_neg_mask.sum() > 0:
                hardest_neg = neg_sims[semi_hard_neg_mask].max()
            else:
                # if no semi-hard found, pick the closest negative that’s still > hardest_pos
                candidates = neg_sims[neg_sims > hardest_pos]
                if candidates.numel() > 0:
                    hardest_neg = candidates.max()
                else:
                    hardest_neg = neg_sims.max()  # last-resort fallback

            # --- base triplet loss ---
            base_loss = F.relu(hardest_neg - hardest_pos + self.margin)

            # --- mistake scaler ---
            weight = torch.tensor(1.0, device=embeddings.device)
            if hardest_pos < self.pos_threshold:
                weight = self.punishment_scale
            if hardest_neg > self.neg_threshold:
                weight = self.punishment_scale

            # --- positive pull ---
            pos_loss = 1.0 - hardest_pos  # push sim_pos -> 1

            # --- negative push ---
            neg_loss = F.relu(hardest_neg + 0.2)  # push sim_neg <= -0.2

            # --- weighted combined loss ---
            loss = (
                (base_loss * weight)
                + (self.pos_weight * pos_loss)
                + (self.neg_weight * neg_loss)
            )

            loss_all.append(loss)

        if len(loss_all) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(loss_all).mean()
