import torch
import numpy as np


def set_bn_to_train(module):
    """Keep only BatchNorm layers in train mode after net.eval()."""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()


class ErrorRates:
    def __init__(self, net, loader, device, debug_mode=False):
        self.net = net
        self.loader = loader
        self.device = device
        self.data = {}
        self.debug_mode = debug_mode  # <-- new

    def calculate(self, threshold=0.5):
        self.data = {"error_rates": {}}

        self.data["error_rates"]["0.5"], best_threshold = self.calculate_false_rate(
            threshold=threshold
        )
        self.data["error_rates"][f"{best_threshold:.4f}"], _ = (
            self.calculate_false_rate(threshold=best_threshold)
        )
        self.calculate_auc()
        return self.data

    def calculate_false_rate(self, threshold=0.5):
        cos = torch.nn.CosineSimilarity(dim=1)

        genuine_scores = []
        impostor_scores = []

        with torch.no_grad():
            for batch_idx, (anc, pos, neg, debug) in enumerate(self.loader, start=1):
                anc, pos, neg = (
                    anc.to(self.device),
                    pos.to(self.device),
                    neg.to(self.device),
                )

                # --- optional train-mode debug ---
                if self.debug_mode and batch_idx == 1:
                    self.net.eval()
                    out_a, out_p, out_n = self.net(anc, pos, neg)
                    sim_pos = cos(out_a, out_p).cpu().numpy()  # genuine
                    sim_neg = cos(out_a, out_n).cpu().numpy()  # impostor

                    self.net.train()
                    out_a_tr, out_p_tr, out_n_tr = self.net(anc, pos, neg)
                    sim_pos_tr = cos(out_a_tr, out_p_tr).cpu().numpy()
                    sim_neg_tr = cos(out_a_tr, out_n_tr).cpu().numpy()

                    debug_triplets = list(zip(*debug))

                    print("\n[DEBUG COMPARISON]")
                    print("Batch:", batch_idx)
                    print("Anchor/Pos sims (eval):", sim_pos[:5])
                    print("Anchor/Pos sims (train):", sim_pos_tr[:5])
                    print("Anchor/Neg sims (eval):", sim_neg[:5])
                    print("Anchor/Neg sims (train):", sim_neg_tr[:5])
                    print("Diff Pos:", (sim_pos[:5] - sim_pos_tr[:5]))
                    print("Diff Neg:", (sim_neg[:5] - sim_neg_tr[:5]))
                    print("File triplets:", debug_triplets)
                    print("-" * 50)

                else:
                    # --- eval mode ---
                    self.net.eval()
                    self.net.apply(set_bn_to_train)

                    out_a, out_p, out_n = self.net(anc, pos, neg)

                    sim_pos = cos(out_a, out_p).cpu().numpy()  # genuine
                    sim_neg = cos(out_a, out_n).cpu().numpy()  # impostor

                genuine_scores.extend(sim_pos)
                impostor_scores.extend(sim_neg)

        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)

        # --- FRR and FAR ---
        FRR = float(np.mean(genuine_scores < threshold) * 100.0)
        FAR = float(np.mean(impostor_scores >= threshold) * 100.0)

        # --- EER ---
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate(
            [np.ones_like(genuine_scores), np.zeros_like(impostor_scores)]
        )

        thresholds = np.linspace(all_scores.min(), all_scores.max(), 200)
        frr_list, far_list = [], []

        for t in thresholds:
            frr_list.append(np.mean(genuine_scores < t))
            far_list.append(np.mean(impostor_scores >= t))

        frr_list, far_list = np.array(frr_list), np.array(far_list)
        eer_idx = np.argmin(np.abs(frr_list - far_list))
        EER = float((frr_list[eer_idx] + far_list[eer_idx]) / 2.0 * 100.0)
        best_threshold = float(thresholds[eer_idx])

        self.labels = labels
        self.all_scores = all_scores

        return {
            "FRR": FRR,
            "FAR": FAR,
            "EER": EER,
        }, best_threshold

    def calculate_auc(self):
        labels = self.labels
        scores = self.all_scores

        # sort scores high→low
        desc_idx = np.argsort(-scores)
        scores, labels = scores[desc_idx], labels[desc_idx]

        tps = np.cumsum(labels == 1)
        fps = np.cumsum(labels == 0)

        tpr = tps / (tps[-1] if tps[-1] > 0 else 1)
        fpr = fps / (fps[-1] if fps[-1] > 0 else 1)

        auc = np.trapz(tpr, fpr)
        self.data["AUC"] = float(auc)
