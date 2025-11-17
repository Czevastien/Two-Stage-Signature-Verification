import cv2
import numpy as np
 
import pandas as pd
 
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy.spatial.distance import cdist
 
import os
from multiprocessing import Pool
import csv
 
 
class KNN:
    def __init__(self):
        self.dataframe = None
        self.PID_list = []
        self.data = {}
 
        self._setup()
 
    def _setup(self):
        csv_path = "data/hard_train_data.csv"
        self.dataframe = pd.read_csv(csv_path)
        self.PID_list = (
            self.dataframe["person_id"].unique().astype(int).tolist()
        )  # [:10]
 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                                                     #
    #    Pre-calculation                                                                  #
    #       - Pre-calculates all embeddings as returned by Harris Corner Detection to     #
    #         save time during benchmarking.                                              #
    #                                                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 
    @staticmethod
    def _extract_harris_features(img_path):
        """Extract Harris corner descriptors aggregated into a histogram."""
 
        IMG_SIZE = (180, 180)
 
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(256)
 
        img = cv2.resize(img, IMG_SIZE)
        harris = cv2.cornerHarris(np.float32(img), 2, 3, 0.04)
        harris = cv2.dilate(harris, None)
        corners = np.argwhere(harris > 0.01 * harris.max())
 
        orb = cv2.ORB_create(nfeatures=200)
        keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 3) for x in corners]
        keypoints, desc = orb.compute(img, keypoints)
 
        if desc is None:
            return np.zeros(256)
 
        hist, _ = np.histogram(desc, bins=256, range=(0, 256))
        return hist / (np.linalg.norm(hist) + 1e-8)
 
    def pre_calculate_embeddings(self):
        for pid in self.PID_list:
            print(f"Processing PID: {pid}/{len(self.PID_list)}")
            signature_sets = self._get_signature_sets(pid)
 
            paths = [f"train/{pid}/{sig['filename']}" for sig in signature_sets]
 
            with Pool(processes=os.cpu_count()) as pool:
                embeddings = pool.map(self._extract_harris_features, paths)
 
            for sig, emb in zip(signature_sets, embeddings):
                sig["embedding"] = emb.tolist()
 
            self.data[str(pid)] = signature_sets
 
    def _get_signature_sets(self, PID: int):
        # returns index of all signature dataset entry from the PID
        data = self.dataframe.index[self.dataframe["person_id"] == PID].tolist()
        datas = [self.dataframe.iloc[idx].to_dict() for idx in data]
        return datas
 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                                                     #
    #    Metric Calculation                                                               #
    #      - Calculates the error rate given a distance metric and k value                #
    #      - Uses dynamic threshold depending on the references.                          #
    #      - Skips dataset samples where k value is greater than (n - 1)                  #
    #         - On inference time, KNN will be skiped when using less than 3 referenes    #
    #           as using 2 or less reference means k-value can only be 1 and that's       #
    #           nearest neighbor and not k-nearest neighbor.                              #
    #                                                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 
    def _calculate_dynamic_threshold(self, data, distance_metric, threshold_margin=2):
        embeddings = [x["embedding"] for x in data if x["label"] == 1]  # genuine only
 
        pairwise_dist = cdist(embeddings, embeddings, metric=distance_metric)
        np.fill_diagonal(pairwise_dist, np.nan)
 
        valid_distances = pairwise_dist[~np.isnan(pairwise_dist)]
 
        # print(f"distances: ", set([round(x*10000)/10000 for x in valid_distances]))
        # print(f"sid: ", [x["signature_id"] for x in data])
 
        mean_dist = np.mean(valid_distances)
        std_dist = np.std(valid_distances)
        threshold = mean_dist + (std_dist * threshold_margin)
 
        return threshold
 
    def _calculate_distance(
        self, unknown_signature, reference_signature, distance_metric
    ):
        return cdist(
            [unknown_signature], [reference_signature], metric=distance_metric
        )[0][0]
 
    def calculate_KNN(self, k_value, distance_metric):
        output_data = []
        for pid, data in self.data.items():
            for sig_index, unknown_signature in enumerate(data):
 
                if len(data) <= k_value:
 
                    temp_output_data = {
                        "person_id": unknown_signature["person_id"],
                        "signature_id": unknown_signature["signature_id"],
                        "label": unknown_signature["label"],
                        "filename": unknown_signature["filename"],
                        "threshold_calculated": None,
                        "neighbor_mean": None,
                        "decision": None,
                        "is_correct": None,
                    }
                    output_data.append(temp_output_data)
 
                    continue
 
                popped_data = [
                    x
                    for x in data
                    if not (
                        x["signature_id"] == unknown_signature["signature_id"]
                        and x["label"] == unknown_signature["label"]
                    )
                    and x["label"] == 1
                ]
                threshold = self._calculate_dynamic_threshold(
                    data=popped_data,
                    distance_metric=distance_metric,
                    threshold_margin=1.2,
                )
 
                distances = [
                    self._calculate_distance(
                        unknown_signature["embedding"],
                        ref["embedding"],
                        distance_metric,
                    )
                    for ref in popped_data
                ]
 
                neighbors = sorted(distances)[:k_value]
 
                predicted_label = 1 if np.mean(neighbors) < threshold else 0
                true_label = unknown_signature["label"]
 
                temp_output_data = {
                    "person_id": unknown_signature["person_id"],
                    "signature_id": unknown_signature["signature_id"],
                    "label": unknown_signature["label"],
                    "filename": unknown_signature["filename"],
                    "threshold_calculated": threshold,
                    "neighbor_mean": np.mean(neighbors),
                    "decision": predicted_label,
                    "is_correct": 1 if predicted_label == true_label else 0,
                }
                output_data.append(temp_output_data)
 
        df = pd.DataFrame(output_data)
        df.to_csv(f"KNN_data/trial-{distance_metric}-k={k_value}.csv", index=False)
 
        return output_data
 
    def calculate_metrics(self, data):
        # eer, far, frr, auc, accuracy, precision, recall, f1-score
 
        metric_data = {}
 
        metric_data["total_samples"] = len(data)
 
        valid_samples = [row for row in data if row["decision"] in [1, 0]]
 
        metric_data["valid_samples"] = len(valid_samples)
 
        correct_prediction = [row for row in data if row["is_correct"] == 1]
 
        false_acceptance = [
            row for row in data if row["label"] == 0 and row["decision"] == 1
        ]
        false_rejection = [
            row for row in data if row["label"] == 1 and row["decision"] == 0
        ]
 
        true_acceptance = [
            row for row in data if row["label"] == 1 and row["decision"] == 1
        ]
        true_rejection = [
            row for row in data if row["label"] == 0 and row["decision"] == 0
        ]
 
        total_negatives = [
            row for row in data if row["label"] == 0 and row["decision"] in [1, 0]
        ]
        total_positives = [
            row for row in data if row["label"] == 1 and row["decision"] in [1, 0]
        ]
 
        metric_data["accuracy"] = (
            len(correct_prediction) / len(valid_samples) if valid_samples else 0
        )
        metric_data["precision"] = (
            len(true_acceptance) / (len(true_acceptance) + len(false_acceptance))
            if true_acceptance or false_acceptance
            else 1
        )
 
        metric_data["FRR"] = (
            len(false_rejection) / len(total_positives) if total_positives else 0
        )
        metric_data["FAR"] = (
            len(false_acceptance) / len(total_negatives) if total_negatives else 0
        )
 
        metric_data["TAR"] = 1 - metric_data["FRR"]
 
        metric_data["TRR"] = 1 - metric_data["FAR"]
 
        metric_data["F!-Score"] = (
            2
            * (metric_data["precision"] * metric_data["TAR"])
            / (metric_data["precision"] + metric_data["TAR"])
        )
 
        return metric_data
 
 
def start():
    distance_metrics = [
        # Recommend ko
        "cosine",
        "correlation",
        "cityblock",
        "braycurtis",
        "canberra",
 
        # Other common options for distance metrics
        "euclidean",
        "chebyshev",
        "sqeuclidean",
        
        # Stupid choices but included for completeness
        "jaccard",
        "dice",
        "hamming",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
    ]
 
    k_values = [1, 3, 5, 7, 9, 11, 13]
 
    final_output_data = []
 
    temp = KNN()
    temp.pre_calculate_embeddings()
 
    for distance_metric in distance_metrics:
        print(f"Testing {distance_metric}")
        for k_value in k_values:
            print(f"    with k = {k_value}")
 
            output_data = temp.calculate_KNN(k_value, distance_metric)
            metric = temp.calculate_metrics(data=output_data)
 
            temp_output_data = {"distance_metric": distance_metric, "k_value": k_value}
            final_output_data.append({**temp_output_data, **metric})
            # break
        # break
 
    df = pd.DataFrame(final_output_data)
    df.to_csv(f"KNN_data/final_output.csv", index=False)
 
 
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
    start()
