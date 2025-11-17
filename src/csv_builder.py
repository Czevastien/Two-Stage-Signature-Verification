import random
import csv
import os


class CSVBuilder:
    def __init__(self, config_src):
        self.catalog = {}
        self.sample_usage = {}
        self.config_src = config_src

    def start(self):
        with open(self.config_src.TARGET_CSV_FILENAME, mode="w", newline="") as file:
            writer = csv.writer(file)
            if self.config_src.TYPE == 0:
                writer.writerow(["person_id", "signature_id", "label", "filename"])
            elif self.config_src.TYPE == 1:
                writer.writerow(["Anchor", "Positive", "Negative"])

        self.generate_catalog()
        self.write_data()

    def generate_catalog(self):
        source = self.config_src.DATASET_PATH

        person_directory = sorted(
            [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))],
            key=lambda x: int(x),
        )

        for person in person_directory:
            genuines = sorted(
                [
                    x
                    for x in os.listdir(os.path.join(source, person))
                    if "_f" not in x and x.endswith(".png")
                ],
                key=lambda x: int("".join([y for y in x if y.isnumeric()])),
            )
            forgery = sorted(
                [
                    x
                    for x in os.listdir(os.path.join(source, person))
                    if "_f" in x and x.endswith(".png")
                ],
                key=lambda x: int("".join([y for y in x if y.isnumeric()])),
            )
            self.catalog[person] = {"genuine": genuines, "forgery": forgery}

    def get_weighted_choice(self, value_dict, do_not_pick=[]):
        try:
            max_val = max(value_dict.values())
        except:
            max_val = 0
        weights = []

        for key, value in value_dict.items():
            if key in do_not_pick:
                weights.append(0)
            else:
                weights.append(max_val - value + 1)

        choice = random.choices(list(value_dict.keys()), weights=weights, k=1)[0]
        value_dict[choice] += 1
        return choice

    def generate_triplets(self):
        global_triplets = []

        global_positive_sample = {
            sample: 0 for pid in self.catalog for sample in self.catalog[pid]["genuine"]
        }

        for pid in self.catalog.keys():
            local_triplets = []

            genuine_sample = {d: 0 for d in self.catalog[pid]["genuine"]}
            forged_sample = {d: 0 for d in self.catalog[pid]["forgery"]}

            anchor_index = 0
            internal_negative_count = (
                self.config_src.SETS_PER_PERSON
                / (
                    self.config_src.FORGERY_NEGATIVE_RATIO
                    + self.config_src.EXTERNAL_NEGATIVE_RATIO
                )
                * self.config_src.FORGERY_NEGATIVE_RATIO
            )

            while len(local_triplets) < self.config_src.SETS_PER_PERSON:
                anchor = list(genuine_sample.keys())[anchor_index]
                positive = self.get_weighted_choice(
                    genuine_sample, do_not_pick=[anchor]
                )
                if len(forged_sample) == 0:
                    internal_negative_count = 0

                if internal_negative_count > 0:
                    negative = self.get_weighted_choice(forged_sample)
                    internal_negative_count -= 1
                else:
                    negative = self.get_weighted_choice(global_positive_sample)
                local_triplets.append([anchor, positive, negative])
                anchor_index += (
                    1
                    if anchor_index < len(genuine_sample) - 1
                    else -(len(genuine_sample) - 1)
                )

            global_triplets.extend(local_triplets)

        return global_triplets

    def write_data(self):

        if self.config_src.TYPE == 0:
            for pid in self.catalog.keys():
                for file in [
                    *self.catalog[pid]["genuine"],
                    *self.catalog[pid]["forgery"],
                ]:
                    sid = file.strip(".png").split("_")[1]
                    label = 1 if file in self.catalog[pid]["genuine"] else 0
                    entry = [pid, sid, label, file]
                    with open(
                        self.config_src.TARGET_CSV_FILENAME, mode="a", newline=""
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(entry)
        elif self.config_src.TYPE == 1:
            triplets = self.generate_triplets()
            with open(
                self.config_src.TARGET_CSV_FILENAME, mode="a", newline=""
            ) as file:
                writer = csv.writer(file)
                writer.writerows(triplets)


if __name__ == "__main__":
    from types import SimpleNamespace

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    config = SimpleNamespace()
    config.TYPE = 1
    config.TARGET_CSV_FILENAME = "train_data_test.csv"
    config.DATASET_PATH = "test"
    config.SETS_PER_PERSON = 2
    config.FORGERY_NEGATIVE_RATIO = 1
    config.EXTERNAL_NEGATIVE_RATIO = 1
    sample = CSVBuilder(config)
    sample.start()
