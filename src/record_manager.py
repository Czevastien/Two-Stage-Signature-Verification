import json
import requests
import os
import csv
import torch


class RecordManager:
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.epoch_record_path = self.cfg["result"]["record"]
        self.histogram_record_dir_path = self.cfg["histogram"]["dir"]

        self.epoch_record = {}
        self.load_record()

    def load_record(self):
        if os.path.exists(self.epoch_record_path):
            with open(self.epoch_record_path, "r") as f:
                try:
                    self.epoch_record = json.load(f)
                except json.JSONDecodeError:
                    self.epoch_record = {}
        else:
            self.epoch_record = {}

    def rewrite_record(self):
        with open(self.epoch_record_path, "w") as f:
            json.dump(self.epoch_record, f, indent=4)

    def broadcast_update(self, webhook_url):

        try:
            # Get the latest epoch key
            current_epoch = sorted(
                self.epoch_record.keys(), key=lambda x: int(x.split("-")[1])
            )[-1]

            # --- First message: send record.json ---
            try:
                with open(self.epoch_record_path, "rb") as f:
                    files = {"file": ("record.json", f)}
                    data = {
                        "content": f"📊 Epoch update: `{current_epoch}`\nAttached: `record.json`"
                    }
                    response = requests.post(webhook_url, data=data, files=files)

                if response.status_code in [200, 204]:
                    print("record.json sent successfully.")
                else:
                    print(
                        f"Failed to send record.json: {response.status_code}, {response.text}"
                    )
            except Exception as e:
                print(f"Error sending record.json: {e}")

            # --- Second message: send histogram file ---
            histogram_path = (
                f"{self.histogram_record_dir_path}/histogram-{current_epoch}.csv"
            )
            try:
                with open(histogram_path, "rb") as f:
                    files = {"file": (f"histogram-{current_epoch}.csv", f)}
                    data = {"content": f"📈 Histogram for `{current_epoch}`"}
                    response = requests.post(webhook_url, data=data, files=files)

                if response.status_code in [200, 204]:
                    print("Histogram sent successfully.")
                else:
                    print(
                        f"Failed to send histogram: {response.status_code}, {response.text}"
                    )
            except Exception as e:
                print(f"Error sending histogram: {e}")

        except Exception as e:
            print(f"Broadcast update failed: {e}")

    def update_epoch_record(self, data):
        self.epoch_record.update(data)
        self.rewrite_record()
        if self.cfg.get("discord", True) or True:
            self.broadcast_update(
                webhook_url = "https://discord.com/api/webhooks/1416080209186127983/Uv9lYeOj_ugqI5JmunbUrhxHelPydgCd6D_8ZCxUk5XiB2fYnjywZzdJdGufeSBoRCcq"
            )
            self.broadcast_update(
                webhook_url = "https://discord.com/api/webhooks/1430898499611791492/7RG1dlEXJDyJ9kpoIOQvKVhrDV3_0KgkAjlU6A5NLzIOtP1oTWoYS-q_fHzRUauwLADu"
            )

    def update_histogram_record(self, epoch, similarities):
        # Ensure the directory exists
        os.makedirs(self.histogram_record_dir_path, exist_ok=True)

        # Build the file path
        file_path = os.path.join(
            self.histogram_record_dir_path, f"histogram-epoch-{epoch}.csv"
        )

        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            for pos, neg in similarities:
                writer.writerow(pos + neg)
