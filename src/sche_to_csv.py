import os
import pathlib
import json
import csv

json_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], "data", "schedule_demo2_data", "schedule.json")

# Load JSON data from a file
with open(json_path, "r") as json_file:
    data = json.load(json_file)

# Define CSV file name
csv_filename = "output.csv"

# Open CSV file for writing
with open(csv_filename, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header
    csv_writer.writerow(["robot_id", "node_id", "ETA"])

    # Process JSON data and write to CSV
    for robot_id, nodes in data.items():
        for node_id, eta in nodes:
            csv_writer.writerow([robot_id, node_id, eta])

print(f"CSV file '{csv_filename}' created successfully.")
