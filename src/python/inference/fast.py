from fastapi import FastAPI, BackgroundTasks
from azure.storage.blob import ContainerClient
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube
import pafy
import datetime
import json
import csv
import os
import io


# Define a function to get the most frequent class and corresponding name
def most_frequent_class_and_name(x):
    mode_class = x["class"].mode()
    if len(mode_class) > 0:
        mode_class = mode_class[0]
        mode_name = x.loc[x["class"] == mode_class, "name"].iloc[0]
        return pd.Series([mode_class, mode_name])
    else:
        return pd.Series([np.nan, np.nan])


def azure_initiate(
    result_blob: str,
    storage_connection_string: str,
):
    azure_client = ContainerClient.from_connection_string(
        storage_connection_string, result_blob
    )
    return azure_client


app = FastAPI()

# Global variable to control the execution of the process
should_continue = True


# Load the YOLOv8 model
model = YOLO("yolov8n.pt")


def process(URL: str, source: str, type: str, container: str, storage_key: str):
    global should_continue
    should_continue = True
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # # Process a video file
    # video_path = "https://youtu.be/LNwODJXcvt4"
    # cap = cv2.VideoCapture(video_path)

    # authentiacate in azure
    result_blob = azure_initiate(container, storage_key)

    # Process a youtube link:
    if source == "youtube" and type != "live":
        cap = cap_from_youtube(URL, "720p")

    # # Process a streaming video
    if source == "youtube" and type == "live":
        # URL='https://www.youtube.com/watch?v=pu6kHxiCSY0'
        video = pafy.new(URL)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)

    # we will store all the results as a list of dictionaries
    all_results = []

    while should_continue:
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # Run YOLOv8 inference on the frame
                results = model.track(frame, persist=True)
                timestamp = datetime.datetime.now()
                # save every box with label
                for box in json.loads(results[0].tojson()):
                    box["input"] = URL
                    box["timestamp"] = timestamp
                    box["date"] = timestamp.strftime("%Y-%m-%d")
                    box["time"] = timestamp.time().strftime("%H:%M:%S")
                    all_results.append(box)

            # Break the loop if the process should not continue
            if not should_continue:
                filename = (
                    URL.split("=")[-1]
                    + "_"
                    + str(timestamp.time().strftime("%Y-%m-%d-%H-%M-%S"))
                )
                df = pd.DataFrame(all_results)
                results_csv_file_name = f"{filename}.csv"
                results_blob_client = result_blob.get_blob_client(results_csv_file_name)
                csv_stream = io.StringIO()
                df.to_csv(csv_stream, index=False)
                # Convert the CSV data to bytes
                csv_bytes = csv_stream.getvalue().encode("utf-8")
                results_blob_client.upload_blob(csv_bytes, overwrite=True)

                df_filtered = df[
                    (df["track_id"] != 0) & (df["track_id"].notna())
                ].copy()
                df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"])

                # Group by 'track_id' and calculate duration, most frequent class and
                # corresponding name for each group
                timestamp_grouped = df_filtered.groupby("track_id")["timestamp"].agg(
                    ["min", "max"]
                )
                class_name_grouped = df_filtered.groupby("track_id").apply(
                    most_frequent_class_and_name
                )

                # Reset index for both groupby results
                timestamp_grouped.reset_index(inplace=True)
                class_name_grouped.reset_index(inplace=True)

                # Rename columns for class_name_grouped
                class_name_grouped.columns = ["track_id", "class", "name"]

                # Merge the two groupby results
                grouped = pd.merge(timestamp_grouped, class_name_grouped, on="track_id")

                # Calculate duration
                grouped["duration"] = grouped["max"] - grouped["min"]

                # Final DataFrame with 'track_id', 'min_timestamp', 'max_timestamp',
                # 'duration', 'class', and 'name'
                final_df = grouped.rename(
                    columns={"min": "min_timestamp", "max": "max_timestamp"}
                )

                # Convert the DataFrame to a string
                output_string = "\n".join(
                    f"{row['name']} with id {row['track_id']} was present in the video for {row['duration']} from {row['min_timestamp']} to {row['max_timestamp']}"
                    for _, row in final_df.iterrows()
                )
                results_txt_file_name = f"{filename}.txt"
                results_blob_client_txt = result_blob.get_blob_client(
                    results_txt_file_name
                )
                results_blob_client_txt.upload_blob(output_string, overwrite=True)
                break
        if not should_continue:
            break


@app.post("/start")
def start_process(
    background_tasks: BackgroundTasks,
    url: str,
    source: str,
    type: str,
    container: str,
    storage_key: str,
):
    background_tasks.add_task(process, url, source, type, container, storage_key)
    return {"status": "Process started"}


@app.get("/stop")
def stop_process():
    global should_continue
    should_continue = False
    return {"status": "Process stopped"}
