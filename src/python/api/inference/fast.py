from ultralytics import YOLO
import cv2
from cap_from_youtube import cap_from_youtube
import pafy
import datetime
import json
import pandas as pd
import numpy as np
import io
from azure.storage.blob import ContainerClient
from fastapi import FastAPI, BackgroundTasks


def azure_initiate(
    result_blob: str,
    storage_connection_string: str,
):
    azure_client = ContainerClient.from_connection_string(
        storage_connection_string, result_blob
    )
    return azure_client


def calculate_percentage(bbox, original_shape):
    bbox_area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
    original_shape_area = original_shape[0] * original_shape[1]
    percentage = (bbox_area / original_shape_area) * 100
    return percentage


def most_frequent_class_and_name(x):
    mode_class = x["class"].mode()
    if len(mode_class) > 0:
        mode_class = mode_class[0]
        mode_name = x.loc[x["class"] == mode_class, "name"].iloc[0]
        return pd.Series([mode_class, mode_name])
    else:
        return pd.Series([np.nan, np.nan])


def summary(df, filename, result_blob):
    if (
        "track_id" in df.columns
        and df["track_id"].notna().any()
        and df["track_id"].ne(0).any()
    ):
        df_filtered = df[(df["track_id"] != 0) & (df["track_id"].notna())].copy()
        # Group by 'track_id' and calculate duration, most frequent class and
        # corresponding name for each group
        # Group by track_id and calculate average box_percentage, min and max timestamp
        summary_df = (
            df_filtered.groupby("track_id")
            .agg(
                average_box_percentage=("box_percentage", "mean"),
                min_timestamp=("timestamp", "min"),
                max_timestamp=("timestamp", "max"),
                most_common_class=(
                    "name",
                    lambda x: x.value_counts().index[0],
                ),  # Most common class per track_id
            )
            .reset_index()
        )
        # Calculate duration
        summary_df["duration"] = (
            summary_df["max_timestamp"] - summary_df["min_timestamp"]
        )

        # Convert the DataFrame to a string
        output_string = "\n".join(
            f"{row['most_common_class']} with id {row['track_id']} was present in the video for {row['duration']} from {row['min_timestamp']} to {row['max_timestamp']} and was taking  {row['average_box_percentage']:.2f}% of the screen"
            for _, row in summary_df.iterrows()
        )
    else:
        output_string = "No objects were detected in the video"

    results_txt_file_name = f"{filename}.txt"
    results_blob_client_txt = result_blob.get_blob_client(results_txt_file_name)
    results_blob_client_txt.upload_blob(output_string, overwrite=True)


def save_df(df, filename, result_blob):
    results_csv_file_name = f"{filename}.csv"
    results_blob_client = result_blob.get_blob_client(results_csv_file_name)
    csv_stream = io.StringIO()
    df.to_csv(csv_stream, index=False)
    # Convert the CSV data to bytes
    csv_bytes = csv_stream.getvalue().encode("utf-8")
    results_blob_client.upload_blob(csv_bytes, overwrite=True)


app = FastAPI()
# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Global variable to control the execution of the process
should_continue = True


def process(link: str, live: bool, container: str, storage_key: str):
    global should_continue
    should_continue = True

    # authentiacate in azure
    result_blob = azure_initiate(container, storage_key)

    # Process a youtube link:
    if not live:
        cap = cap_from_youtube(link, "720p")
    if live and ("rtsp" in link or "rtmp" in link or "tcp" in link):
        cap = cv2.VideoCapture(link)
    elif live:
        video = pafy.new(link)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)

    # we will store all the results as a list of dictionaries
    all_results = []
    timestamp = datetime.datetime.now()
    last_save_time = timestamp
    filename = (
        link.split("=")[-1] + "_" + str(timestamp.time().strftime("%Y-%m-%d-%H-%M-%S"))
    )
    while should_continue:
        # Loop through the video frameshttps://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/accesspolicy/storageAccountId/%2Fsubscriptions%2F99bc817b-d611-4ddb-b2d7-88793457d4fe%2Fresourcegroups%2Frg-salad-dev%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fsasaladyolodev/path/yolo-results/etag/%220x8DBDD85284C7A76%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # Run YOLOv8 inference on the frame
                results = model.track(frame, persist=True)
                timestamp = datetime.datetime.now()
                # save every box with label
                for box in json.loads(results[0].tojson()):
                    box["input"] = link
                    box["timestamp"] = timestamp
                    box["date"] = timestamp.strftime("%Y-%m-%d")
                    box["time"] = timestamp.time().strftime("%H:%M:%S")
                    box["origin_shape"] = results[0].orig_shape
                    box["box_percentage"] = calculate_percentage(
                        box["box"], results[0].orig_shape
                    )
                    box["full_process_speed"] = sum(results[0].speed.values())
                    all_results.append(box)

                    # Get the current time
                    current_time = datetime.datetime.now()
                    # Check if 30 minutes have passed since the last save
                    if (current_time - last_save_time).total_seconds() >= 30 * 60:
                        df = pd.DataFrame(all_results)
                        save_df(df, filename, result_blob)
                        summary(df, filename, result_blob)
                        last_save_time = current_time

                # Break the loop if 'q' is pressed
            if not should_continue:
                df = pd.DataFrame(all_results)
                save_df(df, filename, result_blob)
                summary(df, filename, result_blob)
                break
        if not should_continue:
            # Break the loop if the end of the video is reached
            break


@app.post("/start")
def start_process(
    background_tasks: BackgroundTasks,
    link: str,
    live: bool,
    container: str,
    storage_key: str,
):
    background_tasks.add_task(process, link, live, container, storage_key)
    return {"status": "Process started"}


@app.get("/stop")
def stop_process():
    global should_continue
    should_continue = False
    return {"status": "Process stopped"}


@app.get("/hc")
async def health_check():
    return "OK"
