import pandas as pd

# read data from csv file
locaL_file_name = "pu6kHxiCSY0_2023-10-3115.csv"
df = pd.read_csv(locaL_file_name)

import datetime


df_filtered = df[(df["track_id"] != 0) & (df["track_id"].notna())].copy()
df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"])


# Define a function to get the most frequent class and corresponding name
def most_frequent_class_and_name(x):
    mode_class = x["class"].mode()
    if len(mode_class) > 0:
        mode_class = mode_class[0]
        mode_name = x.loc[x["class"] == mode_class, "name"].iloc[0]
        return pd.Series([mode_class, mode_name])
    else:
        return pd.Series([np.nan, np.nan])


# Group by 'track_id' and calculate duration, most frequent class and corresponding name for each group
timestamp_grouped = df_filtered.groupby("track_id")["timestamp"].agg(["min", "max"])
class_name_grouped = df_filtered.groupby("track_id").apply(most_frequent_class_and_name)

# Reset index for both groupby results
timestamp_grouped.reset_index(inplace=True)
class_name_grouped.reset_index(inplace=True)

# Rename columns for class_name_grouped
class_name_grouped.columns = ["track_id", "class", "name"]

# Merge the two groupby results
grouped = pd.merge(timestamp_grouped, class_name_grouped, on="track_id")

# Calculate duration
grouped["duration"] = grouped["max"] - grouped["min"]

# Final DataFrame with 'track_id', 'min_timestamp', 'max_timestamp', 'duration', 'class', and 'name'
final_df = grouped.rename(columns={"min": "min_timestamp", "max": "max_timestamp"})

with open("pu6kHxiCSY0_2023-10-3115.txt", "w") as f:
    for _, row in final_df.iterrows():
        f.write(
            f"{row['name']} with id {row['track_id']} was present in the video for {row['duration']} from {row['min_timestamp']} to {row['max_timestamp']}\n"
        )
