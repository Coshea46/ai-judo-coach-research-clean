import csv 
import os

clips_dir = "C:/Users/oshea/OneDrive/Desktop/Judo-coach-bot/5-sec-clips"
rel_clips_dir_path = "5-sec-clips/"

clips = os.listdir(clips_dir)

col_names = ["clip_id", "clip_path", "throw", "player"]

with open("labeled_dataset.csv","a",encoding="utf-8",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(col_names)

    for i, clip in enumerate(clips,start=1):
        writer.writerow([i, rel_clips_dir_path + clip, "", ""])
