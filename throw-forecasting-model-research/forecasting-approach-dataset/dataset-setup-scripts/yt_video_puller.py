import csv
import os
from pytubefix import YouTube

# Paths
csv_file = "extra_match.csv"
output_dir = "C:/Users/oshea/OneDrive/Desktop/Judo-coach-bot/temp_video_store"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        url = row[0]
        try:
            print(f"Downloading {url}...")
            yt = YouTube(url)
            
            # Get the best MP4 video (720p/1080p compatible)
            stream = yt.streams.get_highest_resolution()
            
            # Download using the Video ID as the filename (matching your old script)
            stream.download(output_path=output_dir, filename=f"{yt.video_id}.mp4")
            print(" -> Success!")
            
        except Exception as e:
            print(f" -> FAILED: {url} | Error: {e}")