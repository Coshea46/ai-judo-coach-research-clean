import os
import subprocess

# Directories
source_dir = "C:/Users/oshea/OneDrive/Desktop/Judo-coach-bot/clips/full-video-downloads"
target_dir = "C:/Users/oshea/OneDrive/Desktop/Judo-coach-bot/clips/5-sec-partitions"

os.makedirs(target_dir, exist_ok=True)

# Loop through all mp4 files
for file_name in sorted(os.listdir(source_dir)):
    if file_name.lower().endswith(".mp4"):
        base_name = os.path.splitext(file_name)[0]
        input_path = os.path.join(source_dir, file_name)
        
        # FFmpeg command to split into 5-second clips
        # %03d gives zero-padded numbering: 001, 002, ...
        output_pattern = os.path.join(target_dir, f"{base_name}_part%03d.mp4")
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-c", "copy",
            "-map", "0",
            "-segment_time", "5",
            "-f", "segment",
            "-reset_timestamps", "1",
            output_pattern
        ]
        
        subprocess.run(cmd)
        print(f"Finished splitting '{file_name}' into 5-second clips.")
