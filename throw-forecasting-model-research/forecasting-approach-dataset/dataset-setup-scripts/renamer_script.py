import os

# Path to the directory containing the files
directory = "C:/Users/oshea/OneDrive/Desktop/Judo-coach-bot/temp_video_store"

# Get a sorted list of all files in the directory
files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

# Determine padding based on total number of files
num_digits = len(str(len(files)))

# Loop through files and rename
for idx, filename in enumerate(files, start=16):
    # Split filename and extension
    name, ext = os.path.splitext(filename)
    # Create new name with zero padding
    new_name = f"{str(idx).zfill(num_digits)}{ext}"
    # Full paths
    old_path = os.path.join(directory, filename)
    new_path = os.path.join(directory, new_name)
    # Rename file
    os.rename(old_path, new_path)
    print(f"Renamed '{filename}' -> '{new_name}'")
