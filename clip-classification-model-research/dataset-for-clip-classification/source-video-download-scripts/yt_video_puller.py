import sys
import csv

from pytubefix import YouTube

# accepts a txt of yt urls and pulls each into the target output directory

def pull_yt_video(video_url: str, target_output_dir_path: str, video_id: str) -> None:
    """
    Downloads a single Youtube video as an mp4 using its url
    and stores it in the target output directory.

    Should assume all parameters are valid and exist
    Video should be named using the video_id parameter
    with the .mp4 extension added.
    """

    try:
            
        yt = YouTube(video_url)

        # Get the best MP4 video (720p/1080p compatible)
        stream = yt.streams.get_highest_resolution()
                
        # Download using the Video ID as the filename (matching your old script)
        stream.download(output_path=target_output_dir_path, filename=f"{video_id}.mp4")
        print(" -> Success!")
                
    except Exception as e:
        print(f" -> FAILED: {video_url} | Error: {e}")



def read_urls_txt(urls_txt_path: str) -> list[str]:
    """
    Reads in a txt that contains exactly one valid URL per
    line and returns each URL as an element in a list
    """

    with open(urls_txt_path, mode='r') as f:
        urls = [url.strip() for url in f]

    return urls




def main(args):
    """
    Loops through all urls, applying function to extract each
    """

    urls_txt_path = args[0]
    target_output_dir_path = args[1]

    video_urls = read_urls_txt(urls_txt_path=urls_txt_path)

    VIDEO_INDEX_CSV_PATH = 'video_source_index.csv'
    VIDEO_INDEX_CSV_HEADER = ['video_id', 'url']

    with open(VIDEO_INDEX_CSV_PATH, mode='w', newline='', encoding='utf-8') as output_csv_f:

        writer = csv.writer(output_csv_f)
        writer.writerow(VIDEO_INDEX_CSV_HEADER)

        for i, url in enumerate(video_urls):

            video_id = f"source_video{i:04d}"
            pull_yt_video(url, target_output_dir_path, video_id)

            url_video_id_mapping_for_csv = [video_id, url]
            writer.writerow(url_video_id_mapping_for_csv)


if __name__ == "__main__":
    main(sys.argv[1:])