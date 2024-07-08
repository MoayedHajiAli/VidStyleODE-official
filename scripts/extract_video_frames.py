import cv2
import os
import argparse
from tqdm import tqdm

def extract_frames_from_videos(root_dir, target_root_dir):
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)

    for subdir, _, files in tqdm(os.walk(root_dir), desc="Extracting frames"):
        relative_subdir = os.path.relpath(subdir, root_dir)
        video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for idx, video_file in enumerate(video_files):
            video_path = os.path.join(subdir, video_file)
            video_capture = cv2.VideoCapture(video_path)

            if not video_capture.isOpened():
                print(f"Error opening video file {video_path}")
                continue

            output_dir = os.path.join(target_root_dir, f"{relative_subdir}_{idx}")
            os.makedirs(output_dir, exist_ok=True)

            frame_idx = 0
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                frame_filename = os.path.join(output_dir, f"{frame_idx:03d}.png")
                cv2.imwrite(frame_filename, frame)
                frame_idx += 1

            video_capture.release()
            print(f"Extracted frames from {video_path} to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos in a directory of directories.")
    parser.add_argument("-s", "--source_directory", type=str, help="Path to the root directory containing subdirectories of videos.")
    parser.add_argument("-t", "--target_directory", type=str, help="Path to the target directory where frames will be saved.")

    args = parser.parse_args()
    extract_frames_from_videos(args.source_directory, args.target_directory)
