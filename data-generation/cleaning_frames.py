# Import libraries
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tqdm
import cv2
import os
import csv
import matplotlib.pyplot as plt
import multiprocessing

data_directory = "../HumanActivityRecognition-VideoDataset"

# Function to load mp4 file
def load_mp4(file_path, image_size=(224, 224)):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    video = tf.convert_to_tensor(frames, dtype=tf.float32)
    # video = video / 255.0
    return video

def getAllFilesInDirectory(directory):
    files = []
    # List all files and subdirectories in the specified directory
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path) and entry.endswith(".mp4"):
            # If it's a file and ends with .mp4, add its relative path
            files.append(entry)
        elif os.path.isdir(full_path):
            # If it's a subdirectory, list its files
            for sub_entry in os.listdir(full_path):
                sub_full_path = os.path.join(full_path, sub_entry)
                if os.path.isfile(sub_full_path) and sub_entry.endswith(".mp4"):
                    # Add the relative path of the file
                    files.append(os.path.join(entry, sub_entry))
    return files

def writeVideoFrameToFile(directory, file, output_directory):
    full_path = os.path.join(directory, file)
    # Load video
    video = load_mp4(full_path)
    
    # Create a directory for the video file
    video_name = os.path.splitext(file)[0]
    video_output_directory = os.path.join(output_directory, video_name)
    os.makedirs(video_output_directory, exist_ok=True)
    
    # Write each frame to an image file
    for i, frame in enumerate(video):
        frame = frame.numpy().astype(np.uint8)
        frame_path = os.path.join(video_output_directory, f"frame_{i:03d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

files = getAllFilesInDirectory(data_directory)
frames_directory = "../frames"

def process_file(file):
    writeVideoFrameToFile(data_directory, file, frames_directory)

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(process_file, files), total=len(files)):
            pass