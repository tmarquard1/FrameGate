import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import csv
import tqdm

# Read the CSV file
csv_file_path = '../model-training/impact.csv'
data = pd.read_csv(csv_file_path)

def load_frame(video_path, frame_index, image_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Frame {frame_index} not found in video {video_path}")
    frame = cv2.resize(frame, image_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def write_frame_data_to_csv(data, output_csv_path, image_size=(224, 224)):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        header = ['Video', 'Frame n', 'Frame m', 'Perceived Change'] + [f'Pixel_{i}' for i in range(image_size[0] * image_size[1] * 3 * 2)]
        writer.writerow(header)
        
        for index, row in tqdm.tqdm(data.iterrows(), total=data.shape[0]):
            video_path = row['Video']
            frame_n_index = int(row['Frame n'])
            frame_m_index = int(row['Frame m'])
            perceived_change = row['Perceived Change']

            frame_n = load_frame(video_path, frame_n_index, image_size)
            frame_m = load_frame(video_path, frame_m_index, image_size)

            frame_n = frame_n / 255.0
            frame_m = frame_m / 255.0

            # Flatten the frames and concatenate them
            frame_data = np.concatenate([frame_n.flatten(), frame_m.flatten()])

            # Write the row to the CSV file
            writer.writerow([video_path, frame_n_index, frame_m_index, perceived_change] + frame_data.tolist())

# Write the frame data to a CSV file
output_csv_path = 'frame_data.csv'
write_frame_data_to_csv(data, output_csv_path)