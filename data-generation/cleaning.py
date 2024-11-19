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

# Load labels
labels_path = tf.keras.utils.get_file(
    fname='labels.txt',
    origin='https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
)
labels_path = pathlib.Path(labels_path)
lines = labels_path.read_text().splitlines()
KINETICS_600_LABELS = np.array([line.strip() for line in lines])
data_directory = "../HumanActivityRecognition-VideoDataset"

# Download and save the model locally
model_id = 'a2'
model_mode = 'stream'
model_version = '3'
hub_url = f'https://tfhub.dev/tensorflow/movinet/{model_id}/{model_mode}/kinetics-600/classification/{model_version}'
local_model_path = pathlib.Path('movinet_model')
if not local_model_path.exists():
    local_model_path.mkdir(parents=True, exist_ok=True)
    model = hub.load(hub_url)
    tf.saved_model.save(model, str(local_model_path))
else:
    model = tf.saved_model.load(str(local_model_path))

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
    video = video / 255.0
    return video

# Function to get top k labels and probabilities
def get_top_k(probs, k=5, label_map=KINETICS_600_LABELS):
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
    top_labels = tf.gather(label_map, top_predictions, axis=-1)
    top_labels = [label.decode('utf8') for label in top_labels.numpy()]
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    return tuple(zip(top_labels, top_probs))

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


def addVideoFramesToCsv(directory, file):
    full_path = os.path.join(directory, file)
    # Load video
    video = load_mp4(full_path)

    # Load model
    # Prepare CSV file
    csv_file = open('impact.csv', mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Video', 'Frame n', 'Frame m', 'Perceived Change', 
                         'Top 1 Label (n)', 'Top 1 Probability (n)', 'Top 2 Label (n)', 'Top 2 Probability (n)', 'Top 3 Label (n)', 'Top 3 Probability (n)', 'Top 4 Label (n)', 'Top 4 Probability (n)', 'Top 5 Label (n)', 'Top 5 Probability (n)',
                         'Top 1 Label (m)', 'Top 1 Probability (m)', 'Top 2 Label (m)', 'Top 2 Probability (m)', 'Top 3 Label (m)', 'Top 3 Probability (m)', 'Top 4 Label (m)', 'Top 4 Probability (m)', 'Top 5 Label (m)', 'Top 5 Probability (m)'])

    # Iterate over each frame and evaluate impact
    k = 5  # Number of top labels to consider

    impact = []

    n=0
    for m in tqdm.tqdm(range(n+1, len(video))):
        # Initialize states
        init_state = model.init_states(video[tf.newaxis, ...].shape)

        # Process the nth frame
        inputs = init_state.copy()
        inputs['image'] = video[tf.newaxis, n:n+1, ...]
        logits, state = model(inputs)
        probs_nth = tf.nn.softmax(logits[0], axis=-1)
        top_k_labels_probs_nth = get_top_k(probs_nth, k=k)
        top_k_labels_nth = [label for label, _ in top_k_labels_probs_nth]
        top_k_probs_nth = [prob for _, prob in top_k_labels_probs_nth]

        # Process the mth frame
        inputs = state
        inputs['image'] = video[tf.newaxis, m:m+1, ...]
        logits, state = model(inputs)
        probs_mth = tf.nn.softmax(logits[0], axis=-1)
        top_k_labels_probs_mth = get_top_k(probs_mth, k=k)
        top_k_labels_mth = [label for label, _ in top_k_labels_probs_mth]
        top_k_probs_mth = [prob for _, prob in top_k_labels_probs_mth]

        # Calculate the difference in probabilities for the top k labels
        diff_probs = probs_mth - probs_nth
        top_k_diff_probs = {label: diff_probs[KINETICS_600_LABELS == label][0].numpy() for label in top_k_labels_mth}
        
        # Print the probability changes for the top k labels
        print(f'Frame {n} -> {m} impact:')
        for label in top_k_labels_mth:
            print(f'{label:20s}: {top_k_diff_probs[label]:.3f}')
        print()

        # Calculate the perceived change in state by summing the absolute values of the probability changes
        perceived_change = tf.reduce_sum(tf.abs(diff_probs)).numpy()
        print(f'Perceived change in state: {perceived_change:.3f}\n')

        impact.append(perceived_change)

        # Write the perceived change, normalized perceived change, and top k labels and probabilities to the CSV file
        csv_writer.writerow([full_path, f'{n}', f'{m}', perceived_change] + 
                            [item for pair in zip(top_k_labels_nth, top_k_probs_nth) for item in pair] + 
                            [item for pair in zip(top_k_labels_mth, top_k_probs_mth) for item in pair])
    csv_file.close()


files = getAllFilesInDirectory(data_directory)
for file in files:
    addVideoFramesToCsv(data_directory, file)

# # Create a graph of the impact and write it to a file
# plt.plot(impact)
# plt.xlabel('Frame')
# plt.ylabel('Impact')
# plt.title('Impact of each frame')
# plt.savefig('impact_plot.png')
# plt.show()

        # How could I use the logits to get the perceived change in state?