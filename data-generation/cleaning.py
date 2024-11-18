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

# Load video
clapping = load_mp4("Clapping.mp4")

# Load model
id = 'a2'
mode = 'stream'
version = '3'
hub_url = f'https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'
model = hub.load(hub_url)

# Prepare CSV file
csv_file = open('impact.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Video', 'Frame n', 'Frame m', 'Perceived Change', 'Top 1 Label', 'Top 1 Probability', 'Top 2 Label', 'Top 2 Probability', 'Top 3 Label', 'Top 3 Probability', 'Top 4 Label', 'Top 4 Probability', 'Top 5 Label', 'Top 5 Probability'])

# Iterate over each frame and evaluate impact
k = 5  # Number of top labels to consider

impact = []

for n in tqdm.tqdm(range(0, len(clapping)-1)):
    for m in tqdm.tqdm(range(n+1, len(clapping))):
        # Initialize states
        init_states = model.init_states(clapping[tf.newaxis, ...].shape)
        states = init_states

        # Process the nth frame
        inputs = states
        inputs['image'] = clapping[tf.newaxis, n:n+1, ...]
        logits, states = model(inputs)
        probs_nth = tf.nn.softmax(logits[0], axis=-1)

        # Reset states
        states = init_states

        # Process the mth frame
        inputs = states
        inputs['image'] = clapping[tf.newaxis, m:m+1, ...]
        logits, states = model(inputs)
        probs_mth = tf.nn.softmax(logits[0], axis=-1)

        # Get top k labels for the mth frame
        top_k_labels_probs = get_top_k(probs_mth, k=k)
        top_k_labels = [label for label, _ in top_k_labels_probs]
        top_k_probs = [prob for _, prob in top_k_labels_probs]

        # Calculate the difference in probabilities for the top k labels
        diff_probs = probs_mth - probs_nth
        top_k_diff_probs = {label: diff_probs[KINETICS_600_LABELS == label][0].numpy() for label in top_k_labels}
        
        # Print the probability changes for the top k labels
        print(f'Frame {n} -> {m} impact:')
        for label in top_k_labels:
            print(f'{label:20s}: {top_k_diff_probs[label]:.3f}')
        print()

        # Calculate the perceived change in state by summing the absolute values of the probability changes
        perceived_change = tf.reduce_sum(tf.abs(diff_probs)).numpy()
        print(f'Perceived change in state: {perceived_change:.3f}\n')

        impact.append(perceived_change)

        # Write the perceived change, normalized perceived change, and top k labels and probabilities to the CSV file
        csv_writer.writerow(['Clapping.mp4', f'{n}',f'{m}', perceived_change] + [item for pair in zip(top_k_labels, top_k_probs) for item in pair])

        # Clear the session
    tf.keras.backend.clear_session()

# Close the CSV file
csv_file.close()

# Create a graph of the impact and write it to a file
plt.plot(impact)
plt.xlabel('Frame')
plt.ylabel('Impact')
plt.title('Impact of each frame')
plt.savefig('impact_plot.png')
plt.show()

        # How could I use the logits to get the perceived change in state?