# Import libraries
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os
import matplotlib.pyplot as plt

# Load labels
labels_path = tf.keras.utils.get_file(
    fname='labels.txt',
    origin='https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
)
labels_path = pathlib.Path(labels_path)
lines = labels_path.read_text().splitlines()
KINETICS_600_LABELS = np.array([line.strip() for line in lines])

# Function to get top k labels and probabilities
def get_top_k(probs, k=5, label_map=KINETICS_600_LABELS):
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
    top_labels = tf.gather(label_map, top_predictions, axis=-1)
    top_labels = [label.decode('utf8') for label in top_labels.numpy()]
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    return tuple(zip(top_labels, top_probs))

# Load and preprocess the image
def load_image(image_path, image_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = image / 255.0
    return image

# Load the image
image_path = "EmptyRoom.jpg"
image = load_image(image_path)

# Create a batch of 10 identical frames
video = tf.stack([image] * 10, axis=0)

# Expand dimensions to match the model input
video = tf.expand_dims(video, axis=0)

# Load MoViNet model
id = 'a2'
mode = 'stream'
version = '3'
hub_url = f'https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'
model = hub.load(hub_url)

# Initialize states with the shape of the video
init_states = model.init_states(video.shape)

# Run the video through the model and display top k labels
k = 5  # Number of top labels to consider

for i in range(10):
    # Process each frame individually
    frame = video[:, i:i+1, ...]
    inputs = init_states
    inputs['image'] = frame
    logits, states = model(inputs)
    probs = tf.nn.softmax(logits[0], axis=-1)

    # Get top k labels for the frame
    top_k_labels_probs = get_top_k(probs, k=k)
    top_k_labels = [label for label, _ in top_k_labels_probs]

    # Print the top k labels and their probabilities
    print(f'Frame {i+1} top {k} predictions:')
    for label, prob in top_k_labels_probs:
        print(f'{label:20s}: {prob:.3f}')
    print()