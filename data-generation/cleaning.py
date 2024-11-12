# Import libraries
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tqdm
import cv2

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

# Initialize states
init_states = model.init_states(clapping[tf.newaxis, ...].shape)
states = init_states

# Iterate over each frame and print top 5 predictions
for n in tqdm.tqdm(range(len(clapping))):
    inputs = states
    inputs['image'] = clapping[tf.newaxis, n:n+1, ...]
    logits, states = model(inputs)
    probs = tf.nn.softmax(logits[0], axis=-1)
    print(f'Frame {n+1}:')
    for label, p in get_top_k(probs):
        print(f'{label:20s}: {p:.3f}')
    print()