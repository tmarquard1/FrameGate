# Import libraries
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tqdm
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Create the generated-data directory if it doesn't exist
os.makedirs('generated-data', exist_ok=True)

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

# Define ResNet50 model
input_tensor = Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
x = base_model.output
x = GlobalAveragePooling2D()(x)
output_tensor = Dense(1, activation='sigmoid')(x)  # Single output with sigmoid activation
resnet_model = Model(inputs=base_model.input, outputs=output_tensor)

# Compile the model
frame_gate.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Training loop
epochs = 10
batch_size = 32

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    for n in tqdm.tqdm(range(0, len(clapping), batch_size)):
        batch_frames = clapping[n:n+batch_size]
        batch_labels = np.zeros((batch_size,))  # Placeholder for actual labels

        # Train ResNet50
        with tf.GradientTape() as tape:
            frame_gate_output = frame_gate(batch_frames)
            loss = frame_gate.compiled_loss(batch_labels, frame_gate_output)
        gradients = tape.gradient(loss, frame_gate.trainable_variables)
        frame_gate.optimizer.apply_gradients(zip(gradients, frame_gate.trainable_variables))

        print(f'Batch {n//batch_size+1} loss: {loss.numpy()}')

# Save the trained ResNet50 model
frame_gate.save('resnet50_retrained.h5')

# Iterate over each frame and display top k results if frame_gate output is above 0.5
k = 5  # Number of top labels to consider

for n in tqdm.tqdm(range(len(clapping))):
    frame = clapping[n:n+1, ...]

    # Get frame_gate output
    frame_gate_output = frame_gate(frame)
    if frame_gate_output.numpy()[0][0] > 0.5:
        # Initialize states
        init_states = model.init_states(frame[tf.newaxis, ...].shape)
        states = init_states

        # Process the nth frame
        inputs = states
        inputs['image'] = frame
        logits, states = model(inputs)
        probs = tf.nn.softmax(logits[0], axis=-1)

        # Get top k labels for the nth frame
        top_k_labels_probs = get_top_k(probs, k=k)
        top_k_labels = [label for label, _ in top_k_labels_probs]

        # Print the top k labels and their probabilities
        print(f'Frame {n+1} top {k} predictions:')
        for label, prob in top_k_labels_probs:
            print(f'{label:20s}: {prob:.3f}')
        print()
    else:
        print(f'Frame {n+1} skipped by frame_gate')