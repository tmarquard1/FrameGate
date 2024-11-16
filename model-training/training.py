import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model

# Load the ResNet50 model pre-trained on ImageNet, without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False)

# Define the input layers for the two images
input_1 = Input(shape=(224, 224, 3))
input_2 = Input(shape=(224, 224, 3))

# Pass the inputs through the base model
features_1 = base_model(input_1)
features_2 = base_model(input_2)

# Global average pooling to reduce the spatial dimensions
pooled_features_1 = GlobalAveragePooling2D()(features_1)
pooled_features_2 = GlobalAveragePooling2D()(features_2)

# Concatenate the pooled features from both images
concatenated_features = Concatenate()([pooled_features_1, pooled_features_2])

# Add a dense layer with a single output for the probabilistic impact
output = Dense(1, activation='sigmoid')(concatenated_features)

# Create the model
model = Model(inputs=[input_1, input_2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()