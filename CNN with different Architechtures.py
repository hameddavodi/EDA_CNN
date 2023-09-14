import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    concatenate,
    Dropout,
    Reshape,
    Multiply,
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt







# Define the path to your dataset
dataset_path = '/kaggle/input/muffin-vs-chihuahua-image-classification/'

# Define a function to load and preprocess images
def load_and_preprocess_images(folder_path):
    images = []
    labels = []

    # Loop through the class folders
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)

        # Loop through the images in each class folder
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # Read the image in RGB format
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Append the preprocessed image to the list
            images.append(image)

            # Append the corresponding label (class name)
            labels.append(class_name)

    # Ensure all images have the same shape
    images = [cv2.resize(image, (desired_width, desired_height)) for image in images]

    return np.array(images), np.array(labels)

# Define the desired width and height for the images
desired_width = 120
desired_height = 120

# Define paths for the train and test sets
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

# Load and preprocess the train and test images
train_images, train_labels = load_and_preprocess_images(train_path)
test_images, test_labels = load_and_preprocess_images(test_path)

class_to_int = {'muffin': 0, 'chihuahua': 1}  # Add more classes as needed
num_classes=2
# Use the mapping to convert class labels to integers
train_labels_int = [class_to_int[label] for label in train_labels]
test_labels_int = [class_to_int[label] for label in test_labels]

# One-hot encode the integer labels
train_labels_one_hot = to_categorical(train_labels_int, num_classes)
test_labels_one_hot = to_categorical(test_labels_int, num_classes)








# The AlexNet model
AlexNet = Sequential([
    # Layer 1
    Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(desired_width, desired_height, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    
    # Layer 2
    Conv2D(256, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    
    # Layer 3
    Conv2D(384, (3, 3), activation='relu'),
    
    # Layer 4
    Conv2D(384, (3, 3), activation='relu'),
    
    # Layer 5
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    
    # Flatten the output
    Flatten(),
    
    # Fully connected layers
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    
    # Output layer
    Dense(num_classes, activation='softmax')
])

# Compile the model
AlexNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Print model summary
AlexNet.summary()
# Visualize the architecture and save it to a file
plot_model(AlexNet, to_file='alexnet_architecture.png', show_shapes=True, show_layer_names=True)




# The EfficientNetB0 model
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(desired_width, desired_height, 3))

# Add custom layers on top of the base model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(2, activation='softmax')(x)

Efficient_NetB0 = Model(inputs=base_model.input, outputs=output)

# Compile the model
Efficient_NetB0.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
Efficient_NetB0.summary()
plot_model(Efficient_NetB0, to_file='Efficient_NetB0.png', show_shapes=True, show_layer_names=True)







# The Sequential model
Sequential_CNN = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(desired_width, desired_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
Sequential_CNN.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display a summary of the model
Sequential_CNN.summary()

# Plot the model's structure and save it as an image
plot_model(Sequential_CNN, to_file='Sequential_CNN.png', show_shapes=True, show_layer_names=True)




# Input tensor
input_tensor = Input(shape=(desired_width, desired_height, 3))

# The custom SE-Net-like block as a layer
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=16):
        super(SEBlock, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.squeeze = GlobalAveragePooling2D()
        self.excitation1 = Dense(num_channels // self.ratio, activation='relu')
        self.excitation2 = Dense(num_channels, activation='sigmoid')

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = Reshape((1, 1, -1))(x)
        x = self.excitation1(x)
        x = self.excitation2(x)
        return Multiply()([inputs, x])

SE_NET = Sequential([
    Conv2D(64, (3, 3), padding='same', input_shape=(desired_width, desired_height, 3)),
    BatchNormalization(),
    Activation('relu'),

    # Apply the custom SE-Net-like block after the first convolutional layer
    SEBlock(),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    # Apply the custom SE-Net-like block after the second convolutional layer
    SEBlock(),

    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    # Apply the custom SE-Net-like block after the third convolutional layer
    SEBlock(),

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(2, activation='softmax')
])

# Compile the model
SE_NET.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display a summary of the model
SE_NET.summary()

# Plot the model's structure and save it as an image
plot_model(SE_NET, to_file='SE_NET.png', show_shapes=True, show_layer_names=True)




# The (simplified) PolyNet CNN
def simplified_poly_net(input_shape, num_classes):
    input_layer = Input(shape=(desired_width, desired_height, 3))
    x = input_layer

    # Convolution Blocks
    for filters, repetitions in [(64, 2), (128, 4), (256, 6)]:
        for _ in range(repetitions):
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)

    # Global Average Pooling Layer
    x = tf.reduce_mean(x, axis=[1, 2])  # Global average pooling

    # Fully Connected Layers
    x = Dense(512, activation='relu')(x)

    # Output Layer
    output_layer = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
input_shape = (desired_width, desired_height, 3)  # Replace with your input shape
num_classes = 2  # Replace with the number of classes in your task

PolyNet = simplified_poly_net(input_shape, num_classes)

# Compile the model
PolyNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

PolyNet.summary()

# Plot the model's structure and save it as an image
plot_model(PolyNet, to_file='PolyNet.png', show_shapes=True, show_layer_names=True)






# Customized Function to Perform 5-fold CV with batch size of 128.
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import zero_one_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def cv_5(model, X, y, epochs, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    all_confusion_matrices = []  # To store confusion matrices for each fold

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        history = model.fit(
            tf.constant(X_train),
            y_train,  
            validation_data=(tf.constant(X_test), y_test),
            batch_size=128,
            epochs=epochs,
            verbose=1
        )

        # Predict on the test data
        y_pred = model.predict(tf.constant(X_test))

        # Calculate zero-one loss
        zero_one = 1 - zero_one_loss(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        accuracies.append(zero_one)

        # Calculate confusion matrix
        cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        all_confusion_matrices.append(cm)

    # Calculate average metrics over all folds
    avg_accuracy = np.mean(accuracies)

    # Calculate the average confusion matrix
    avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    avg_metrics = {"Average Accuracy": avg_accuracy, "Average Confusion Matrix": avg_confusion_matrix}

    # Plot the average confusion matrix
    sns.heatmap(avg_metrics["Average Confusion Matrix"], annot=True, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Average Confusion Matrix")
    plt.show()
    
    return {"Average Accuracy": avg_accuracy, "Average Confusion Matrix": avg_confusion_matrix}

# Of course there is a wide range of approaches to determine the optimum number of Epochs but I used trial and error :)
cv_5(PolyNet, train_images, train_labels_one_hot, epochs=5)
cv_5(SE_NET, train_images, train_labels_one_hot, epochs=5)
cv_5(Sequential_CNN, train_images, train_labels_one_hot, epochs=5)
cv_5(Efficient_NetB0, train_images, train_labels_one_hot, epochs=5)
cv_5(AlexNet, train_images, train_labels_one_hot, epochs=5)


