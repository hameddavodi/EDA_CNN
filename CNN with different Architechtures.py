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
desired_width = 224
desired_height = 224

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





# Input tensor
input_tensor = Input(shape=(desired_width, desired_height, 3))

# Initial convolution layer
x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_tensor)

# PolyNet inception blocks
def polynet_inception_block(x, num_filters):
    branch1x1 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(x)
    
    branch3x3 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(x)
    branch3x3 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(branch3x3)
    
    branch3x3stack = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(x)
    branch3x3stack = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(branch3x3stack)
    branch3x3stack = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(branch3x3stack)
    
    # Concatenate the outputs of the branches
    x = concatenate([branch1x1, branch3x3, branch3x3stack], axis=-1)
    return x

# Apply PolyNet inception blocks
for _ in range(3):  # You can adjust the number of blocks
    x = polynet_inception_block(x, 64)

# Global Average Pooling Layer
x = GlobalAveragePooling2D()(x)

# Fully connected layers
x = Dense(256, activation='relu')(x)

# Output layer for classification (adjust the number of classes as needed)
output = Dense(num_classes, activation='softmax')(x)

# Create the PolyNet-like model
PolyNet = Model(inputs=input_tensor, outputs=output)

# Compile the model
PolyNet.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
PolyNet.summary()
# Plot the model's structure and save it as an image
plot_model(PolyNet, to_file='PolyNet.png', show_shapes=True, show_layer_names=True)





import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def cv_5(model, X, y, epochs, n_splits=5):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize lists to store training and validation loss for each epoch
        train_loss = []
        val_loss = []

        history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, verbose=1)

        # Append training and validation loss for this epoch
        train_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Calculate zero-one loss
        zero_one_loss = 1 - accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        # Calculate other metrics using y_pred and y_test
        precision = precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')
        recall = recall_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')
        f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='weighted')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(history.history['accuracy'])
        val_accuracies.append(history.history['val_accuracy'])
        # Evaluate the model on the test set for this fold
        _, test_accuracy = model.evaluate(X_test, y_test)

        # Calculate the zero-one loss (misclassification error) for this fold
        zero_one_loss = 1.0 - test_accuracy

        accuracies.append(1 - zero_one_loss)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Calculate average metrics over all folds
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    # Calculate the mean and standard deviation of loss and accuracy across folds
    mean_train_loss = np.mean(train_losses, axis=0)
    std_train_loss = np.std(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    std_val_loss = np.std(val_losses, axis=0)

    mean_train_accuracy = np.mean(train_accuracies, axis=0)
    std_train_accuracy = np.std(train_accuracies, axis=0)
    mean_val_accuracy = np.mean(val_accuracies, axis=0)
    std_val_accuracy = np.std(val_accuracies, axis=0)

    # Define a function for moving average
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    # Smoothing window size (adjust as needed)
    window_size = 5

    # Plotting the training and validation loss
    plt.figure(figsize=(12, 5))

    # Subplot for training and validation loss
    plt.subplot(1, 2, 1)

    # Smoothing the training and validation loss curves
    smoothed_train_loss = moving_average(mean_train_loss, window_size)
    smoothed_val_loss = moving_average(mean_val_loss, window_size)

    # Create epochs_range based on the length of smoothed_val_loss
    epochs_range = range(1, len(smoothed_val_loss) + 1)

    # Plotting smoothed validation loss and its confidence interval
    plt.plot(epochs_range, smoothed_val_loss, label='Validation Loss', color='red')
    plt.fill_between(epochs_range, smoothed_val_loss - std_val_loss, smoothed_val_loss + std_val_loss, color='red', alpha=0.2)

    # Plotting smoothed training loss and its confidence interval
    plt.plot(epochs_range, smoothed_train_loss, label='Training Loss', color='blue')
    plt.fill_between(epochs_range, smoothed_train_loss - std_train_loss, smoothed_train_loss + std_train_loss, color='blue', alpha=0.2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    return {
        "Average Accuracy": avg_accuracy,
        "Average Precision": avg_precision,
        "Average Recall": avg_recall,
        "Average F1 Score": avg_f1,
    }





