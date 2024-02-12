import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import cv2 as cv
import numpy as np
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class ImageClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classification App")

        self.data_dir = 'char_dataset'  # training dataset folder
        self.save_model = 'cnn_model'  # Specify the path to save the trained model
        self.build_gui()

    def build_gui(self):
        # Create and place widgets
        self.load_data_button = tk.Button(self.root, text="Load Data", command=self.browse_data_dir)
        self.load_data_button.pack(pady=10)

        self.data_dir_label = tk.Label(self.root, text="Selected Directory: " + self.data_dir)
        self.data_dir_label.pack()

        self.train_model_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_model_button.pack(pady=10)
        
        self.epochs_label = tk.Label(self.root, text="Epochs:")
        self.epochs_label.pack()

        self.epochs_entry = tk.Entry(self.root)
        self.epochs_entry.pack()

        self.img_height_label = tk.Label(self.root, text="img_height:")
        self.img_height_label.pack()

        self.img_height_entry = tk.Entry(self.root)
        self.img_height_entry.pack()

        self.img_width_label = tk.Label(self.root, text="img_width:")
        self.img_width_label.pack()

        self.img_width_entry = tk.Entry(self.root)
        self.img_width_entry.pack()

        self.batch_size_label = tk.Label(self.root, text="batch_size:")
        self.batch_size_label.pack()

        self.batch_size_entry = tk.Entry(self.root)
        self.batch_size_entry.pack()

    def browse_data_dir(self):
        # Allow the user to browse and select a directory
        selected_dir = filedialog.askdirectory()
        
        # Update the data_dir and label with the selected directory
        if selected_dir:
            self.data_dir = selected_dir
            self.data_dir_label.config(text="Selected Directory: " + self.data_dir)

    def load_data(self):
        # Implement your data loading logic here
        # You can use filedialog.askdirectory() to get the directory path
        # Update self.data_dir with the selected directory
        pass

    def train_model(self):
        # Implement your model training logic here
        # Use the provided code and update paths as necessary
        data_dir = self.data_dir
        save_model = self.save_model

        np.random.seed(42)
        tf.random.set_seed(42)

        batch_size = int(self.batch_size_entry.get())
        img_height = int(self.img_height_entry.get())
        img_width = int(self.img_width_entry.get())
        
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        
        class_names = train_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]

        num_classes = 36

        model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(36, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(36, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(36, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        model.summary()

        epochs = int(self.epochs_entry.get())

        history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
        )
        # ... Rest of your model training code

        # Save the trained model
        model.save(save_model)
        print(f"Model saved to {save_model}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassificationApp(root)
    root.mainloop()