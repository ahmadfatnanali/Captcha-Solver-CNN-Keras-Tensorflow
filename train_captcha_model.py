import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from pathlib import Path

# Parameters
img_width = 182
img_height = 50
max_length = 6
batch_size = 16
epochs = 100

# Character set (digits only)
characters = '0123456789'
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None, num_oov_indices=0)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True, num_oov_indices=0
)

def encode_single_sample(img_path, label):
    """Preprocess and encode a single CAPTCHA sample"""
    # Read image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    
    # Convert to float32 and normalize
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    
    # Transpose to shape (width, height, channels)
    img = tf.transpose(img, perm=[1, 0, 2])
    
    # Convert label to numeric
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    
    return {"image": img, "label": label}

def prepare_dataset(data_dir):
    """Prepare dataset from directory"""
    data_dir = Path(data_dir)
    
    # Get all image paths and labels
    images = []
    labels = []
    
    for img_file in sorted(data_dir.glob("*.png")):
        label = img_file.stem  # Filename without extension
        if len(label) == 6 and label.isdigit():  # Only valid 6-digit labels
            images.append(str(img_file))
            labels.append(label)
    
    print(f"Found {len(images)} valid CAPTCHA images")
    
    # Split into train and validation
    split_idx = int(0.9 * len(images))
    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    val_images = images[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    return (train_images, train_labels), (val_images, val_labels)

def create_dataset(images, labels, batch_size):
    """Create TensorFlow dataset"""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

class CTCLayer(layers.Layer):
    """CTC loss layer"""
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred
    
    def get_config(self):
        return super().get_config()

def build_model():
    """Build the CAPTCHA recognition model"""
    # Input layers
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Reshape for RNN layers
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(len(characters) + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer - pass as positional arguments
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define model
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="captcha_ocr")
    
    # Compile
    model.compile(optimizer=keras.optimizers.Adam())
    
    return model

def train_model(data_dir):
    """Main training function"""
    print("="*70)
    print("CAPTCHA OCR Model Training")
    print("="*70)
    
    # Prepare datasets
    (train_images, train_labels), (val_images, val_labels) = prepare_dataset(data_dir)
    
    train_dataset = create_dataset(train_images, train_labels, batch_size)
    val_dataset = create_dataset(val_images, val_labels, batch_size)
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    
    # Callbacks - Increased patience to allow more training
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True, verbose=1
    )
    
    # Save model in the CaptchaSolverCNN directory
    model_save_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(model_save_dir, "captcha_model_2k_best.keras")
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
    )
    
    # Add progress callback
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\nEpoch {epoch+1}/{epochs} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}")
    
    progress = ProgressCallback()
    
    # Train
    print("\nStarting training...")
    print(f"Will train for up to {epochs} epochs (early stopping patience: 30)")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint, progress],
        verbose=2
    )
    
    # Save final model in the CaptchaSolverCNN directory
    model_save_dir = os.path.dirname(os.path.abspath(__file__))
    final_path = os.path.join(model_save_dir, "captcha_model_2k_final.keras")
    model.save(final_path)
    print(f"\nModel saved as '{checkpoint_path}' and '{final_path}'")
    
    return model, history

if __name__ == "__main__":
    # Train the model on the solved dataset (2000 images with labels)
    data_dir = "D:\\UpWork\\CaptchaSolerUpwork\\CaptchaSolverCNN\\Captcha_Dataset\\solved"
    model, history = train_model(data_dir)
    
    print("\nTraining complete!")
    print("Use 'predict_captcha.py' to make predictions on new images")