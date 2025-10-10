import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add parent directory to path to import from main folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parameters (must match training)
img_width = 182
img_height = 50
max_length = 6

# Character set
characters = '0123456789'
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None, num_oov_indices=0)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True, num_oov_indices=0
)

class CTCLayer(layers.Layer):
    """CTC loss layer (needed for loading model)"""
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

def preprocess_image(img_path):
    """Preprocess a single image for prediction"""
    # Read image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    
    # Convert to float32 and normalize
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # Resize
    img = tf.image.resize(img, [img_height, img_width])
    
    # Transpose
    img = tf.transpose(img, perm=[1, 0, 2])
    
    return img

def decode_prediction(pred):
    """Decode CTC prediction"""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    # Use greedy search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    
    # Convert to text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    
    return output_text[0]

def predict_captcha(model, img_path):
    """Predict CAPTCHA for a single image"""
    # Preprocess
    img = preprocess_image(img_path)
    img = tf.expand_dims(img, 0)  # Add batch dimension
    
    # Create dummy labels for prediction (not used but required by model)
    dummy_labels = tf.zeros((1, max_length), dtype=tf.float32)
    
    # Predict using full model
    pred = model.predict([img, dummy_labels], verbose=0)
    
    # Decode
    result = decode_prediction(pred)
    
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python use_2k_model.py <path_to_image>")
        print("Example: python use_2k_model.py ../cap1.jpeg")
        sys.exit(1)
    
    img_path = sys.argv[1]
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Model path (in the same directory as this script)
    model_path = os.path.join(script_dir, "captcha_model_2k_best.keras")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using: python train_captcha_model.py")
        sys.exit(1)
    
    print("Loading 2K trained model...")
    model = keras.models.load_model(
        model_path,
        custom_objects={"CTCLayer": CTCLayer}
    )
    
    print(f"Predicting CAPTCHA for: {img_path}")
    
    try:
        prediction = predict_captcha(model, img_path)
        print(f"\nPrediction: {prediction}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()