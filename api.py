import os
import time
import base64
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# Global model variable
model = None

def load_model():
    """Load model once at startup"""
    global model
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "captcha_model_2k_best.keras")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found!")
        print("Please train the model first using: python train_captcha_model.py")
        return None
    
    print(f"Loading CAPTCHA model from {model_path}...")
    start_time = time.time()
    model = keras.models.load_model(
        model_path,
        custom_objects={"CTCLayer": CTCLayer}
    )
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.2f} seconds")
    return model

def preprocess_image_from_bytes(image_bytes):
    """Preprocess image from bytes for prediction"""
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale if not already
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Convert to tensor and add channel dimension
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=-1)
    
    # Normalize
    img_tensor = img_tensor / 255.0
    
    # Resize
    img_tensor = tf.image.resize(img_tensor, [img_height, img_width])
    
    # Transpose to shape (width, height, channels)
    img_tensor = tf.transpose(img_tensor, perm=[1, 0, 2])
    
    return img_tensor

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

def solve_captcha(image_bytes):
    """Solve CAPTCHA from image bytes"""
    # Preprocess
    img = preprocess_image_from_bytes(image_bytes)
    img = tf.expand_dims(img, 0)  # Add batch dimension
    
    # Create dummy labels
    dummy_labels = tf.zeros((1, max_length), dtype=tf.float32)
    
    # Predict
    pred = model.predict([img, dummy_labels], verbose=0)
    
    # Decode
    result = decode_prediction(pred)
    
    return result

# Initialize FastAPI app
app = FastAPI(
    title="CAPTCHA Solver API - 2K Model",
    version="2.0.0",
    description="High-accuracy CAPTCHA solver trained on 2000 labeled images",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()
    if model:
        print("=" * 70)
        print("CAPTCHA Solver API - 2K Model is ready!")
        print("=" * 70)
        print("Access Swagger UI at: http://localhost:8001/docs")
        print("Access ReDoc at: http://localhost:8001/redoc")
        print("=" * 70)
    else:
        print("ERROR: Failed to load model. API will not work properly.")

@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint - API information
    
    Returns basic API information and available endpoints.
    """
    return {
        "name": "CAPTCHA Solver API - 2K Model",
        "version": "2.0.0",
        "description": "High-accuracy CAPTCHA solver trained on 2000 labeled images",
        "status": "running",
        "model_loaded": model is not None,
        "training_dataset": "2000 labeled CAPTCHAs",
        "endpoints": {
            "GET /": "API information (this page)",
            "GET /health": "Health check endpoint",
            "POST /solve": "Upload CAPTCHA image to solve (multipart/form-data)",
            "POST /solve-base64": "Send base64 encoded CAPTCHA image (application/json)",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)"
        },
        "example_usage": {
            "curl_file": "curl -X POST 'http://localhost:8001/solve' -F 'file=@captcha.png'",
            "curl_base64": "curl -X POST 'http://localhost:8001/solve-base64' -H 'Content-Type: application/json' -d '{\"image\":\"base64_string\"}'"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the API and model.
    """
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_type": "CNN + BiLSTM with CTC loss",
        "training_samples": 2000,
        "expected_accuracy": "95%+"
    }

@app.post("/solve", tags=["CAPTCHA Solver"])
async def solve_captcha_file(file: UploadFile = File(..., description="CAPTCHA image file (PNG, JPEG, JPG)")):
    """
    Solve CAPTCHA from uploaded image file
    
    **Parameters:**
    - **file**: CAPTCHA image file to solve
    
    **Returns:**
    - **success**: Boolean indicating if solving was successful
    - **captcha**: Predicted CAPTCHA text (6 digits)
    - **solve_time_ms**: Time taken to solve in milliseconds
    - **filename**: Original filename
    
    **Example using curl:**
    ```bash
    curl -X POST "http://localhost:8001/solve" -F "file=@captcha.png"
    ```
    
    **Example using Python requests:**
    ```python
    import requests
    with open('captcha.png', 'rb') as f:
        response = requests.post('http://localhost:8001/solve', files={'file': f})
        print(response.json())
    ```
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    # Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (PNG, JPEG, JPG)")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Solve CAPTCHA
        start_time = time.time()
        result = solve_captcha(image_bytes)
        solve_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return JSONResponse(content={
            "success": True,
            "captcha": result,
            "solve_time_ms": round(solve_time, 2),
            "filename": file.filename,
            "model": "2K trained model"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/solve-base64", tags=["CAPTCHA Solver"])
async def solve_captcha_base64(data: dict):
    """
    Solve CAPTCHA from base64 encoded image
    
    **Request body:**
    ```json
    {
        "image": "base64_encoded_image_string"
    }
    ```
    
    **Returns:**
    - **success**: Boolean indicating if solving was successful
    - **captcha**: Predicted CAPTCHA text (6 digits)
    - **solve_time_ms**: Time taken to solve in milliseconds
    
    **Example using curl:**
    ```bash
    curl -X POST "http://localhost:8001/solve-base64" \\
         -H "Content-Type: application/json" \\
         -d '{"image": "base64_string_here"}'
    ```
    
    **Example using Python requests:**
    ```python
    import requests
    import base64
    
    with open('captcha.png', 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    response = requests.post('http://localhost:8001/solve-base64',
                            json={'image': img_b64})
    print(response.json())
    ```
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    if "image" not in data:
        raise HTTPException(status_code=400, detail="Missing 'image' field in request body")
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(data["image"])
        
        # Solve CAPTCHA
        start_time = time.time()
        result = solve_captcha(image_bytes)
        solve_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return JSONResponse(content={
            "success": True,
            "captcha": result,
            "solve_time_ms": round(solve_time, 2),
            "model": "2K trained model"
        })
    
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("Starting CAPTCHA Solver API - 2K Model")
    print("=" * 70)
    print("Model: captcha_model_2k_best.keras")
    print("Training: 2000 labeled images")
    print("Port: 8001")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8001)