from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, supports_credentials=True)

# Constants (matching diseases.py)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load the trained model
try:
    model = tf.keras.models.load_model('../best_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Define class names (make sure these match your training classes)
CLASS_NAMES = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

def preprocess_image(image_bytes):
    """Preprocess the image bytes for model prediction"""
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize image
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to array and preprocess
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    return img_array

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict plant disease from image matrix"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'matrix' not in data:
            return jsonify({'error': 'No image matrix provided in request'}), 400
        
        # Convert input matrix to numpy array
        try:
            image_matrix = np.array(data['matrix'])
            
            # Validate matrix dimensions
            if image_matrix.shape != (IMG_HEIGHT, IMG_WIDTH, 3):
                return jsonify({
                    'error': f'Invalid matrix dimensions. Expected shape: ({IMG_HEIGHT}, {IMG_WIDTH}, 3), '
                    f'got: {image_matrix.shape}'
                }), 400
            
            # Add batch dimension and normalize if not already normalized
            if image_matrix.max() > 1.0:
                image_matrix = image_matrix / 255.0
            
            processed_image = np.expand_dims(image_matrix, 0)
            
            # Make prediction
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            
            # Prepare response
            response = {
                'disease': CLASS_NAMES[predicted_class_index],
                'confidence': confidence,
                'predictions': {
                    CLASS_NAMES[i]: float(predictions[0][i]) 
                    for i in range(len(CLASS_NAMES))
                }
            }
            
            return jsonify(response), 200
            
        except ValueError as ve:
            return jsonify({'error': f'Invalid matrix format: {str(ve)}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
