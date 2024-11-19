import requests
import numpy as np
from PIL import Image
import json

def load_and_process_image(image_path):
    """Load and preprocess the image"""
    # Load image
    img = Image.open(image_path)
    
    # Resize image to match model requirements
    img = img.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    return img_array

def predict_disease(image_path, api_url='http://localhost:5000/predict'):
    """Send image matrix to API and get prediction"""
    try:
        # Load and process image
        img_matrix = load_and_process_image(image_path)
        
        # Prepare the request
        headers = {'Content-Type': 'application/json'}
        payload = {
            'matrix': img_matrix.tolist()  # Convert numpy array to list for JSON serialization
        }
        
        # Make API request
        response = requests.post(api_url, json=payload, headers=headers)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Results:")
            print(f"Disease: {result['disease']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("\nAll Predictions:")
            for disease, prob in result['predictions'].items():
                print(f"{disease}: {prob:.2%}")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            return None
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    image_path = "image.jpg"  # Replace with your image path
    result = predict_disease(image_path)
