# Tomato Disease Detection 🍅

FasaL is a web-based application that leverages deep learning to detect diseases in tomato plants through image analysis. Built with Python, Flask, React, and TypeScript, it provides farmers and agricultural enthusiasts with an accessible tool for early disease detection.

## 🌟 Features

- Real-time disease detection from plant images
- Support for 10 different tomato disease classifications
- User-friendly web interface with drag-and-drop functionality
- High accuracy (86% training, 81% validation)
- Detailed confidence scores for predictions

## 🔧 Tech Stack

### Backend
- Python 3.9.5
- Flask
- TensorFlow
- NumPy
- PIL
- Flask-CORS

### Frontend
- React
- TypeScript
- Tailwind CSS

### Deployment
- Backend: Render
- Frontend: Vercel

## 🚀 Web application

Visit the application: [https://crop-diseases.vercel.app/](https://crop-diseases.vercel.app/)

## 📊 Model Architecture

The system uses a Convolutional Neural Network (CNN) with:
- Input shape: 224x224x3
- 4 convolutional layers with increasing filter sizes
- MaxPooling layers for feature extraction
- Dropout (0.5) for regularization
- Dense layers for final classification
- SoftMax activation for probability distribution

## 🎯 Supported Diseases

1. Tomato Bacterial Spot
2. Tomato Early Blight
3. Tomato Late Blight
4. Tomato Leaf Mold
5. Tomato Septoria Leaf Spot
6. Tomato Spider Mites
7. Tomato Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Healthy Tomato Plants

## 🛠️ Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/srishtisahi/CropDiseases.git
cd CropDiseases
```

### 2. Install backend dependencies

```bash
cd api
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
```

### 4. Run the development servers

### Backend (from api directory)
python server.py

### Frontend (from frontend directory)
npm run dev

## 📈 Performance Metrics

- Training Accuracy: 86%
- Validation Accuracy: 81%
- Training Loss: 0.3
- Validation Loss: 0.6

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any queries or suggestions, please reach out to me on [LinkedIn](https://www.linkedin.com/in/srishtisahi/)

