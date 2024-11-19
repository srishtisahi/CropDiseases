'use client'

import { useState, useRef } from 'react'
import { UploadCloud, AlertCircle } from 'lucide-react'

interface PredictionResult {
  disease: string
  confidence: number
  predictions: { [key: string]: number }
}

export default function CropDiseaseDetector() {
  const [file, setFile] = useState<File | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0])
      setPrediction(null)
      setError(null)
    }
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
      setFile(event.dataTransfer.files[0])
      setPrediction(null)
      setError(null)
    }
  }

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
  }

  const predictDisease = async () => {
    if (!file) return

    setIsLoading(true)
    setError(null)

    try {
      // Create a FileReader to read the image file
      const reader = new FileReader()
      
      reader.onload = async (e) => {
        try {
          // Create an image element to load the file data
          const img = new Image()
          img.src = e.target?.result as string
          
          await new Promise((resolve) => {
            img.onload = resolve
          })

          // Create a canvas to get image data
          const canvas = document.createElement('canvas')
          canvas.width = 224
          canvas.height = 224
          const ctx = canvas.getContext('2d')
          
          if (!ctx) throw new Error('Could not get canvas context')
          
          // Draw and resize image to match model requirements
          ctx.drawImage(img, 0, 0, 224, 224)
          
          // Get image data as array
          const imageData = ctx.getImageData(0, 0, 224, 224)
          const matrix = new Array(224).fill(0).map(() => 
            new Array(224).fill(0).map(() => new Array(3).fill(0))
          )
          
          // Convert image data to 3D matrix
          for (let i = 0; i < imageData.data.length; i += 4) {
            const row = Math.floor((i / 4) / 224)
            const col = (i / 4) % 224
            matrix[row][col] = [
              imageData.data[i],     // R
              imageData.data[i + 1], // G
              imageData.data[i + 2]  // B
            ]
          }

          // Send matrix to API
          const response = await fetch('https://cropdiseases.onrender.com/predict', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            body: JSON.stringify({ matrix }),
          })

          if (!response.ok) {
            throw new Error('Failed to get prediction')
          }

          const result: PredictionResult = await response.json()
          setPrediction(result)
        } catch (err) {
          setError('An error occurred while processing the image. Please try again.')
          console.error(err)
        } finally {
          setIsLoading(false)
        }
      }

      reader.onerror = () => {
        setError('Error reading the image file')
        setIsLoading(false)
      }

      reader.readAsDataURL(file)
    } catch (err) {
      setError('An error occurred while predicting. Please try again.')
      console.error(err)
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-100 to-yellow-100">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-green-800 mb-2">FasaL</h1>
          <p className="text-xl text-green-600">Detect crop diseases in tomatoes</p>
        </header>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div
            className="border-2 border-dashed border-green-300 rounded-lg p-8 text-center cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              className="hidden"
            />
            <UploadCloud className="mx-auto h-12 w-12 text-green-400 mb-4" />
            <p className="text-green-600 mb-2">Drag and drop your image here, or click to select</p>
            {file && <p className="text-sm text-green-500">{file.name}</p>}
          </div>

          {file && (
            <button
              onClick={predictDisease}
              disabled={isLoading}
              className="mt-4 w-full bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded transition duration-300 ease-in-out"
            >
              {isLoading ? 'Predicting...' : 'Predict Disease'}
            </button>
          )}
        </div>

        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-8" role="alert">
            <div className="flex">
              <AlertCircle className="h-6 w-6 mr-2" />
              <p>{error}</p>
            </div>
          </div>
        )}

        {prediction && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-green-800 mb-4">Prediction Results</h2>
            <p className="text-lg mb-2">
              <span className="font-semibold text-black">Disease:</span>{' '}
              <span className="text-black">{prediction.disease}</span>
            </p>
            <p className="text-lg mb-4">
              <span className="font-semibold text-black">Confidence:</span>{' '}
              <span className="text-black">{(prediction.confidence * 100).toFixed(2)}%</span>
            </p>
            <h3 className="text-xl font-semibold text-green-700 mb-2">All Predictions:</h3>
            <ul className="space-y-1">
              {Object.entries(prediction.predictions).map(([disease, probability]) => (
                <li key={disease} className="flex justify-between text-black">
                  <span>{disease}:</span>
                  <span>{(probability * 100).toFixed(2)}%</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}