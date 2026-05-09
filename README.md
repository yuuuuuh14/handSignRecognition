# Hand Sign Recognition 🖐️

A real-time Korean Sign Language (KSL) translation web application. This project uses a PC webcam to recognize hand gestures and translates them into Korean text on the screen.

## Features

- **Real-Time Translation:** Translates sign language into text instantly via live webcam feed.
- **Machine Learning Integration:** Utilizes a pre-trained PyTorch KSLRNet model combined with Google MediaPipe for robust hand and face landmark detection.
- **Modern Web Interface:** Built with a sleek, dark-mode first design using React and Tailwind CSS for optimal accessibility and user experience.
- **WebSocket Streaming:** Uses high-performance WebSockets to stream image frames from the frontend to the backend for low-latency inference.

## Supported Vocabulary

The model currently recognizes the following 10 words:
1. 안녕하세요 (Hello)
2. 감사합니다 (Thank you)
3. 사랑 (Love)
4. 학교 (School)
5. 친구 (Friend)
6. 물 (Water)
7. 밥 (Rice/Meal)
8. 가다 (Go)
9. 오다 (Come)
10. 끝 (End)

## Tech Stack

- **Frontend:** React 18, Vite, Tailwind CSS v4, Radix UI Primitives, Lucide React (Icons).
- **Backend:** Python 3.11, FastAPI, Uvicorn, WebSockets.
- **ML & Computer Vision:** PyTorch, OpenCV, MediaPipe.

---

## 🚀 How to Run

To run this project locally, you need to start both the **Backend ML Server** and the **Frontend Web App**.

### 1. Start the Backend Server

The backend requires Python (3.10+ recommended).

```powershell
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt fastapi uvicorn websockets python-multipart

# Start the FastAPI WebSocket server
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```
*Note: The backend runs on `http://127.0.0.1:8000` and listens for WebSocket connections at `/ws/{session_id}`.*

### 2. Start the Frontend App

The frontend requires Node.js.

```powershell
# Open a new terminal window and navigate to the frontend directory
cd frontend

# Install Node modules
npm install

# Start the Vite development server
npm run dev
```

### 3. Start Translating!
- Open your browser and navigate to the local URL provided by Vite (usually `http://localhost:5173`).
- Click the **"Turn Camera On"** button and grant webcam permissions.
- Make one of the supported hand gestures in front of the camera, and the translated text will appear on the screen!

## Project Structure

```text
handSignRecognition/
├── backend/            # Python FastAPI server & KSLR PyTorch Model
│   ├── models/         # Neural network architecture definitions
│   ├── realtime/       # Webcam processing pipeline
│   ├── runs/           # Pre-trained model weights (best.pt)
│   └── server.py       # WebSocket entry point
├── frontend/           # React web application
│   ├── public/         # Static assets
│   ├── src/
│   │   ├── app/        # Main application components & hooks
│   │   └── styles/     # Tailwind CSS configuration
│   └── package.json    # Frontend dependencies
└── README.md           # Project documentation
```

## Credits & References
- The machine learning model architecture and training pipeline were heavily inspired by the [KSLR (Korean Sign Language Recognition) repository](https://github.com/quirinal36/handSignRecognition).
