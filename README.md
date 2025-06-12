# Intelligent Traffic Control System (ITCS)

An AI-powered traffic management system that uses YOLOv8 computer vision to detect emergency vehicles (ambulances) in real-time video streams and automatically adjusts traffic signals to prioritize emergency response.

## 🚨 Features

- **Real-time Emergency Vehicle Detection**: Custom-trained YOLOv8 model for ambulance detection
- **Intelligent Traffic Signal Control**: Automatic traffic light management based on emergency vehicle presence
- **Web-based Dashboard**: Modern React.js interface for video upload and analysis
- **High Accuracy Detection**: Confidence-based detection with adjustable thresholds
- **Video Processing**: Support for multiple video formats (MP4, AVI, MOV, etc.)
- **Real-time Feedback**: Live traffic light status with confidence scores

## 🏗️ Architecture

```
Detection_yolov8/
├── smart-traffic-backend/     # FastAPI backend server
│   ├── main.py               # Main API server with YOLO integration
│   ├── weights/              # Trained model weights
│   │   ├── best.pt          # Custom YOLOv8 model for ambulance detection
│   │   └── best_pt.zip      # Backup model weights
│   ├── requirements.txt      # Python dependencies
│   └── venv/                # Virtual environment
├── traffic-ui/               # React.js frontend
│   ├── src/
│   │   ├── App.js           # Main application component
│   │   ├── App.css          # Styling and UI components
│   │   └── ...
│   ├── public/              # Static assets
│   └── package.json         # Node.js dependencies
└── test_files/              # Sample video files for testing
    ├── ambulance_video1.mp4
    └── car_sample2_yolo.mp4
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **YOLOv8 (Ultralytics)**: State-of-the-art object detection model
- **OpenCV**: Computer vision and video processing
- **Python 3.11+**: Core programming language

### Frontend
- **React.js 19.1.0**: Modern JavaScript UI framework
- **Axios**: HTTP client for API communication
- **Tailwind CSS**: Utility-first CSS framework
- **Modern ES6+**: JavaScript features

### AI/ML
- **Custom YOLOv8 Model**: Trained specifically for ambulance detection
- **Computer Vision Pipeline**: Real-time video frame analysis
- **Confidence Scoring**: Adjustable detection thresholds

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11 or higher
- Node.js 16 or higher
- npm or yarn package manager

### Backend Setup

1. **Navigate to backend directory**:
```bash
cd smart-traffic-backend
```

2. **Create and activate virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

4. **Start the FastAPI server**:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
```bash
cd traffic-ui
```

2. **Install Node.js dependencies**:
```bash
npm install
```

3. **Start the React development server**:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## 🎯 Usage

1. **Start both servers** (backend on port 8000, frontend on port 3000)
2. **Open the web interface** at `http://localhost:3000`
3. **Upload a traffic video** using the file upload interface
4. **Click "Analyze Video"** to process the video
5. **View results** on the traffic light dashboard:
   - 🔴 **RED**: No ambulance detected - normal traffic flow
   - 🟢 **GREEN**: Ambulance detected - emergency priority mode

## 📊 API Endpoints

### POST `/predict`
Analyzes uploaded video for ambulance detection.

**Request**: 
- `file`: Video file (multipart/form-data)

**Response**:
```json
{
  "ambulance": true/false,
  "frame": "base64_encoded_image",
  "confidence": 0.95
}
```

## ⚙️ Configuration

### Detection Parameters
- **Confidence Threshold**: `0.75` (adjustable in `main.py`)
- **Minimum Detection Frames**: `1` frame required for positive detection
- **Maximum Processing Frames**: `100` frames per video
- **Debug Mode**: Available for testing and development

### Model Configuration
- **Model Path**: `weights/best.pt`
- **Class Mapping**: `{0: 'ambulance'}`
- **Input Format**: Video frames (RGB)

## 🧪 Testing

### Sample Videos
The `test_files/` directory contains sample videos for testing:
- `ambulance_video1.mp4`: Contains ambulance for positive detection testing
- `car_sample2_yolo.mp4`: Regular traffic for negative detection testing

### Running Tests
```bash
# Backend tests
cd smart-traffic-backend
python -m pytest

# Frontend tests
cd traffic-ui
npm test
```

## 🔧 Development

### Debug Mode
Enable debug mode in `main.py` by setting `DEBUG_MODE = True` for enhanced logging and forced detection capabilities.

### Model Training
The YOLOv8 model was custom-trained for ambulance detection. To retrain:
1. Prepare annotated dataset
2. Use Ultralytics YOLOv8 training pipeline
3. Replace `weights/best.pt` with new model

### Adding New Vehicle Types
1. Update `CLASS_MAPPING` in `main.py`
2. Retrain model with new classes
3. Update frontend UI accordingly

## 🚦 Traffic Light Logic

- **GREEN (Emergency Mode)**: 
  - Ambulance detected with confidence ≥ 75%
  - Traffic signals prioritize emergency vehicle passage
  - Clear path instructions displayed

- **RED (Normal Mode)**:
  - No emergency vehicles detected
  - Normal traffic flow maintained
  - Standard traffic signal timing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

**ITCS Development Team**
- AI/ML Engineering
- Full-Stack Development
- Computer Vision Specialists

## 🆘 Support

For technical support or questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation for troubleshooting

## 🔮 Future Enhancements

- [ ] Multi-vehicle type detection (fire trucks, police cars)
- [ ] Real-time traffic camera integration
- [ ] Mobile application development
- [ ] Cloud deployment and scaling
- [ ] Advanced traffic optimization algorithms
- [ ] Integration with city traffic management systems

---

**© 2024 ITCS - Intelligent Traffic Control System**
