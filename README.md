# Age_and_Gender_Detection_Using_CNN

A beginner-friendly Computer Vision and Deep Learning project that detects human faces from a live webcam and predicts **Age** and **Gender** in real time using pretrained deep learning models.

## Technologies Used

- Python
- OpenCV
- InsightFace
- ONNX Runtime
- NumPy

The detected predictions are displayed directly on webcam frames with bounding boxes and labels.

---

# Features

- Real-time webcam face detection
- Predicts:
  - Age
  - Gender
- Displays results directly on webcam frames
- Lightweight and beginner-friendly
- Runs on CPU (No GPU required)
- Clean and simple code structure
- Easy to explain during viva

---

# Technologies Purpose

| Technology | Purpose |
|------------|----------|
| Python | Main programming language |
| OpenCV | Webcam handling and drawing |
| InsightFace | Face analysis and prediction |
| ONNX Runtime | Running pretrained ONNX models |
| NumPy | Array and image processing |

---

# Project Structure

```text
Age-and-Gender-Detection-Using-CNN/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Varshabachu050/CNN-based-age-and-gender-detection-system.git
```

---

## 2. Open Project Folder

```bash
cd CNN-based-age-and-gender-detection-system
```

---

## 3. Install Required Libraries

```bash
pip install insightface onnxruntime opencv-python numpy
```

---

# How to Run

Run the Python file:

```bash
python main.py
```

The webcam will open automatically.

Press:

```text
Q
```

to quit the application.

---

# How the Project Works

1. OpenCV captures live webcam video.
2. InsightFace detects human faces.
3. Pretrained ONNX deep learning models predict:
   - Age
   - Gender
4. Results are displayed directly on webcam frames.

---

# Output

The application displays:

- Face bounding boxes
- Predicted age
- Predicted gender

---

# Requirements

- Python 3.9+
- Webcam
- Internet connection (only first run for model download)

---

# License

This project is open-source and free to use for educational purposes.

--- 

# Author

Developed as a beginner-friendly mini project for Computer Science Engineering students - Bachu Varsha



