import os

# Must be set BEFORE importing numpy / cv2 / onnxruntime (keeps RAM low)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import base64
import gc
import shutil
import urllib.request
import zipfile

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template_string
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

app = Flask(__name__)

# Reject uploads bigger than 4 MB
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

# Photos larger than this (pixels, longest side) are shrunk before detection
MAX_IMAGE_SIDE = 640

# Detector input size. 320 = lowest RAM. Raise to (480, 480) or (640, 640)
# if you have more memory and need to detect small faces.
DET_SIZE = (320, 320)


# =========================================================
# LOAD MODEL (low-memory settings)
# =========================================================

print("Loading age and gender model...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, ".insightface", "models", "buffalo_s")
NEEDED_MODELS = ["det_500m.onnx", "genderage.onnx"]
MODEL_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
)


def ensure_models():
    """Make sure ONLY the 2 needed model files exist (skips the other 3 big ones)."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if all(os.path.exists(os.path.join(MODEL_DIR, n)) for n in NEEDED_MODELS):
        return

    zip_path = os.path.join(MODEL_DIR, "buffalo_s.zip")
    print("Downloading model pack (one time)...")
    urllib.request.urlretrieve(MODEL_URL, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            name = os.path.basename(member)
            if name in NEEDED_MODELS:
                with z.open(member) as src, open(
                    os.path.join(MODEL_DIR, name), "wb"
                ) as out:
                    shutil.copyfileobj(src, out)

    os.remove(zip_path)


ensure_models()

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.enable_cpu_mem_arena = False   # don't hoard RAM
sess_options.enable_mem_pattern = False
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

_ort_kwargs = dict(providers=["CPUExecutionProvider"], sess_options=sess_options)

detector = model_zoo.get_model(
    os.path.join(MODEL_DIR, "det_500m.onnx"), **_ort_kwargs
)
detector.prepare(ctx_id=-1, input_size=DET_SIZE, det_thresh=0.5)

genderage = model_zoo.get_model(
    os.path.join(MODEL_DIR, "genderage.onnx"), **_ort_kwargs
)
genderage.prepare(ctx_id=-1)


def analyze(frame):
    """Detect faces and predict (bbox, gender, age) for each one."""
    bboxes, kpss = detector.detect(frame, max_num=0, metric="default")
    found = []
    if bboxes is None or bboxes.shape[0] == 0:
        return found
    for i in range(bboxes.shape[0]):
        face = Face(
            bbox=bboxes[i, 0:4],
            kps=kpss[i] if kpss is not None else None,
            det_score=bboxes[i, 4],
        )
        genderage.get(frame, face)   # fills face.gender and face.age
        found.append(face)
    return found


print("Model loaded successfully.")


# =========================================================
# WEB PAGE
# =========================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Age & Gender Detection</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* { box-sizing: border-box; }
body { margin: 0; padding: 30px 15px; font-family: Arial, sans-serif; background: #f4f6f8; color: #222; }
.container { max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 3px 15px rgba(0,0,0,0.08); }
h1 { text-align: center; margin: 0 0 8px; }
.subtitle { text-align: center; color: #666; margin-bottom: 30px; }
.workspace { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
.panel { border: 1px solid #ddd; border-radius: 10px; padding: 15px; background: #fafafa; }
.panel h2 { margin: 0 0 12px; font-size: 18px; text-align: center; }
.media-box { width: 100%; aspect-ratio: 16 / 9; background: #111; border-radius: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; }
video, #preview, #resultImage { width: 100%; height: 100%; object-fit: contain; display: none; }
video { background: black; }
.placeholder { color: #aaa; text-align: center; padding: 20px; }
canvas { display: none; }
.upload-section { margin-bottom: 20px; }
label { display: block; font-weight: bold; margin-bottom: 8px; }
input[type="file"] { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; background: white; }
.buttons { display: flex; gap: 10px; margin-top: 12px; }
button { flex: 1; padding: 12px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; }
button:hover { opacity: 0.9; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.camera-button { background: #555; color: white; }
.capture-button { background: #777; color: white; }
.detect-button { width: 100%; background: #222; color: white; margin-top: 20px; }
#message { text-align: center; margin-top: 15px; font-weight: bold; }
#result { margin-top: 15px; padding: 12px; background: #f1f3f5; border-radius: 8px; white-space: pre-line; line-height: 1.7; }
@media (max-width: 800px) {
  .workspace { grid-template-columns: 1fr; }
  .buttons { flex-direction: column; }
}
</style>
</head>
<body>
<div class="container">
  <h1>Age & Gender Detection</h1>
  <p class="subtitle">Upload an image or capture an image using your camera.</p>

  <div class="upload-section">
    <label for="file">Upload Image</label>
    <input type="file" id="file" accept="image/*" onchange="handleUpload()">
  </div>

  <div class="workspace">
    <div class="panel">
      <h2>Input</h2>
      <div class="media-box">
        <video id="camera" autoplay playsinline></video>
        <img id="preview" alt="Input image">
        <div id="inputPlaceholder" class="placeholder">Upload an image or open the camera</div>
      </div>
      <div class="buttons">
        <button class="camera-button" onclick="startCamera()">Open Camera</button>
        <button class="capture-button" onclick="captureImage()">Capture Image</button>
      </div>
    </div>

    <div class="panel">
      <h2>Detection Result</h2>
      <div class="media-box">
        <img id="resultImage" alt="Detection result">
        <div id="resultPlaceholder" class="placeholder">Detection result will appear here</div>
      </div>
      <div id="result"></div>
    </div>
  </div>

  <button id="detectBtn" class="detect-button" onclick="detect()">Detect Age & Gender</button>
  <p id="message"></p>
  <canvas id="canvas"></canvas>
</div>

<script>
let cameraStream = null;
let capturedImage = null;
const $ = (id) => document.getElementById(id);

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  $("camera").srcObject = null;
  $("camera").style.display = "none";
}

async function startCamera() {
  try {
    stopCamera();
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" }, audio: false
    });
    $("camera").srcObject = cameraStream;
    $("camera").style.display = "block";
    $("preview").style.display = "none";
    $("inputPlaceholder").style.display = "none";
    $("message").innerText = "Camera started.";
  } catch (e) {
    console.error(e);
    $("message").innerText = "Camera could not be opened. Please allow camera permission.";
  }
}

function captureImage() {
  if (!cameraStream) { alert("Please click Open Camera first."); return; }
  const cam = $("camera"), canvas = $("canvas");
  if (!cam.videoWidth || !cam.videoHeight) {
    alert("Camera is not ready yet. Please wait a moment and try again.");
    return;
  }
  canvas.width = cam.videoWidth;
  canvas.height = cam.videoHeight;
  canvas.getContext("2d").drawImage(cam, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    if (!blob) { $("message").innerText = "Could not capture image."; return; }
    capturedImage = blob;
    $("file").value = "";
    $("preview").src = URL.createObjectURL(blob);
    $("preview").style.display = "block";
    $("inputPlaceholder").style.display = "none";
    stopCamera();
    $("message").innerText = "Image captured successfully. Camera stopped.";
  }, "image/jpeg", 0.9);
}

function handleUpload() {
  const f = $("file");
  if (f.files.length === 0) return;
  capturedImage = null;
  stopCamera();
  $("preview").src = URL.createObjectURL(f.files[0]);
  $("preview").style.display = "block";
  $("inputPlaceholder").style.display = "none";
  $("message").innerText = "Image selected.";
}

async function detect() {
  const f = $("file");
  const formData = new FormData();

  if (f.files.length > 0) formData.append("image", f.files[0]);
  else if (capturedImage) formData.append("image", capturedImage, "camera.jpg");
  else { alert("Please upload an image or capture an image."); return; }

  $("message").innerText = "Detecting... (first request may take a few seconds)";
  $("result").innerText = "";
  $("resultImage").style.display = "none";
  $("resultPlaceholder").style.display = "flex";
  $("detectBtn").disabled = true;

  try {
    const response = await fetch("/detect", { method: "POST", body: formData });
    const data = await response.json();

    if (data.image) {
      $("resultImage").src = "data:image/jpeg;base64," + data.image;
      $("resultImage").style.display = "block";
      $("resultPlaceholder").style.display = "none";
    }
    $("result").innerText = data.text || "No result received.";
    $("message").innerText = response.ok ? "Detection completed." : "Detection failed.";
  } catch (e) {
    console.error(e);
    $("message").innerText = "Could not connect to the server.";
  } finally {
    $("detectBtn").disabled = false;
  }
}
</script>
</body>
</html>
"""


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return "ok", 200


@app.errorhandler(413)
def too_large(_):
    return jsonify({"text": "Image is too large. Maximum size is 4 MB."}), 413


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"text": "No image received."}), 400

    image_bytes = request.files["image"].read()
    if not image_bytes:
        return jsonify({"text": "Empty image."}), 400

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    del image_bytes

    if frame is None:
        return jsonify({"text": "Could not read the image."}), 400

    # Shrink large images to save memory
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    try:
        faces = analyze(frame)
        results = []

        for index, face in enumerate(faces, start=1):
            x1, y1, x2, y2 = face.bbox.astype(int)
            age = int(round(float(face.age)))

            # InsightFace: 0 = Female, 1 = Male
            gender = "Male" if int(face.gender) == 1 else "Female"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{gender}, Age: {age}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            results.append(f"Person {index}: {gender}, Age = {age}")

        if not faces:
            results.append("No face detected.")

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return jsonify({"text": "Could not create result image."}), 500

        payload = {
            "image": base64.b64encode(encoded.tobytes()).decode("utf-8"),
            "text": "\n".join(results),
        }
        return jsonify(payload)

    except Exception as error:
        print("Detection error:", error)
        return jsonify({"text": "An error occurred during detection."}), 500

    finally:
        del frame
        gc.collect()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
