import os
import base64

import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template_string
from insightface.app import FaceAnalysis


app = Flask(__name__)


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading age and gender model...")

model = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "genderage"],
    providers=["CPUExecutionProvider"]
)

model.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Model loaded successfully.")


# =========================================================
# WEB PAGE
# =========================================================

HTML = """
<!DOCTYPE html>
<html>

<head>

    <title>Age & Gender Detection</title>

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 30px 15px;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #222;
        }

        .container {
            max-width: 1200px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 3px 15px rgba(0, 0, 0, 0.08);
        }

        h1 {
            text-align: center;
            margin: 0 0 8px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }

        .workspace {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }

        .panel {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            background: #fafafa;
        }

        .panel h2 {
            margin: 0 0 12px;
            font-size: 18px;
            text-align: center;
        }

        .media-box {
            width: 100%;
            aspect-ratio: 16 / 9;
            background: #111;
            border-radius: 8px;
            overflow: hidden;

            display: flex;
            align-items: center;
            justify-content: center;
        }

        video,
        #preview,
        #resultImage {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: none;
        }

        video {
            background: black;
        }

        .placeholder {
            color: #aaa;
            text-align: center;
            padding: 20px;
        }

        canvas {
            display: none;
        }

        .upload-section {
            margin-bottom: 20px;
        }

        label {
            display: block;
            font-weight: bold;
            margin-bottom: 8px;
        }

        input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
            background: white;
        }

        .buttons {
            display: flex;
            gap: 10px;
            margin-top: 12px;
        }

        button {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
        }

        button:hover {
            opacity: 0.9;
        }

        .camera-button {
            background: #555;
            color: white;
        }

        .capture-button {
            background: #777;
            color: white;
        }

        .detect-button {
            width: 100%;
            background: #222;
            color: white;
            margin-top: 20px;
        }

        #message {
            text-align: center;
            margin-top: 15px;
            font-weight: bold;
        }

        #result {
            margin-top: 15px;
            padding: 12px;
            background: #f1f3f5;
            border-radius: 8px;
            white-space: pre-line;
            line-height: 1.7;
        }

        @media (max-width: 800px) {

            .workspace {
                grid-template-columns: 1fr;
            }

            .buttons {
                flex-direction: column;
            }

        }

    </style>

</head>


<body>

<div class="container">

    <h1>Age & Gender Detection</h1>

    <p class="subtitle">
        Upload an image or capture an image using your camera.
    </p>


    <!-- UPLOAD -->

    <div class="upload-section">

        <label for="file">
            Upload Image
        </label>

        <input
            type="file"
            id="file"
            accept="image/*"
            onchange="handleUpload()"
        >

    </div>


    <!-- INPUT + RESULT -->

    <div class="workspace">


        <!-- INPUT PANEL -->

        <div class="panel">

            <h2>Input</h2>

            <div class="media-box">

                <video
                    id="camera"
                    autoplay
                    playsinline
                ></video>

                <img
                    id="preview"
                    alt="Input image"
                >

                <div
                    id="inputPlaceholder"
                    class="placeholder"
                >
                    Upload an image or open the camera
                </div>

            </div>


            <div class="buttons">

                <button
                    class="camera-button"
                    onclick="startCamera()"
                >
                    Open Camera
                </button>

                <button
                    class="capture-button"
                    onclick="captureImage()"
                >
                    Capture Image
                </button>

            </div>

        </div>


        <!-- RESULT PANEL -->

        <div class="panel">

            <h2>Detection Result</h2>

            <div class="media-box">

                <img
                    id="resultImage"
                    alt="Detection result"
                >

                <div
                    id="resultPlaceholder"
                    class="placeholder"
                >
                    Detection result will appear here
                </div>

            </div>

            <div id="result"></div>

        </div>

    </div>


    <!-- DETECT BUTTON -->

    <button
        class="detect-button"
        onclick="detect()"
    >
        Detect Age & Gender
    </button>


    <p id="message"></p>


    <canvas id="canvas"></canvas>

</div>


<script>

let cameraStream = null;
let capturedImage = null;


// =========================================================
// START CAMERA
// =========================================================

async function startCamera() {

    try {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(function(track) {
                    track.stop();
                });

        }


        cameraStream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    facingMode: "user"
                },

                audio: false

            });


        const camera =
            document.getElementById("camera");

        const preview =
            document.getElementById("preview");

        const placeholder =
            document.getElementById("inputPlaceholder");


        camera.srcObject =
            cameraStream;

        camera.style.display =
            "block";


        preview.style.display =
            "none";

        placeholder.style.display =
            "none";


        document.getElementById("message").innerText =
            "Camera started.";

    }

    catch (error) {

        console.error(error);

        document.getElementById("message").innerText =
            "Camera could not be opened. Please allow camera permission.";

    }

}


// =========================================================
// CAPTURE IMAGE
// =========================================================

function captureImage() {

    if (!cameraStream) {

        alert(
            "Please click Open Camera first."
        );

        return;
    }


    const camera =
        document.getElementById("camera");

    const canvas =
        document.getElementById("canvas");

    const preview =
        document.getElementById("preview");

    const placeholder =
        document.getElementById("inputPlaceholder");


    if (
        camera.videoWidth === 0 ||
        camera.videoHeight === 0
    ) {

        alert(
            "Camera is not ready yet. Please wait a moment and try again."
        );

        return;

    }


    canvas.width =
        camera.videoWidth;

    canvas.height =
        camera.videoHeight;


    const context =
        canvas.getContext("2d");


    context.drawImage(
        camera,
        0,
        0,
        canvas.width,
        canvas.height
    );


    canvas.toBlob(

        function(blob) {

            if (!blob) {

                document.getElementById("message").innerText =
                    "Could not capture image.";

                return;

            }


            capturedImage =
                blob;


            preview.src =
                URL.createObjectURL(blob);


            preview.style.display =
                "block";


            placeholder.style.display =
                "none";


            // Stop camera after capture

            if (cameraStream) {

                cameraStream
                    .getTracks()
                    .forEach(function(track) {
                        track.stop();
                    });

            }


            camera.srcObject =
                null;

            camera.style.display =
                "none";

            cameraStream =
                null;


            document.getElementById("message").innerText =
                "Image captured successfully. Camera stopped.";

        },

        "image/jpeg",

        0.95

    );

}


// =========================================================
// UPLOAD IMAGE
// =========================================================

function handleUpload() {

    const file =
        document.getElementById("file");


    if (file.files.length === 0) {
        return;
    }


    capturedImage =
        null;


    const preview =
        document.getElementById("preview");

    const camera =
        document.getElementById("camera");

    const placeholder =
        document.getElementById("inputPlaceholder");


    // Stop camera if running

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(function(track) {
                track.stop();
            });

        cameraStream =
            null;

    }


    camera.srcObject =
        null;

    camera.style.display =
        "none";


    preview.src =
        URL.createObjectURL(
            file.files[0]
        );


    preview.style.display =
        "block";


    placeholder.style.display =
        "none";


    document.getElementById("message").innerText =
        "Image selected.";

}


// =========================================================
// DETECT
// =========================================================

async function detect() {

    const file =
        document.getElementById("file");


    const formData =
        new FormData();


    if (file.files.length > 0) {

        formData.append(
            "image",
            file.files[0]
        );

    }

    else if (capturedImage) {

        formData.append(
            "image",
            capturedImage,
            "camera.jpg"
        );

    }

    else {

        alert(
            "Please upload an image or capture an image."
        );

        return;

    }


    document.getElementById("message").innerText =
        "Detecting...";


    document.getElementById("result").innerText =
        "";


    document.getElementById("resultImage").style.display =
        "none";


    document.getElementById("resultPlaceholder").style.display =
        "flex";


    try {

        const response =
            await fetch(
                "/detect",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (data.image) {

            const resultImage =
                document.getElementById("resultImage");

            const resultPlaceholder =
                document.getElementById(
                    "resultPlaceholder"
                );


            resultImage.src =
                "data:image/jpeg;base64," +
                data.image;


            resultImage.style.display =
                "block";


            resultPlaceholder.style.display =
                "none";

        }


        document.getElementById("result").innerText =
            data.text || "No result received.";


        document.getElementById("message").innerText =
            "Detection completed.";

    }

    catch (error) {

        console.error(error);

        document.getElementById("message").innerText =
            "Could not connect to the server.";

    }

}

</script>

</body>

</html>
"""


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template_string(HTML)


# =========================================================
# DETECTION
# =========================================================

@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:

        return jsonify({
            "text": "No image received."
        }), 400


    file = request.files["image"]

    image_bytes = file.read()


    if not image_bytes:

        return jsonify({
            "text": "Empty image."
        }), 400


    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )


    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        return jsonify({
            "text": "Could not read the image."
        }), 400


    try:

        faces = model.get(frame)

        results = []


        for index, face in enumerate(
            faces,
            start=1
        ):

            # -------------------------------------------------
            # Face coordinates
            # -------------------------------------------------

            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )


            # -------------------------------------------------
            # Age
            # -------------------------------------------------

            age = int(
                round(
                    float(face.age)
                )
            )


            # -------------------------------------------------
            # Gender
            #
            # InsightFace:
            # 0 = Female
            # 1 = Male
            # -------------------------------------------------

            raw_gender = int(face.gender)

            if raw_gender == 1:
                gender = "Male"
            else:
                gender = "Female"


            # -------------------------------------------------
            # Debug information
            #
            # This helps verify what the model actually predicts.
            # It is printed in the terminal, not shown to users.
            # -------------------------------------------------

            print(
                f"Person {index}: "
                f"raw_gender={raw_gender}, "
                f"predicted_gender={gender}, "
                f"age={age}"
            )


            # -------------------------------------------------
            # Draw bounding box
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # -------------------------------------------------
            # Draw label
            # -------------------------------------------------

            label = (
                f"{gender}, Age: {age}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            # -------------------------------------------------
            # Result text
            # -------------------------------------------------

            results.append(
                f"Person {index}: "
                f"{gender}, Age = {age}"
            )


        # -----------------------------------------------------
        # No face
        # -----------------------------------------------------

        if not faces:

            results.append(
                "No face detected."
            )


        # -----------------------------------------------------
        # Encode result image
        # -----------------------------------------------------

        success, encoded = cv2.imencode(
            ".jpg",
            frame
        )


        if not success:

            return jsonify({
                "text": "Could not create result image."
            }), 500


        image_base64 = base64.b64encode(
            encoded.tobytes()
        ).decode("utf-8")


        return jsonify({

            "image": image_base64,

            "text": "\n".join(results)

        })


    except Exception as error:

        print(
            "Detection error:",
            error
        )


        return jsonify({

            "text":
                "An error occurred during detection."

        }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )


    app.run(
        host="0.0.0.0",
        port=port
    )

